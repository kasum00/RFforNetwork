"""Explicit user-driven queue/monitor; no automatic fit, retry or method switching."""
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from .common import Blocked, METHODS, files_checked, paths, read, utc
from .controller import launch


def metadata(source, dataset, configs):
    reference = read(Path(source) / "phuong_anB_step2/evidence/dataset_reference.json")
    output = {}
    for config in configs:
        expected = reference["configs"][config]
        folder = Path(dataset) / config
        files_checked(folder, expected["sidecars"])
        integrity = read(folder / "window_integrity.json")
        if integrity["windows_by_split"] != expected["windows_by_split"]:
            raise Blocked("Frozen counts differ: " + config)
        output[config] = dict(status="METADATA_HASHES_VERIFIED_NOT_SELECTION_VERIFIED",
            rows=expected["windows_by_split"], test_file_opened=False)
    return output


def start_queue(requests):
    if not requests or len({r["method"] for r in requests}) != 1:
        raise Blocked("Queue must contain explicitly selected configs of ONE method")
    requests = json.loads(json.dumps(requests))
    state = dict(running=True, error=None, current=None, current_started=None, completed=[], requests=requests)
    def work():
        try:
            for request in requests:
                state["current_started"] = utc()
                state["current"] = request
                value = launch(request)
                if value.get("native_exit_code") != 0 or value.get("status") != "PASS":
                    raise Blocked(str(value))
                state["completed"].append(dict(method=request["method"], config=request["config"], result=value))
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["running"] = False
    thread = threading.Thread(target=work, daemon=True)
    state["thread"] = thread
    thread.start()
    return state


def monitor(job, interval=15):
    from IPython.display import clear_output
    from tqdm.auto import tqdm
    bar = tqdm(total=len(job["requests"]), desc="Completed configurations", unit="config")
    try:
        while True:
            bar.update(max(0, len(job["completed"])-bar.n))
            request = job["current"]
            if request:
                _, remote = paths(request)
                path = remote / "heartbeat.json"
                if path.is_file():
                    value = read(path)
                    identity = read(remote / "identity_manifest.json")
                    if (value["identity"] != identity["sha256"] or identity["run_id"] != request["run_id"]
                            or identity["selection_run_id"] != request["selection_run_id"]
                            or value["utc"] < job["current_started"]):
                        print(request["method"], request["config"], "PREPARING/VERIFYING — saved heartbeat belongs to an older attempt", flush=True)
                    else:
                        age = (datetime.now(timezone.utc)-datetime.fromisoformat(value["utc"])).total_seconds()
                        print(request["method"], request["config"], "|", value["stage"], "|", value["status"],
                            "| elapsed", round(value["elapsed_seconds"]), "s | CPU", value["cpu_percent"],
                            "% | RAM", round(value["ram_bytes"]/1024**3, 2), "GiB | checkpoint", value["last_checkpoint"],
                            "| heartbeat age", round(age), "s | error", value["error"], flush=True)
                else:
                    print(request["method"], request["config"], "WAITING_FOR_HEARTBEAT; no invented fit progress", flush=True)
            if not job["running"]:
                print("QUEUE FINISHED | ERROR:", job["error"], flush=True)
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitor stopped; worker may still be running. Do not start a second queue or change source.")
    finally:
        bar.close()


def safe_launch(request):
    value = launch(request)
    if value.get("native_exit_code") != 0 or value.get("status") != "PASS":
        raise Blocked(str(value))
    return value
