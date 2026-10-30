"""Notebook queue, read-only monitor and safe stop; numerical work stays in workers."""
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from .contract import ALGORITHM, Blocked, paths, read, redact, utc
from .controller import launch


def planned_action(request):
    _, remote = paths(request)
    if (remote / "results/SELECTION_COMPLETE.json").is_file():
        return "verify"
    return "resume" if remote.exists() else "run"


def build_plan(requests):
    plan = []
    for base in requests:
        action = planned_action(base)
        result = launch({**base, "mode": "preflight"})
        _, remote = paths(base)
        action_ok = not (action == "run" and remote.exists())
        action_ok = action_ok and not (action in ("resume", "verify") and not remote.exists())
        ready = action_ok and result["native_exit_code"] == 0 and result["status"] == "PASS"
        plan.append({"request": {**base, "mode": action}, "preflight": result, "ready": ready})
        print(base["config"], base["run_id"], action, "READY" if ready else "BLOCKED")
        print("REASONS:", result.get("reasons", result.get("message")))
        print("UNRELEASED LEASE:", result.get("unreleased_lease"))
        print("LAST CHECKPOINT:", result.get("last_checkpoint"))
    return plan


class Queue:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = {"running": False, "current_request": None, "results": [],
                      "error": None, "stop_queue": False}
        self.thread = None

    def start(self, requests):
        requests = json.loads(json.dumps(requests))
        with self.lock:
            if self.state["running"]:
                raise Blocked("B queue is already running")
            self.state.update(running=True, current_request=None, results=[], error=None, stop_queue=False)
        def work():
            try:
                for request in requests:
                    with self.lock:
                        if self.state["stop_queue"]:
                            break
                        self.state["current_request"] = request
                    result = launch(request)
                    self.state["results"].append({"config": request["config"], "run_id": request["run_id"],
                                                  "action": request["mode"], "result": result})
                    if result["native_exit_code"] != 0 or result["status"] not in ("PASS", "PAUSED"):
                        raise Blocked(json.dumps(result, ensure_ascii=False))
                    if result["status"] == "PAUSED":
                        break
                    if result.get("phase_status") != "SELECTION_COMPLETE":
                        raise Blocked("Worker has not reported completed selection")
                    if request["mode"] != "verify":
                        verify_request = {**request, "mode": "verify"}
                        with self.lock:
                            self.state["current_request"] = verify_request
                        verified = launch(verify_request)
                        self.state["results"].append({"config": request["config"], "run_id": request["run_id"],
                                                     "action": "verify", "result": verified})
                        if (verified["native_exit_code"] != 0 or verified["status"] != "PASS"
                                or verified.get("phase_status") != "SELECTION_COMPLETE"):
                            raise Blocked("Verification failed: " + json.dumps(verified, ensure_ascii=False))
            except BaseException as exc:
                self.state["error"] = redact(f"{type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    self.state["running"], self.state["current_request"] = False, None
                print("B queue ended; review result/error. No automatic retry.", flush=True)
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def safe_stop(self):
        with self.lock:
            current = self.state["current_request"]
            if not self.state["running"] or not current:
                raise Blocked("No worker owned by this queue")
            self.state["stop_queue"] = True
            if current["mode"] == "verify":
                return "Verify is finishing; no next config will start"
            local, remote = paths(current)
            if (remote / "results/SELECTION_COMPLETE.json").is_file():
                return "Selection finished; waiting for verify, then queue stops"
            local.mkdir(parents=True, exist_ok=True)
            path = local / "STOP_REQUESTED"
            path.write_text(utc(), encoding="utf-8")
            return "REQUESTED LOCALLY: " + str(path) + "; wait for COMMITTED and PAUSED"

    def stop_after_config(self):
        with self.lock:
            if not self.state["running"]:
                raise Blocked("No running queue")
            self.state["stop_queue"] = True
            return "Current config may finish and verify; no next config will start"


def read_display(path, warnings):
    try:
        if path.is_file():
            return read(path)
    except (OSError, ValueError) as exc:
        warnings.append(str(path) + ": " + str(exc))
    return None


def monitor_snapshot(request):
    local, remote = paths(request)
    warnings = []
    snapshot = {"local": local, "remote": remote, "warnings": warnings,
                "identity": None, "heartbeat": None, "time": None, "path": None}
    manifest = read_display(remote / "run_manifest.json", warnings)
    if manifest is None:
        manifest = read_display(local / "run_manifest.json", warnings)
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if (not isinstance(identity, dict) or identity.get("algorithm") != ALGORITHM
            or identity.get("rfe_step") != 2 or identity.get("config") != request["config"]
            or identity.get("run_id") != request["run_id"]):
        return snapshot
    snapshot["identity"] = identity
    choices = []
    for root in (local, remote):
        path = root / "heartbeat.json"
        row = read_display(path, warnings)
        if not isinstance(row, dict):
            continue
        if (row.get("identity") != identity["sha256"] or row.get("algorithm") != ALGORITHM
                or row.get("rfe_step") != 2 or row.get("config") != request["config"]
                or row.get("run_id") != request["run_id"]):
            warnings.append("Rejected foreign heartbeat: " + str(path))
            continue
        try:
            timestamp = datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("Missing heartbeat timezone")
            choices.append((timestamp, path, row))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(str(exc))
    if choices:
        timestamp, path, row = max(choices, key=lambda item: item[0])
        snapshot.update(time=timestamp, path=path, heartbeat=row)
    return snapshot


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "--"
    hours, rest = divmod(int(value), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def monitor(requests, queue=None, *, refresh_seconds=15, max_refresh=0):
    """tqdm shows whole durable units only; heartbeat supplies elapsed and measured ETA."""
    from IPython.display import clear_output
    from tqdm.auto import tqdm
    if type(refresh_seconds) is not int or refresh_seconds < 5:
        raise ValueError("Monitor interval must be >=5 seconds")
    requests = json.loads(json.dumps(requests))
    count = 0
    try:
        while max_refresh == 0 or count < max_refresh:
            count += 1
            clear_output(wait=True)
            job = queue.state if queue is not None else {}
            print("B STEP2 LIVE MONITOR — READ ONLY", utc())
            print("JOB:", {key: job.get(key) for key in ("running", "error", "stop_queue")})
            for request in requests:
                snapshot = monitor_snapshot(request)
                hb = snapshot["heartbeat"]
                label = request["config"] + "/" + request["run_id"]
                if hb is None:
                    print(label, "WAITING / NO VALID HEARTBEAT")
                    for warning in snapshot["warnings"]:
                        print("READ WARNING:", warning)
                    continue
                done, total = hb.get("phase_completed_durable"), hb.get("phase_total")
                if type(done) is not int or type(total) is not int or not 0 <= done <= total:
                    print(label, "INVALID PROGRESS REPORT; no percentage inferred")
                    continue
                best = hb.get("best_validation_accuracy")
                accuracy = f"{best:.8f}" if type(best) in (int, float) and math.isfinite(best) else "--"
                postfix = (f"k={hb.get('current_k')} best_k={hb.get('best_k_so_far')} Val={accuracy} "
                    f"elapsed={duration(hb.get('session_elapsed_seconds'))} "
                    f"ETA~{duration(hb.get('eta_seconds_estimate'))}")
                # Re-render each snapshot; built-in elapsed/rate is intentionally omitted.
                # The visible elapsed/ETA comes from measured worker fields, not monitor age.
                bar = tqdm(total=total, initial=done, desc=label + " " + hb["phase"],
                    bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{postfix}]", leave=True)
                bar.set_postfix_str(postfix, refresh=True)
                bar.close()
                print("STATUS:", hb["status"], "| local candidates:", hb.get("local_completed_candidates"),
                      "| durable candidates:", hb.get("durable_completed_candidates"),
                      "| fit elapsed:", duration(hb.get("current_fit_elapsed_seconds")))
                print("CPU%:", hb.get("cpu_percent"), "| RSS GiB:", hb.get("ram_used_gib"),
                      "| available GiB:", hb.get("available_ram_gib"))
                age = (datetime.now(timezone.utc)-snapshot["time"]).total_seconds()
                print("CHECKPOINT:", hb.get("checkpoint_status"), hb.get("last_checkpoint"),
                      "| saved UTC:", hb.get("last_checkpoint_utc"), "| heartbeat age:", round(age), "s")
                if age > 90 and hb["status"] == "RUNNING":
                    print("WARNING: stale RUNNING heartbeat does not prove worker is alive or stopped")
                print("Test values read:", hb.get("test_values_read"), "| source:", snapshot["path"])
                for warning in snapshot["warnings"]:
                    print("READ WARNING:", warning)
            print("ETA is an estimate from completed units. No mid-fit percentage is inferred.")
            if job.get("running") is not True:
                print("Snapshot finished. JOB=False is not a verification PASS.")
                break
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("Monitor interrupted; no STOP_REQUESTED was sent by this cell.")


def summarize(requests):
    for request in requests:
        _, remote = paths(request)
        path = remote / "results/selection.json"
        marker = remote / "results/SELECTION_COMPLETE.json"
        if not path.is_file() or not marker.is_file():
            print(request["config"], request["run_id"], "NOT_COMPLETE_OR_NOT_PUBLISHED")
            continue
        value = read(path)
        identity = read(remote / "run_manifest.json")["identity"]
        if (value["identity"] != identity["sha256"] or read(marker)["identity"] != identity["sha256"]
                or identity["config"] != request["config"] or identity["run_id"] != request["run_id"]):
            raise Blocked("Saved report belongs to a different config/run")
        print(request["config"], request["run_id"], "| best k:", value["best_k"],
              "| coarse best k:", value["coarse_best_k"], "| fine:", value["fine"],
              "| Val Accuracy:", value["validation_metrics"]["accuracy"],
              "| SAVED_REPORT_ONLY, run verify for full SHA checks")
        print("RESULTS:", remote / "results")
