"""Live user-side launcher with append-only aggregate logging and native exits."""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from .contract import atomic, paths, redact, utc
from .cli import MARKER

TEST_MARKER = "__B2_TEST_RESULT__"


def _execute(command, source, log_root, aggregate, metadata, *, payload=None, stop_path=None, marker=MARKER):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    task_id, started, clock = uuid.uuid4().hex, utc(), time.monotonic()
    local_log = log_root / (task_id + ".log")
    aggregate_ok = True

    def append(text):
        nonlocal aggregate_ok
        if not aggregate_ok:
            return
        try:
            aggregate.parent.mkdir(parents=True, exist_ok=True)
            with aggregate.open("a", encoding="utf-8") as handle:
                handle.write(redact(text))
        except OSError as exc:
            aggregate_ok = False
            print("LOG WARNING: aggregate output not fully saved: " + str(exc), flush=True)

    header = {"task_id": task_id, "start": started, "command": command,
              "working_directory": str(source), **metadata}
    append("\n" + "="*72 + "\n" + json.dumps(header, ensure_ascii=False) + "\nSTDOUT/STDERR (merged):\n")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process, result, native, raised = None, None, None, None
    with local_log.open("x", encoding="utf-8") as handle:
        def consume(line):
            nonlocal result
            clean = redact(line)
            handle.write(clean)
            handle.flush()
            append("[" + task_id + "] " + clean)
            if line.startswith(marker):
                if result is not None:
                    raise RuntimeError("Duplicate structured worker result")
                result = json.loads(line[len(marker):])
            else:
                print(clean, end="", flush=True)
        try:
            process = subprocess.Popen(command, cwd=str(source), env=env,
                stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1)
            if payload is not None:
                process.stdin.write(json.dumps(payload))
                process.stdin.close()
            for line in process.stdout:
                consume(line)
            native = process.wait()
        except BaseException as exc:
            raised = redact(f"{type(exc).__name__}: {exc}")
            if process is not None and process.poll() is None:
                if stop_path is not None:
                    stop_path.write_text("Controller requested safe stop: " + utc(), encoding="utf-8")
                    print("Waiting for B worker safe boundary; do not start another worker.", flush=True)
                for line in process.stdout:
                    consume(line)
                native = process.wait()
            raise
        finally:
            if process is not None and native is None:
                native = process.poll()
            record = {**header, "end": utc(), "runtime_seconds": time.monotonic()-clock,
                      "native_exit_code": native,
                      "reported_status": result.get("status") if result else "UNKNOWN_INTERRUPTED_OR_ERROR",
                      "scientific_status": result.get("scientific_status") if result else None,
                      "result": result, "exception": raised, "local_log": str(local_log)}
            append("\n" + json.dumps(record, ensure_ascii=False) + "\n" + "="*72 + "\n")
            record["aggregate_log_fully_saved"] = aggregate_ok
            try:
                atomic(log_root / (task_id + ".json"), record, exclusive=True)
            except OSError as exc:
                aggregate_ok = False
                print("LOG WARNING: controller receipt not saved: " + str(exc), flush=True)
    if result is None:
        raise RuntimeError(f"Worker exited {native} without structured result; inspect {local_log}")
    result["native_exit_code"] = native
    result["aggregate_log_fully_saved"] = aggregate_ok
    result["controller_log"] = str(local_log)
    print(metadata.get("config", "B TESTS"), "EXIT:", native, "STATUS:", result.get("status"), flush=True)
    return result


def launch(request):
    local, remote = paths(request)
    aggregate = Path(request["dataset"]).parent / "code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log"
    return _execute([sys.executable, "-B", "-u", "-m", "phuong_anB_step2.cli"],
        Path(request["source"]), local / "controller_logs", aggregate,
        {"stage": "B STEP2 SELECTION", "task": request["mode"],
         "experiment": "phuong_anB_step2", "config": request["config"], "run_id": request["run_id"],
         "output_directory": str(remote), "request": request},
        payload=request, stop_path=local / "STOP_REQUESTED" if request["mode"] in ("run", "resume") else None)


def launch_tests(source, local_root, drive):
    root = Path(local_root) / "phuong_anB_step2/_unit_evidence"
    root.mkdir(parents=True, exist_ok=True)
    evidence = root / (uuid.uuid4().hex + ".json")
    result = _execute([sys.executable, "-B", "-u", "-m", "phuong_anB_step2.selftest",
        "--source", str(source), "--evidence", str(evidence)],
        Path(source), root, Path(drive) / "code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log",
        {"stage": "B SYNTHETIC TESTS", "task": "unit_tests", "evidence_file": str(evidence)},
        marker=TEST_MARKER)
    if not evidence.is_file():
        raise RuntimeError("Selftest did not persist evidence")
    saved = json.loads(evidence.read_text(encoding="utf-8"))
    # Transport metadata is not part of the original evidence identity.
    return result, saved
