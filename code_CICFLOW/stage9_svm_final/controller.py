"""Live stdout/stderr + per-task/aggregate append logs; called only by the user."""
import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .common import once, utc
from mi_rfe_svm_colab.common import redact


def launch(request, *, tests=False):
    task = "synthetic-tests" if tests else request["mode"]
    task_id = uuid.uuid4().hex
    log_root = Path(request["output"]) / "stage9_svm_final/logs" / request["method"] / request["config"]
    aggregate = Path(request["dataset"]).parent / "code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log"
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        aggregate.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print("WARNING: log setup failed; child process NOT started: " + str(exc), file=sys.stderr, flush=True)
        raise
    evidence_path = log_root / (task_id + ".tests.json")
    command = [sys.executable, "-B", "-u", "-m", "stage9_svm_final.selftest" if tests else "stage9_svm_final.cli"]
    if tests:
        command += ["--source", request["source"], "--evidence", str(evidence_path)]
    started, clock = utc(), time.perf_counter()
    saved, errors = True, []
    handle = None
    try:
        handle = (log_root / (task_id + ".log")).open("x", encoding="utf-8")
    except OSError as exc:
        saved, errors = False, [str(exc)]

    def emit(stream, text):
        nonlocal saved
        safe = redact(text)
        print(safe, file=sys.stderr if stream == "stderr" else sys.stdout, flush=True)
        line = f"[{task_id}][{stream}] {safe}\n"
        try:
            if handle is None:
                raise OSError("Per-task log unavailable")
            handle.write(line)
            handle.flush()
            with aggregate.open("a", encoding="utf-8") as agg:
                agg.write(line)
                agg.flush()
        except OSError as exc:
            saved = False
            if str(exc) not in errors:
                errors.append(str(exc))
                print("WARNING: execution log write FAILED: " + str(exc), file=sys.stderr, flush=True)

    emit("header", json.dumps(dict(start=started, stage=9, task=task, method=request["method"],
        config=request["config"], run_id=request["run_id"], selection_run_id=request["selection_run_id"],
        command=command, stdin_request=None if tests else request, output_directory=str(log_root)), default=str))
    process = None
    native, values = None, []
    protocol_errors = []
    interruption = False
    try:
        process = subprocess.Popen(command, cwd=request["source"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        if not tests:
            process.stdin.write(json.dumps(request))
        process.stdin.close()
        events = queue.Queue()

        def pump(stream, name):
            try:
                for line in stream:
                    events.put((name, line.rstrip("\n")))
            finally:
                events.put((name, None))

        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            threading.Thread(target=pump, args=(stream, name), daemon=True).start()
        done = 0
        while done < 2:
            try:
                name, line = events.get(timeout=0.5)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                interruption = True
                emit("warning", "Monitor interrupted; worker is NOT killed. Waiting for actual child exit; do not start another worker.")
                continue
            if line is None:
                done += 1
                continue
            emit(name, line)
            if line.startswith("__STAGE9_RESULT__"):
                try:
                    values.append(json.loads(line[len("__STAGE9_RESULT__"):]))
                except ValueError as exc:
                    protocol_errors.append(str(exc))
        native = process.wait()
    except BaseException as exc:
        emit("stderr", f"Launcher error: {type(exc).__name__}: {exc}")
        # Never fabricate native exit or claim completion for a live child.
        if process is not None and process.poll() is not None:
            native = process.returncode
    finally:
        result = values[0] if len(values) == 1 and not protocol_errors else dict(status="UNCONFIRMED", reason="No unique valid worker result marker")
        if tests and evidence_path.is_file():
            from .common import read
            try:
                result = read(evidence_path)
            except (OSError, ValueError) as exc:
                result = dict(status="UNCONFIRMED", reason="Synthetic evidence unreadable: " + str(exc))
        live_child = process is not None and process.poll() is None
        if live_child:
            result["status"] = "BLOCKED_RUNNING_CHILD"
        result.update(native_exit_code=native, logs_saved=saved, log_errors=errors,
                      worker_still_running=live_child, protocol_errors=protocol_errors,
                      task_log=str(log_root / (task_id + ".log")), aggregate_log=str(aggregate))
        emit("footer", json.dumps(dict(end=utc(), runtime_seconds=time.perf_counter()-clock,
             native_exit_code=native, program_status=result.get("status"), scientific_conclusion="NOT_INFERRED_FROM_EXIT_CODE",
             worker_still_running=process is not None and process.poll() is None, interrupted_monitor=interruption)))
        result["logs_saved"], result["log_errors"] = saved, errors
        try:
            once(log_root / (task_id + ".receipt.json"), result)
        except (OSError, ValueError) as exc:
            result["logs_saved"] = False
            print("WARNING: execution receipt was NOT saved: " + str(exc), file=sys.stderr)
        if handle:
            handle.close()
    return result
