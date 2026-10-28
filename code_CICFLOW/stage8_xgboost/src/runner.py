"""Four sequential isolated workers. No task runs until the user calls run_all."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import unittest
import uuid
from pathlib import Path
from .common import Blocked, CONFIGS, SCIENTIFIC, confirmation, ensure_owned, output_path, read, snapshot, utc, write
from . import dataset_loader, drive_artifact_manager as artifacts, xgboost_preflight as admission

MARKER = "__CICFLOW_STAGE8_RESULT__"

def redact(text):
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", str(text))
    return re.sub(r"(?i)((?:password|token|secret|api_key)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)

def validate_batch(request):
    confirmation(request)
    if request.get("mode") not in ("preflight", "full", "resume", "verify"):
        raise Blocked("MODE must be preflight/full/resume/verify")
    if request.get("experiment") != "baseline_xgb":
        raise Blocked("mi_rfe_xgb is not approved")
    for config in CONFIGS:
        output_path({**request, "config": config})

def collect_tests(request):
    suite = unittest.defaultTestLoader.discover(str(Path(request["source"]) / "stage8_xgboost/tests"),
                                               pattern="test_*.py", top_level_dir=request["source"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"status": "PASS" if result.testsRun and result.wasSuccessful() and not result.skipped else "FAIL",
            "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "source_sha256": snapshot(request["source"])["sha256"],
            "environment": admission.environment(request["backend"])}

def worker(action, request):
    directory = ensure_owned(request)
    if action == "tests":
        return collect_tests(request)
    if action == "metadata":
        return dataset_loader.inspect(request)
    if action == "resources":
        return admission.inspect(request, require_tests=False, persist_identity=False)
    if action == "preflight":
        return admission.inspect(request)
    if action == "verify":
        return artifacts.verify(directory)
    if action == "run":
        from .xgboost_training import execute
        return execute(request)
    raise Blocked("Unknown action")

def tree_rss(pid):
    import psutil
    total = 0
    try:
        parent = psutil.Process(pid)
        for proc in [parent, *parent.children(recursive=True)]:
            try:
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    return total

def launch(action, request):
    directory = ensure_owned(request)
    log_dir = directory / "logs"
    log_dir.mkdir(exist_ok=True)
    token = utc().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:6]
    command = [sys.executable, "-u", "-m", "stage8_xgboost.src.runner", "--worker", action]
    start, peak, process, native_exit = time.perf_counter(), 0, None, None
    peak_gpu_global = 0
    errors, results = [], []
    # Opening the log must succeed before a child is allowed to start.
    try:
        log_handle = (log_dir / "execution.log").open("a", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"LOG OPEN FAILED: {log_dir / 'execution.log'}; worker not started: {exc}") from exc
    with log_handle as handle:
        def log(text):
            try:
                handle.write(redact(text) + "\n")
                handle.flush()
            except OSError as exc:
                errors.append(str(exc))
                print("LOG WRITE FAILED: " + str(exc), file=sys.stderr, flush=True)
                raise
        log("="*72)
        log(json.dumps({"START": utc(), "TASK": action, "COMMAND_ARGV": command,
                        "CONFIG": request["config"], "RUN": directory.name, "REQUEST": request, **SCIENTIFIC}))
        try:
            env = dict(os.environ, PYTHONPATH=request["source"], PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=request["source"], env=env, text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            messages = queue.Queue()

            def pump(stream, name):
                try:
                    for line in iter(stream.readline, ""):
                        messages.put((name, line.rstrip("\r\n")))
                finally:
                    messages.put((name, None))
                    stream.close()

            threads = [threading.Thread(target=pump, args=(stream, label), daemon=True)
                       for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
            for thread in threads:
                thread.start()
            closed, last_sample, budget_failure = 0, 0., None
            while closed < 2:
                try:
                    label, line = messages.get(timeout=1)
                    if line is None:
                        closed += 1
                    else:
                        log(f"{utc()} [{label}] {line}")
                        if label == "stdout" and line.startswith(MARKER):
                            results.append(json.loads(line[len(MARKER):]))
                        else:
                            print(redact(line), file=sys.stderr if label == "stderr" else sys.stdout, flush=True)
                except queue.Empty:
                    pass
                if action == "run" and time.perf_counter()-last_sample >= 2 and process.poll() is None:
                    import psutil
                    last_sample = time.perf_counter()
                    rss, budget = tree_rss(process.pid), request["budget"]
                    peak = max(peak, rss)
                    gpu = admission.gpu_inventory() if request["backend"] == "cuda" else None
                    if gpu:
                        peak_gpu_global = max(peak_gpu_global, gpu["total_bytes"]-gpu["free_bytes"])
                    if (rss > budget["max_tree_rss_bytes"] or psutil.virtual_memory().available < budget["min_runtime_available_ram_bytes"]
                            or time.perf_counter()-start > budget["max_elapsed_seconds"]
                            or (gpu and gpu["free_bytes"] < budget["min_gpu_free_bytes"])
                            or __import__("shutil").disk_usage(directory).free < budget["min_free_disk_bytes"]):
                        budget_failure = "Resource/time/disk budget exceeded; no parameter fallback"
                        process.terminate()
            for thread in threads:
                thread.join()
            native_exit = process.wait()
            result = results[0] if len(results) == 1 else {"status": "FAIL", "error": "Missing/duplicate structured worker result"}
            status = "BLOCKED" if budget_failure else result.get("status", "FAIL")
            if native_exit != 0 and status == "PASS":
                status = "FAIL"
            if budget_failure:
                result["resource_error"] = budget_failure
            if action == "run" and native_exit == 0 and status == "PASS":
                result = artifacts.complete_after_worker_exit(directory, native_exit)
            envelope = {"action": action, "config": request["config"], "run_name": directory.name,
                        "program_status": status, "exit_code": native_exit, "result": result,
                        "runtime_seconds": time.perf_counter()-start, "peak_tree_rss_bytes_sampled": peak,
                        "peak_gpu_global_used_bytes_sampled": peak_gpu_global,
                        "end": utc(), "log": str(log_dir / "execution.log"), "logging_errors": errors, **SCIENTIFIC}
            log(json.dumps(envelope))
            write(directory / "reports" / f"execution_{action}_{token}.json", envelope)
            return envelope
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    native_exit = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    native_exit = process.wait()
            elif process is not None:
                native_exit = process.returncode
            record = {"status": "INTERRUPTED_OR_FAILED", "exit_code": native_exit, "error": str(exc),
                      "end": utc(), "completed": False, **SCIENTIFIC}
            try:
                log(json.dumps(record))
                write(directory / "reports" / f"FAILED_{token}.json", record)
            except OSError as logging_error:
                print("FAILURE LOG NOT SAVED: " + str(logging_error), file=sys.stderr)
            exc.native_exit_code = native_exit
            raise

def run_all(request, action=None):
    """Exactly four sequential tasks; independent output + checkpoint per config."""
    validate_batch(request)
    mode = request["mode"]
    action = action or ("preflight" if mode == "preflight" else "verify" if mode == "verify" else "run")
    if action == "run" and request.get("confirmation") != confirmation(request):
        raise Blocked("Confirm the exact ALL4 phrase once before starting the queue")
    rows = []
    for index, config in enumerate(CONFIGS, 1):
        current = {**request, "config": config}
        print(f"\n=== STAGE8 {index}/4: {config} | {action} ===", flush=True)
        try:
            directory = ensure_owned(current)
            if action in ("run", "preflight") and (directory / "RUN_COMPLETE.json").exists():
                # Fresh admission binds current dataset/source/environment before skipping.
                plan = launch("preflight", current)
                if plan["program_status"] != "PASS" or plan["exit_code"] != 0:
                    row = plan
                else:
                    row = {"config": config, "run_name": directory.name, "program_status": "PASS", "exit_code": 0,
                           "result": artifacts.verify(directory), "action": "VERIFIED_COMPLETE_SKIPPED"}
            else:
                row = launch(action, current)
        except KeyboardInterrupt:
            raise  # never keep starting configurations after an explicit stop
        except Exception as exc:
            row = {"config": config, "program_status": "BLOCKED" if isinstance(exc, Blocked) else "FAIL",
                   "exit_code": getattr(exc, "native_exit_code", None), "error": str(exc)}
            print(row, file=sys.stderr, flush=True)
        rows.append(row)
        if row["program_status"] != "PASS" and request.get("stop_on_error", False):
            break
    summary = {"status": "PASS" if len(rows) == 4 and all(r["program_status"] == "PASS" for r in rows) else "INCOMPLETE_OR_FAILED",
               "mode": mode, "action": action, "rows": rows, **SCIENTIFIC}
    # No batch log at MyDrive/repository root: a copy stays in each owned run.
    token = uuid.uuid4().hex
    for row in rows:
        try:
            directory = ensure_owned({**request, "config": row["config"]})
            write(directory / "reports" / f"BATCH_{token}.json", summary)
        except Exception as exc:
            print("BATCH SUMMARY NOT SAVED for " + row["config"] + ": " + str(exc), file=sys.stderr)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, choices=("tests", "metadata", "resources", "preflight", "run", "verify"))
    args = parser.parse_args()
    try:
        request = json.load(sys.stdin)
        result = worker(args.worker, request)
        exit_code = 0 if result.get("status") == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "BLOCKED" if isinstance(exc, (Blocked, FileExistsError)) else "FAIL", "error": str(exc), **SCIENTIFIC}
        exit_code = 2 if result["status"] == "BLOCKED" else 1
    print(MARKER + json.dumps(result, allow_nan=False), flush=True)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
