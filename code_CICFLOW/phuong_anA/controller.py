"""Notebook-side sequential subprocess launcher with live local logging."""
import json
import os
import subprocess
import sys
import uuid
from .contract import atomic, paths, redact, utc
from .cli import MARKER


def launch(request):
    local, remote = paths(request)
    log_root = local / "controller_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp, started = uuid.uuid4().hex, utc()
    command = [sys.executable, "-B", "-u", "-m", "phuong_anA.cli"]
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.Popen(command, cwd=request["source"], env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    result = None
    try:
        child.stdin.write(json.dumps(request)); child.stdin.close()
        with (log_root / (stamp + ".log")).open("x", encoding="utf-8") as log:
            for line in child.stdout:
                line = redact(line); log.write(line); log.flush()
                if line.startswith(MARKER):
                    if result is not None:
                        raise RuntimeError("Duplicate structured worker result")
                    result = json.loads(line[len(MARKER):])
                else:
                    print(line, end="", flush=True)
    except BaseException:
        if child.poll() is None:
            (local / "STOP_REQUESTED").write_text("Controller requested stop after output/interrupt error", encoding="utf-8")
            print("Waiting for worker safe boundary; do not start another worker.", flush=True)
            for line in child.stdout:
                print(redact(line), end="", flush=True)
            child.wait()
        raise
    native_exit = child.wait()
    atomic(log_root / (stamp + ".json"), {"command": command, "start": started, "end": utc(),
        "config": request["config"], "run_id": request["run_id"], "mode": request["mode"],
        "native_exit_code": native_exit, "reported_status": (result or {}).get("status")}, exclusive=True)
    if result is None:
        raise RuntimeError(f"Worker exited {native_exit} without structured result; inspect {log_root}")
    result["native_exit_code"] = native_exit
    print(request["config"], "EXIT:", native_exit, "STATUS:", result.get("status"), flush=True)
    return result
