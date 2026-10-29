"""Measured heartbeat and safe-boundary session budget; no synthetic progress."""
import json
import shutil
import threading
import time
from pathlib import Path
from .contract import ALGORITHM, Blocked, atomic, redact, utc


class Pause(Exception):
    pass


class Capture:
    def __init__(self, path, stdout, stderr):
        self.path, self.stdout, self.stderr = Path(path), stdout, stderr
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")
        self.lock, self.offset = threading.RLock(), 0

    def stream(self, is_error=False):
        owner = self
        class Stream:
            def write(self, text):
                clean = redact(text)
                with owner.lock:
                    (owner.stderr if is_error else owner.stdout).write(clean)
                    owner.file.write(clean)
                return len(text)
            def flush(self):
                with owner.lock:
                    owner.file.flush()
                    (owner.stderr if is_error else owner.stdout).flush()
        return Stream()

    def snapshot(self, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.file.flush()
            with self.path.open("rb") as src, target.open("xb") as dst:
                src.seek(self.offset)
                shutil.copyfileobj(src, dst, 64*1024)
                self.offset = src.tell()
        return target

    def close(self):
        with self.lock:
            self.file.close()


class Telemetry:
    def __init__(self, store, request, *, clock=time.monotonic):
        import psutil
        self.store, self.request, self.clock = store, request, clock
        self.psutil, self.proc = psutil, psutil.Process()
        self.started, self.fit_started, self.fit_active = clock(), None, False
        self.status, self.stage, self.current_k, self.completed = "RUNNING", "INITIALIZING", None, 0
        self.best_k, self.best_accuracy, self.cpu = None, None, None
        self.checkpoint_status, self.last_checkpoint = "NOT_SAVED", None
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.error, self.peak_rss = None, 0
        self.interval = request["runtime"]["heartbeat_seconds"]
        self.last_snapshot = None

    def stop_requested(self):
        runtime = self.request["runtime"]
        seconds = runtime["session_budget_seconds"]
        deadline_due = seconds > 0 and self.clock()-self.started >= seconds-runtime["stop_margin_seconds"]
        return (self.store.local / "STOP_REQUESTED").exists() or deadline_due

    def check(self):
        if self.error:
            raise Blocked("Heartbeat failed: " + self.error)
        if self.stop_requested():
            raise Pause("User stop or session budget reached at safe boundary")
        budget = self.request["budget"]
        rss, available = self.proc.memory_info().rss, self.psutil.virtual_memory().available
        if rss > budget["max_process_rss_bytes"] or available < budget["min_available_ram_bytes"]:
            raise Blocked("RAM/RSS budget breached; no automatic cache/data/parameter reduction")
        if shutil.disk_usage(self.store.local).free < 1024**3:
            raise Blocked("Local disk below 1 GiB safe-boundary headroom")

    def event(self, event, **details):
        with self.lock:
            if event.startswith("mi_"):
                self.stage = "MI TOP100"
            if event.startswith("rfe_"):
                self.stage = "RF-RFE STEP5 TARGET1"
            if event == "k_started":
                self.stage, self.current_k = "SVM K-SEARCH", details["k"]
            if event.endswith("fit_started") or event == "mi_started" or event == "rfe_final_estimator_fit_started":
                self.fit_started, self.fit_active = self.clock(), True
            if event.endswith("fit_completed") or event in ("mi_completed", "rfe_step_completed", "k_completed"):
                self.fit_active = False
            if "completed_k" in details:
                self.completed = details["completed_k"]
            if event == "k_completed":
                self.completed = details["k"]
                self.best_k, self.best_accuracy = details["best_k"], details["best_validation_accuracy"]
            row = {"utc": utc(), "event": event, "identity": self.store.identity["sha256"], **details}
            with (self.store.local / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(redact(json.dumps(row, allow_nan=False)) + "\n")
            print(redact(json.dumps(row, allow_nan=False)), flush=True)

    def snapshot(self):
        with self.lock:
            vm, rss = self.psutil.virtual_memory(), self.proc.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            return {"utc": utc(), "experiment": "phuong_anA", "algorithm": ALGORITHM,
                "config": self.request["config"], "run_id": self.request["run_id"],
                "identity": self.store.identity["sha256"], "status": self.status, "stage": self.stage,
                "progress": f"{self.completed}/100", "current_k": self.current_k,
                "last_completed_k": self.completed, "current_fit_elapsed_seconds":
                    self.clock()-self.fit_started if self.fit_active and self.fit_started is not None else None,
                "total_elapsed_seconds_this_session": self.clock()-self.started,
                "cpu_percent": self.cpu, "cpu_scope": "worker process; may exceed 100% with RF threads",
                "ram_used_gib": rss/1024**3, "available_ram_gib": vm.available/1024**3,
                "peak_process_rss_bytes_sampled_this_session": self.peak_rss,
                "best_k_so_far": self.best_k, "best_validation_accuracy": self.best_accuracy,
                "checkpoint_status": self.checkpoint_status, "last_checkpoint": self.last_checkpoint,
                "durable_last_completed_k": max([int(r["record"]["stage"][2:]) for r in self.store.rows
                    if r["record"]["stage"].startswith("k_")], default=0),
                "test_values_read": False}

    def emit(self):
        row = self.snapshot()
        atomic(self.store.local / "heartbeat.json", row)
        atomic(self.store.local / "progress.json", row)
        self.store.mutable("heartbeat.json", row)
        self.store.mutable("progress.json", row)
        self.last_snapshot = row
        print("A1 HEARTBEAT " + json.dumps(row, allow_nan=False), flush=True)

    def loop(self):
        while not self.stop.wait(self.interval):
            try:
                # A single nonblocking sample per timed poll, primed in __enter__.
                self.cpu = self.proc.cpu_percent(interval=None)
                self.emit()
            except Exception as exc:
                self.error = redact(exc)
                print("HEARTBEAT NOT SAVED: " + self.error, flush=True)
                return

    def __enter__(self):
        self.proc.cpu_percent(interval=None)  # Discard initial meaningless sample.
        self.emit()  # cpu_percent remains None until a real interval has elapsed.
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        try:
            self.emit()
        except Exception as exc:
            self.error = redact(exc)
            print("FINAL HEARTBEAT NOT SAVED: " + self.error, flush=True)
