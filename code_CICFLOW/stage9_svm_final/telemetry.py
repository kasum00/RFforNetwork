"""Measured heartbeat; SVC.fit has no invented internal percentage."""
import threading
import time
from .common import mutable, utc


class Heartbeat:
    def __init__(self, root, identity, interval=15):
        import psutil
        self.psutil, self.proc = psutil, psutil.Process()
        self.root, self.identity, self.interval = root, identity, interval
        self.started = time.perf_counter()
        self.stage, self.status, self.error, self.last = "INITIALIZING", "RUNNING", None, None
        self.peak, self.telemetry_error = 0, None
        self.stop = threading.Event()

    def snapshot(self):
        rss = self.proc.memory_info().rss
        self.peak = max(self.peak, rss)
        return dict(utc=utc(), identity=self.identity["sha256"], method=self.identity["method"],
            config=self.identity["config"], stage=self.stage, status=self.status,
            elapsed_seconds=time.perf_counter()-self.started, cpu_percent=self.proc.cpu_percent(),
            ram_bytes=rss, peak_rss_bytes_sampled=self.peak,
            available_ram_bytes=self.psutil.virtual_memory().available,
            last_checkpoint=self.last, error=self.error, svc_internal_percent=None)

    def emit(self):
        try:
            value = self.snapshot()
            mutable(self.root / "heartbeat.json", value)
            print("HEARTBEAT: " + str(value), flush=True)
        except Exception as exc:
            self.telemetry_error = str(exc)
            print("WARNING: heartbeat write failed: " + str(exc), flush=True)

    def event(self, stage, **metadata):
        self.stage = stage
        print("EVENT: " + stage + " " + str(metadata), flush=True)
        self.emit()

    def loop(self):
        while not self.stop.wait(self.interval):
            self.emit()

    def __enter__(self):
        self.proc.cpu_percent()
        self.emit()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, tb):
        self.stop.set()
        self.thread.join()
        self.status = "FAILED" if kind else "FINISHED_WORKER"
        self.error = str(value) if value else None
        self.emit()
