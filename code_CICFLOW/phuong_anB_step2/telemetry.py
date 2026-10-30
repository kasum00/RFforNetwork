"""Measured heartbeat; local and durable completed work stay separate."""
import json
import shutil
import threading
import time
from .contract import ALGORITHM, Blocked, atomic, redact, utc
from .history import validate_rows
from .io_utils import Capture
from .policy import winner


class Pause(Exception):
    pass


class Telemetry:
    def __init__(self, store, request, *, clock=time.monotonic):
        import psutil
        self.store, self.request, self.clock = store, request, clock
        self.psutil, self.proc = psutil, psutil.Process()
        self.started, self.fit_started, self.fit_active = clock(), None, False
        self.status, self.phase, self.current_k = "RUNNING", "INITIALIZING", None
        self.rows, self.phase_total, self.phase_times = [], 1, []
        self.best_k, self.best_accuracy, self.cpu = None, None, None
        self.checkpoint_status, self.last_checkpoint, self.last_checkpoint_utc = "NOT_SAVED", None, None
        self.rf_started = self.rf_completed = self.svm_started = self.svm_completed = 0
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.error, self.peak_rss = None, 0
        self.interval = request["runtime"]["heartbeat_seconds"]

    def stop_requested(self):
        runtime = self.request["runtime"]
        budget = runtime["session_budget_seconds"]
        return ((self.store.local / "STOP_REQUESTED").exists()
                or budget > 0 and self.clock()-self.started >= budget-runtime["stop_margin_seconds"])

    def check(self):
        if self.error:
            raise Blocked("Heartbeat failed: " + self.error)
        if self.stop_requested():
            raise Pause("User stop/session budget at safe boundary")
        rss, available = self.proc.memory_info().rss, self.psutil.virtual_memory().available
        if rss > self.request["budget"]["max_process_rss_bytes"] or available < self.request["budget"]["min_available_ram_bytes"]:
            raise Blocked("RAM/RSS budget breached; no data or parameter reduction")
        if shutil.disk_usage(self.store.local).free < 1024**3:
            raise Blocked("Local disk below safe-boundary headroom")

    def set_phase(self, phase, total, durations=()):
        if type(total) is not int or total < 0:
            raise Blocked("Invalid real progress total")
        with self.lock:
            self.phase, self.phase_total, self.phase_times = phase, total, list(durations)
            self.current_k = None

    def restore_history(self, rows, ranked):
        validate_rows(rows, ranked, self.store.identity["search_policy"],
            validation_count=self.store.identity["dataset"]["splits"]["val"]["rows"])
        best = winner(rows) if rows else None
        with self.lock:
            self.rows = list(rows)
            self.best_k = best["k"] if best else None
            self.best_accuracy = best["accuracy"] if best else None

    def event(self, event, **details):
        with self.lock:
            if event.startswith("mi_"):
                self.phase, self.phase_total = "MI", 1
            if event.startswith("rfe_"):
                self.phase, self.phase_total = "RFE_COARSE", 51
            if event == "candidate_started":
                self.phase, self.current_k = details["phase"], details["k"]
            if event in ("rfe_fit_started", "rfe_final_estimator_fit_started"):
                self.rf_started += 1
            if event == "svm_fit_started":
                self.svm_started += 1
            if event.endswith("fit_started") or event == "mi_started":
                self.fit_started, self.fit_active = self.clock(), True
            if event == "rfe_fit_completed":
                self.rf_completed += 1
                self.phase_times.append(details["fit_seconds"])
            if event == "svm_fit_completed":
                self.svm_completed += 1
            if event.endswith("fit_completed") or event in ("mi_completed", "candidate_completed", "attempt_failed", "paused_at_safe_boundary"):
                self.fit_active = False
            row = {"utc": utc(), "event": event, "identity": self.store.identity["sha256"], **details}
            with (self.store.local / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(redact(json.dumps(row, allow_nan=False)) + "\n")
            print(redact(json.dumps(row, allow_nan=False)), flush=True)

    def snapshot(self):
        with self.lock:
            vm, rss = self.psutil.virtual_memory(), self.proc.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            ledger = list(self.store.rows)
            candidates = [r for r in ledger if r["record"]["stage"].startswith("candidate_")]
            counts = {phase: sum(r["record"]["metadata"]["phase"] == phase for r in candidates) for phase in ("COARSE", "FINE")}
            if self.phase == "RFE_COARSE":
                durable = sum(r["record"]["stage"].startswith("b2rfe_") for r in ledger)
                durable += int(any(r["record"]["stage"] == "rfe" for r in ledger))
            elif self.phase in counts:
                durable = counts[self.phase]
            elif self.phase == "MI":
                durable = int(any(r["record"]["stage"] == "mi" for r in ledger))
            else:
                durable = int(self.status == "SELECTION_COMPLETE")
            measured = self.phase_times
            if self.phase in counts:
                measured = [row["unit_seconds"] for row in self.rows if row["phase"] == self.phase]
            remaining = max(0, self.phase_total-durable)
            eta = 0.0 if remaining == 0 else sum(measured)/len(measured)*remaining if measured else None
            return {"utc": utc(), "experiment": "phuong_anB_step2", "algorithm": ALGORITHM,
                "config": self.request["config"], "run_id": self.request["run_id"],
                "identity": self.store.identity["sha256"], "status": self.status, "phase": self.phase,
                "rfe_step": 2, "phase_completed_durable": durable, "phase_total": self.phase_total,
                "current_k": self.current_k, "local_completed_candidates": len(self.rows),
                "durable_completed_candidates": len(candidates), "durable_by_phase": counts,
                "current_fit_elapsed_seconds": self.clock()-self.fit_started if self.fit_active and self.fit_started is not None else None,
                "session_elapsed_seconds": self.clock()-self.started,
                "eta_seconds_estimate": eta, "eta_scope": "mean measured completed unit time; current fit progress unknown",
                "cpu_percent": self.cpu, "ram_used_gib": rss/1024**3, "available_ram_gib": vm.available/1024**3,
                "best_k_so_far": self.best_k, "best_validation_accuracy": self.best_accuracy,
                "checkpoint_status": self.checkpoint_status, "last_checkpoint": self.last_checkpoint,
                "last_checkpoint_utc": self.last_checkpoint_utc,
                "peak_process_rss_bytes_sampled_this_session": self.peak_rss,
                "rf_fit_attempts_started_this_session": self.rf_started,
                "rf_fits_completed_this_session": self.rf_completed,
                "svm_fit_attempts_started_this_session": self.svm_started,
                "svm_fits_completed_this_session": self.svm_completed,
                "test_values_read": False}

    def emit(self):
        row = self.snapshot()
        for name in ("heartbeat.json", "progress.json"):
            atomic(self.store.local / name, row)
            self.store.mutable(name, row)
        print("B2 HEARTBEAT " + json.dumps(row, allow_nan=False), flush=True)

    def loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.cpu = self.proc.cpu_percent(interval=None)
                self.emit()
            except Exception as exc:
                self.error = redact(exc)
                print("HEARTBEAT NOT SAVED: " + self.error, flush=True)
                return

    def __enter__(self):
        self.proc.cpu_percent(interval=None)
        self.emit()
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
