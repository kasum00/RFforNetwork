"""Durable 60s heartbeat, immediate events, 300s console status; no algorithm changes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from colab.svm_resumable_training import DurableProgress, SessionDeadline
from colab.svm_resumable_store import CheckpointStore, atomic_json, file_sha, read_json, utc_now


class LocalDeadline(SessionDeadline):
    def __init__(self, *args, stop_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_path = stop_path

    def may_start(self):
        return not self.stop_path.exists() and super().may_start()


class LocalProgress(DurableProgress):
    def __init__(self, *args, console_interval=300, session_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.console_interval, self.session_dir = console_interval, session_dir
        self.last_console = 0.0

    def __call__(self, event, **details):
        if event == "session_paused_at_checkpoint" and (self.session_dir / "STOP_REQUESTED.json").exists():
            details["error"] = "USER_REQUESTED_STOP_AT_SAFE_BOUNDARY"
        if event == "test_evaluation_started":
            # Canonical engine already committed final AND claimed Test before this event.
            from types import SimpleNamespace
            from .windows_preflight import verify_content
            session = read_json(self.session_dir / "local_execution.json")
            root = self.root
            binding = read_json(root / "run_manifest.json")["binding"]
            store = CheckpointStore(root, binding, resume=True)
            claim = read_json(root / "results" / "test_evaluation_claim.json")
            if claim.get("final_checkpoint_sha256") != store.reference("final"):
                raise PermissionError("Test content hashing requires the verified final-model claim")
            paths = SimpleNamespace(data_root=Path(session["paths"]["data_root"]))
            evidence = Path(session["dataset_checksums"])
            if file_sha(evidence) != session["dataset_checksums_sha256"]:
                raise PermissionError("Dataset checksum manifest changed before final Test")
            verify_content(paths, read_json(evidence), roles=("test",))
        with self.lock:
            if self.logging_error:
                raise OSError(f"Durable local log unavailable: {self.logging_error}")
            if event != "heartbeat":
                self.stage, self.stage_started = event, time.monotonic()
            self.current_k = details.get("k", self.current_k)
            self.completed_k = details.get("completed_k", self.completed_k)
            self.last_checkpoint = details.get("checkpoint", self.last_checkpoint)
            self.last_error = details.get("error", self.last_error)
            import psutil
            rss = 0
            for process in [self.proc, *self.proc.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            elapsed = time.monotonic() - self.started
            record = {"utc": utc_now(), "event": event, "stage": self.stage,
                      "session_id": self.session_id, "worker_pid": os.getpid(),
                      "process_status": self.proc.status(), "current_k": self.current_k,
                      "completed_k": self.completed_k, "elapsed_seconds": elapsed,
                      "total_elapsed_seconds": self.prior_observed_seconds + elapsed,
                      "total_elapsed_is_observed_worker_time_not_idle_between_sessions": True,
                      "stage_elapsed_seconds": time.monotonic() - self.stage_started,
                      "tree_rss_bytes": rss, "peak_tree_rss_bytes": self.peak_rss,
                      "available_ram_bytes": psutil.virtual_memory().available,
                      "cpu_percent": self.proc.cpu_percent(),
                      "last_completed_checkpoint": self.last_checkpoint,
                      "last_error_or_stop_reason": self.last_error, "details": details}
            destination = self.root / ("heartbeat.jsonl" if event == "heartbeat" else "events.jsonl")
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_json(self.root / "progress.json", record)
            if event != "heartbeat" or time.monotonic() - self.last_console >= self.console_interval:
                self.last_console = time.monotonic()
                print(f"[{record['utc']}] {self.stage} k={self.current_k} completed={self.completed_k}/100 "
                      f"elapsed={elapsed:.0f}s stage_elapsed={record['stage_elapsed_seconds']:.0f}s "
                      f"RSS={rss/1024**3:.2f}GiB CPU={record['cpu_percent']:.1f}% "
                      f"last={self.last_checkpoint} error={self.last_error}", flush=True)
            if event == "k_checkpoint_committed":
                stage = details["checkpoint"]
                print(f"CHECKPOINT {stage} SHA256={file_sha(self.root / 'checkpoints' / stage / 'checkpoint.json')}", flush=True)
            if event == "k_completed" and "result" in details:
                row = details["result"]
                print(f"FIT DONE k={details.get('k')} seconds={row['fit']['seconds']:.3f} "
                      f"ValAccuracy={row.get('accuracy')} ValF1={row.get('f1_attack')}", flush=True)


