"""Versioned, checksummed bundles. No cloud I/O on an ordinary k boundary."""
from __future__ import annotations

import os
import shutil
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from .common import (Blocked, FORMAT, DRIVE_CHECKPOINT_INTERVAL_SECONDS, atomic,
                     check_files, digest, drive_json_once, info, read, sha, utc)


def state_files(state):
    paths = []
    for p in sorted(Path(state).rglob("*")):
        if p.is_symlink():
            raise Blocked("Symlink forbidden in checkpoint")
        if not p.is_file() or ".partial" in p.name:
            continue
        if p.suffix not in (".json", ".jsonl", ".csv", ".log", ".joblib"):
            raise Blocked("Cache/dataset or unsupported file in checkpoint state: " + str(p))
        if p.suffix == ".joblib" and p.name not in ("top100_scaler.joblib", "model.joblib", "final_scaler.joblib"):
            raise Blocked("Only completed preprocessing/final models may be bundled")
        paths.append(p)
    return paths


def committed(root, identity=None):
    root = Path(root)
    records = []
    for path in sorted((root / "checkpoints").glob("*.COMMITTED.json")):
        value = read(path)  # A malformed commit marker fails closed.
        if (value.get("format") != FORMAT or value.get("status") != "COMMITTED"
                or value.get("id") + ".COMMITTED.json" != path.name
                or (identity is not None and value.get("identity") != identity)):
            raise Blocked("Invalid committed checkpoint marker: " + str(path))
        records.append(value)
    sequences = [r["sequence"] for r in records]
    if len(set(sequences)) != len(sequences):
        raise Blocked("Concurrent or forked checkpoint sequence")
    records = sorted(records, key=lambda r: r["sequence"])
    previous = None
    for sequence, row in enumerate(records, 1):
        if row["sequence"] != sequence or row.get("parent") != previous:
            raise Blocked("Checkpoint parent/sequence gap")
        previous = row["id"]
    return records


def latest(root, identity):
    rows = committed(root, identity)
    if not rows:
        return None
    row = rows[-1]
    path = Path(root) / "checkpoints" / (row["id"] + ".zip")
    if info(path) != row["bundle"]:
        raise Blocked("Latest COMMITTED bundle is corrupt; never silently roll back")
    return row


def lease_released(root, lease):
    path = Path(root) / "leases" / (lease["id"] + ".RELEASED.json")
    if not path.exists():
        return False
    try:
        record = read(path)
        if not isinstance(record, dict):
            return False
        return (record.get("identity") == lease["identity"] and isinstance(record.get("utc"), str)
                and record.get("status") in {"COMPLETE", "COMPLETE_RECOVERED", "SELECTION_FROZEN",
                    "SELECTION_ALREADY_FROZEN", "FINAL_REFIT_SAVED", "PAUSED", "INTERRUPTED", "FAILED"})
    except (ValueError, OSError):
        return False  # A partial release is not proof that a worker stopped.


def unpack(bundle, destination, identity, *, max_bytes):
    """Unzip only after checksums, traversal, size and membership checks."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "BUNDLE_MANIFEST.json" not in names:
            raise Blocked("Duplicate/missing bundle manifest")
        import json
        manifest = json.loads(z.read("BUNDLE_MANIFEST.json"))
        files = manifest.get("files", {})
        if (manifest.get("format") != FORMAT or manifest.get("identity") != identity
                or not files or set(names) != set(files) | {"BUNDLE_MANIFEST.json"}):
            raise Blocked("Bundle manifest identity/membership mismatch")
        total = 0
        for name, record in files.items():
            rel = Path(name)
            member = z.getinfo(name)
            if (rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name
                    or name.startswith("/") or member.file_size != record["bytes"]
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise Blocked("Unsafe bundle member")
            total += member.file_size
        if total > max_bytes:
            raise Blocked("Insufficient local disk budget to restore bundle")
        destination.mkdir(parents=True)
        for name in files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*1024**2)
        check_files(destination, files)
    if read(destination / "run_identity.json")["sha256"] != identity:
        raise Blocked("Restored run identity mismatch")
    return manifest


class DriveCheckpoint:
    def __init__(self, local, remote, identity, *, clock=time.monotonic, takeover=None):
        self.local, self.remote, self.identity = Path(local), Path(remote), identity
        self.clock, self.last_success = clock, clock()
        self.latest_record, self.lease = None, None
        self.remote.mkdir(parents=True, exist_ok=True)
        owner = self.remote / "OWNER.json"
        manifest = self.remote / "run_identity.json"
        if not owner.exists():
            if any(self.remote.iterdir()):
                raise Blocked("Existing/legacy run directory protected; choose a new run tag")
            drive_json_once(owner, {"format": FORMAT, "identity": identity["sha256"]})
            drive_json_once(manifest, identity)
        if read(owner) != {"format": FORMAT, "identity": identity["sha256"]} or read(manifest) != identity:
            raise Blocked("Run identity/source/data/parameters/environment differ")
        if (self.remote / "RUN_COMPLETE.json").exists():
            raise Blocked("Completed run is immutable; use verify")
        # One-time session claims, never a per-k Drive heartbeat. A new runtime
        # must explicitly acknowledge the exact unreleased lease it supersedes.
        claims = sorted((self.remote / "leases").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        prior = read(claims[-1]) if claims else None
        if prior and not lease_released(self.remote, prior):
            if takeover != "PREVIOUS RUNTIME STOPPED " + prior["id"]:
                raise Blocked("Unreleased session: confirm previous runtime stopped: " + prior["id"])
        self.lease = {"id": f"LEASE_{len(claims)+1:06d}_{uuid.uuid4().hex}",
                      "identity": identity["sha256"], "started": utc(),
                      "supersedes": prior["id"] if prior else None}
        # Same sequence filename prevents two simultaneous claim winners.
        drive_json_once(self.remote / "leases" / f"LEASE_{len(claims)+1:06d}.json", self.lease)
        self.latest_record = latest(self.remote, identity["sha256"])
        if self.latest_record:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(self.latest_record["finished"])).total_seconds()
            if age < -300:
                raise Blocked("Checkpoint timestamp is in the future")
            self.last_success -= max(0, age)

    def assert_lease(self):
        claims = sorted((self.remote / "leases").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if not claims or read(claims[-1]) != self.lease:
            raise Blocked("Drive session lease was superseded; stop this worker")

    def release(self, status):
        self.assert_lease()
        drive_json_once(self.remote / "leases" / (self.lease["id"] + ".RELEASED.json"),
                        {"status": status, "identity": self.identity["sha256"], "utc": utc()})

    def due(self):
        return self.clock() - self.last_success >= DRIVE_CHECKPOINT_INTERVAL_SECONDS

    def sync(self, state, *, reason, safe_boundary, force=False):
        # This return occurs before any Drive stat/list/open.
        if not force and not self.due():
            return None
        if not safe_boundary:
            raise Blocked("Drive upload forbidden during SVC.fit/predict")
        if (self.remote / "RUN_COMPLETE.json").exists():
            raise Blocked("Completed run cannot receive another checkpoint")
        self.assert_lease()
        state = Path(state)
        prior = self.latest_record
        sequence = 1 if prior is None else prior["sequence"] + 1
        checkpoint_id = f"CP_{sequence:06d}_{uuid.uuid4().hex}"
        staging = self.local / "bundles"
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / (checkpoint_id + ".zip")
        paths = state_files(state)
        files = {p.relative_to(state).as_posix(): info(p) for p in paths}
        manifest = {"format": FORMAT, "identity": self.identity["sha256"], "files": files}
        import json
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in paths:
                z.write(p, p.relative_to(state).as_posix())
            z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, allow_nan=False))
        target = self.remote / "checkpoints" / archive.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8*1024**2)
            dst.flush()
            os.fsync(dst.fileno())
        local_info = info(archive)
        if info(target) != local_info:
            raise OSError("Drive upload verification failed; previous committed checkpoint retained")
        self.assert_lease()
        elapsed = self.clock() - self.last_success
        record = {"format": FORMAT, "status": "COMMITTED", "id": checkpoint_id,
                  "sequence": sequence, "parent": prior["id"] if prior else None,
                  "identity": self.identity["sha256"], "bundle": local_info,
                  "state_digest": digest(files), "reason": reason,
                  "completed_k": len(list((state / "k").glob("k_*.json"))),
                  "mi_state_sha256": sha(state / "mi_state.json") if (state / "mi_state.json").exists() else None,
                  "rfe_state_sha256": sha(state / "rfe_state.json") if (state / "rfe_state.json").exists() else None,
                  "top100_scaler_sha256": sha(state / "top100_scaler_state.json") if (state / "top100_scaler_state.json").exists() else None,
                  "final_state_sha256": sha(state / "final_state.json") if (state / "final_state.json").exists() else None,
                  "selection_sha256": sha(state / "selection_frozen.json") if (state / "selection_frozen.json").exists() else None,
                  "test_result_sha256": sha(state / "test_result.json") if (state / "test_result.json").exists() else None,
                  "result_sha256": sha(state / "result.json") if (state / "result.json").exists() else None,
                  "seconds_since_previous_sync": elapsed,
                  "interval_overrun_seconds": max(0, elapsed-DRIVE_CHECKPOINT_INTERVAL_SECONDS),
                  "safe_boundary": True, "finished": utc()}
        drive_json_once(target.with_name(checkpoint_id + ".COMMITTED.json"), record)
        if latest(self.remote, self.identity["sha256"]) != record:
            raise OSError("Committed checkpoint verification failed")
        self.last_success, self.latest_record = self.clock(), record
        atomic(self.local / "sync_receipt.json", record)
        print(f"DRIVE CHECKPOINT COMMITTED: {checkpoint_id}; completed_k={record['completed_k']}; reason={reason}", flush=True)
        return record


def restore(local, remote, identity):
    """Restore only the newest COMMITTED bundle; never overwrite local progress."""
    local, remote = Path(local), Path(remote)
    state = local / "state"
    latest_record = latest(remote, identity["sha256"])
    if state.exists():
        if read(state / "run_identity.json") != identity:
            raise Blocked("Existing local state has another identity")
        if latest_record:
            receipt_path = local / "sync_receipt.json"
            if not receipt_path.exists() or read(receipt_path)["id"] != latest_record["id"]:
                raise Blocked("Local state ancestry not proven against latest Drive bundle")
        return state
    if latest_record:
        stage = local / ("restore-" + uuid.uuid4().hex)
        manifest = unpack(remote / "checkpoints" / (latest_record["id"] + ".zip"), stage,
                          identity["sha256"], max_bytes=shutil.disk_usage(local).free - 1024**3)
        if digest(manifest["files"]) != latest_record["state_digest"]:
            raise Blocked("Bundle state digest differs from COMMITTED marker")
        state_files(stage)
        os.rename(stage, state)
        atomic(local / "sync_receipt.json", latest_record)
    else:
        state.mkdir(parents=True)
        atomic(state / "run_identity.json", identity, exclusive=True)
    return state
