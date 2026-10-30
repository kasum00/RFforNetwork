"""B-only, immutable delta checkpoints on Drive after every stage/k."""
from __future__ import annotations
import os
import re
import shutil
import uuid
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, SCIENTIFIC, Blocked, atomic, check_files,
                       digest, drive_json_once, info, read, sha, utc)
from .policy import validate_policy


def released(root, lease):
    p = Path(root) / "logs" / (lease["id"] + ".RELEASED.json")
    if not p.is_file():
        return False
    try:
        r = read(p)
        return isinstance(r, dict) and r.get("identity") == lease["identity"] and r.get("status") in ("PAUSED", "FAILED", "SELECTION_COMPLETE")
    except (ValueError, OSError):
        return False


def checked(directory, identity, sequence, previous, stages):
    if not directory.is_dir() or directory.is_symlink():
        raise Blocked("Invalid checkpoint directory")
    marker, record = read(directory / "COMMITTED.json"), read(directory / "checkpoint.json")
    if (marker.get("status") != "COMMITTED" or marker.get("identity") != identity["sha256"]
            or marker.get("record") != info(directory / "checkpoint.json")
            or record.get("format") != FORMAT or record.get("algorithm") != ALGORITHM
            or record.get("identity") != identity["sha256"] or record.get("sequence") != sequence
            or record.get("config") != identity["config"] or record.get("run_id") != identity["run_id"]
            or record.get("dataset_fingerprint") != identity["dataset_fingerprint"]
            or record.get("source_sha256") != identity["source"]["sha256"]
            or record.get("git_commit") != identity["git_commit"] or record.get("parameters") != identity["contract"]
            or record.get("rfe_step") != 2
            or record.get("search_policy_sha256") != digest(identity["search_policy"])
            or record.get("parent") != previous or record.get("stage") in stages
            or directory.name != record.get("directory")):
        raise Blocked("Checkpoint algorithm/identity/sequence/parent mismatch")
    metadata = record.get("metadata", {})
    if "feature_names_order" in metadata and metadata.get("feature_order_sha256") != digest(metadata["feature_names_order"]):
        raise Blocked("Checkpoint feature order digest differs")
    files = record.get("files", {})
    if not isinstance(files, dict) or not files:
        raise Blocked("Checkpoint has no verified artifacts")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if present != set(files) | {"checkpoint.json", "COMMITTED.json"}:
        raise Blocked("Missing/extra checkpoint file")
    check_files(directory, files)
    return {"path": directory, "record": record, "sha256": sha(directory / "checkpoint.json")}


def ledger(root, identity):
    rows, previous, stages = [], None, set()
    for directory in sorted((Path(root) / "checkpoints").glob("CP_*")):
        row = checked(directory, identity, len(rows)+1, previous, stages)
        stages.add(row["record"]["stage"])
        previous = row["sha256"]
        rows.append(row)
    return rows


class Store:
    def __init__(self, local, remote, identity, *, resume, takeover=None):
        validate_policy(identity.get("search_policy"))
        if (identity.get("format") != FORMAT or identity.get("algorithm") != ALGORITHM
                or identity.get("rfe_step") != 2 or identity.get("contract", {}).get("rfe", {}).get("step") != 2
                or identity.get("sha256") != digest({k: v for k, v in identity.items() if k != "sha256"})):
            raise Blocked("Store accepts only an intact B step2 target1 identity")
        self.local, self.remote, self.identity = Path(local), Path(remote), identity
        self.state = self.local / "state"
        self.local.mkdir(parents=True, exist_ok=True)
        if self.remote.exists():
            if not resume or not (self.remote / "run_manifest.json").is_file():
                raise Blocked("Existing output protected; choose a fresh run ID or explicit resume")
            manifest = read(self.remote / "run_manifest.json")
            if manifest.get("format") != FORMAT or manifest.get("identity") != identity:
                raise Blocked("Run is not the same B dataset/config/source/parameters identity")
        else:
            if resume:
                raise Blocked("No existing B run to resume")
            self.remote.mkdir(parents=True, exist_ok=False)
            manifest = {"format": FORMAT, "algorithm": ALGORITHM, "identity": identity, "created": utc(), **SCIENTIFIC}
            drive_json_once(self.remote / "run_manifest.json", manifest)
        self.manifest = manifest
        self.rows = ledger(self.remote, identity)
        self.lease = None
        self.state.mkdir(parents=True, exist_ok=True)
        local_manifest = self.local / "run_manifest.json"
        if local_manifest.exists() and read(local_manifest) != manifest:
            raise Blocked("Local state belongs to another run; do not merge")
        if not local_manifest.exists():
            atomic(local_manifest, manifest, exclusive=True)
        self.restore()
        if self.complete:
            return  # Completed output is read-only, including telemetry.
        claims = sorted((self.remote / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        prior = read(claims[-1]) if claims else None
        if prior and not released(self.remote, prior) and takeover != "PREVIOUS RUNTIME STOPPED " + prior["id"]:
            raise Blocked("Unreleased session; confirm previous runtime stopped: " + prior["id"])
        self.lease = {"id": f"LEASE_{len(claims)+1:06d}_{uuid.uuid4().hex}",
                      "identity": identity["sha256"], "started": utc()}
        drive_json_once(self.remote / "logs" / f"LEASE_{len(claims)+1:06d}.json", self.lease)

    @property
    def complete(self):
        return (self.remote / "results/SELECTION_COMPLETE.json").exists()

    def guard(self):
        if self.complete:
            raise Blocked("Completed B selection is immutable")
        claims = sorted((self.remote / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if not self.lease or not claims or read(claims[-1]) != self.lease:
            raise Blocked("Worker lease superseded; do not continue")

    def restore(self):
        for row in self.rows:
            for name, record in row["record"]["files"].items():
                target = self.state / name
                if target.exists():
                    if info(target) != record:
                        raise Blocked("Local scientific artifact differs from committed checkpoint: " + name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (row["path"] / name).open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 8*1024**2)
                if info(target) != record:
                    raise Blocked("Restored checkpoint checksum mismatch")

    def saved(self, stage):
        return next((r for r in self.rows if r["record"]["stage"] == stage), None)

    def commit(self, stage, files, *, metadata=None):
        self.guard()
        if not re.fullmatch(r"[a-z0-9_]{1,90}", stage) or self.saved(stage):
            raise Blocked("Checkpoint stage already committed/invalid; no overwrite")
        sequence, token = len(self.rows)+1, uuid.uuid4().hex
        name = f"CP_{sequence:06d}_{stage}_{token}"
        working = self.remote / "checkpoints" / (".incomplete_" + name)
        working.mkdir(parents=True, exist_ok=False)
        records = {}
        for relative, path in files.items():
            p = Path(relative)
            if (p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative
                    or p.suffix not in (".json", ".csv", ".log", ".joblib", ".md")
                    or p.suffix == ".joblib" and p.name != "top100_scaler.joblib"):
                raise Blocked("Training data or fitted per-k model forbidden in checkpoint")
            path, target = Path(path), working / relative
            if path.is_symlink():
                raise Blocked("Symlink checkpoint source")
            records[relative] = info(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*1024**2)
                dst.flush()
                os.fsync(dst.fileno())
            if info(target) != records[relative]:
                raise OSError("Upload checksum/size failed; older checkpoints preserved")
        identity = self.identity
        record = {"format": FORMAT, "algorithm": ALGORITHM, "identity": identity["sha256"],
                  "config": identity["config"], "run_id": identity["run_id"],
                  "dataset_fingerprint": identity["dataset_fingerprint"], "git_commit": identity["git_commit"],
                  "source_sha256": identity["source"]["sha256"], "parameters": identity["contract"],
                  "stage": stage, "sequence": sequence, "directory": name,
                  "rfe_step": 2, "search_policy_sha256": digest(identity["search_policy"]),
                  "parent": self.rows[-1]["sha256"] if self.rows else None,
                  "files": records, "metadata": metadata or {}, "finished": utc(), **SCIENTIFIC}
        drive_json_once(working / "checkpoint.json", record)
        check_files(working, records)
        drive_json_once(working / "COMMITTED.json", {"status": "COMMITTED", "identity": identity["sha256"],
                                                    "record": info(working / "checkpoint.json")})
        self.guard()
        destination = working.with_name(name)
        if destination.exists():
            raise Blocked("Concurrent checkpoint publication")
        os.rename(working, destination)
        verified = checked(destination, identity, sequence, record["parent"],
                           {r["record"]["stage"] for r in self.rows})
        if verified["record"] != record:
            raise Blocked("Commit readback/sequence verification failed")
        self.rows.append(verified)
        return verified

    def mutable(self, name, value):
        if name not in ("progress.json", "heartbeat.json", "cache_manifest.json"):
            raise Blocked("Only owned current telemetry/cache manifest may be updated")
        self.guard()
        # Atomic local replacement logic is reused; Drive FUSE durability is best effort.
        atomic(self.remote / name, value)
        if read(self.remote / name) != value:
            raise OSError("Drive telemetry readback failed: " + name)

    def release(self, status):
        if not self.lease:
            return
        drive_json_once(self.remote / "logs" / (self.lease["id"] + ".RELEASED.json"),
                        {"identity": self.identity["sha256"], "status": status, "utc": utc()})
