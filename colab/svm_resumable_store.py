"""Immutable stage checkpoints for MI-RFE SVM. No estimator or dataset imports."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "stage73b.mi_rfe_svm.checkpoint.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_file(path: Path) -> None:
    """Sync an existing artifact without truncating it, including on Windows.

    Windows requires a writable descriptor for fsync. Writers must have closed
    their own handles before this call. Sync errors deliberately abort commit.
    """
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value, *, exclusive=False) -> None:
    """Exclusive records fail closed even if a process dies during their write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_stage(stage: str) -> Path:
    path = Path(stage)
    if path.is_absolute() or not path.parts or any(
            part in (".", "..") or ":" in part or "\\" in part for part in path.parts):
        raise ValueError("Unsafe checkpoint stage path")
    return path


class CheckpointStore:
    """Only a committed directory with a matching manifest is a checkpoint.

    Incomplete sibling directories are preserved and never loaded. FUSE rename
    durability is best effort; every reuse verifies every artifact checksum.
    """
    def __init__(self, root: Path, binding: dict, *, resume: bool):
        self.root, self.binding = Path(root), binding
        self.binding_sha256 = digest(binding)
        manifest = self.root / "run_manifest.json"
        if self.root.exists():
            if not resume:
                raise FileExistsError("Existing resumable run requires explicit resume authorization")
            if not manifest.is_file():
                raise ValueError("Run directory without a complete identity manifest needs manual review")
            known = read_json(manifest)
            if (known.get("schema") != SCHEMA or known.get("binding") != binding
                    or known.get("binding_sha256") != self.binding_sha256):
                raise PermissionError("Checkpoint run identity/source/data/features/params differ")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            atomic_json(manifest, {
                "schema": SCHEMA, "binding": binding,
                "binding_sha256": self.binding_sha256, "created_utc": utc_now(),
                "resume_supported": "completed_stages_only",
                "svc_mid_fit_resume_supported": False,
            }, exclusive=True)

    def directory(self, stage: str) -> Path:
        return self.root / "checkpoints" / safe_stage(stage)

    def load(self, stage: str, *, parents: dict | None = None) -> dict | None:
        directory = self.directory(stage)
        if not directory.exists():
            return None
        record_path = directory / "checkpoint.json"
        if not record_path.is_file():
            raise ValueError(f"Uncommitted checkpoint directory: {directory}")
        record = read_json(record_path)
        if (record.get("schema") != SCHEMA or record.get("stage") != stage
                or record.get("binding_sha256") != self.binding_sha256
                or record.get("status") != "COMPLETE"
                or (parents is not None and record.get("parents") != parents)):
            raise ValueError(f"Checkpoint identity/dependencies differ: {stage}")
        files = record.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Checkpoint lacks artifacts: {stage}")
        if {p.name for p in directory.iterdir()} != set(files) | {"checkpoint.json"}:
            raise ValueError(f"Unexpected/missing checkpoint artifact: {stage}")
        for name, meta in files.items():
            if Path(name).name != name or name in (".", "..", "checkpoint.json"):
                raise ValueError("Unsafe checkpoint artifact name")
            path = directory / name
            if (not isinstance(meta, dict) or path.is_symlink() or not path.is_file()
                    or path.stat().st_size != meta.get("bytes")
                    or file_sha(path) != meta.get("sha256")):
                raise ValueError(f"Checkpoint checksum/size failed: {stage}/{name}")
        return record

    def reference(self, stage: str) -> str:
        self.load_required(stage)
        return file_sha(self.directory(stage) / "checkpoint.json")

    def load_required(self, stage: str, *, parents=None) -> dict:
        record = self.load(stage, parents=parents)
        if record is None:
            raise ValueError(f"Required completed checkpoint absent: {stage}")
        return record

    def save(self, stage: str, data: dict, *, parents: dict | None = None,
             writers: dict | None = None) -> dict:
        destination = self.directory(stage)
        if destination.exists():
            raise FileExistsError(f"Never overwrite successful checkpoint: {stage}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
        temporary.mkdir()
        atomic_json(temporary / "data.json", data, exclusive=True)
        for name, writer in (writers or {}).items():
            if Path(name).name != name or name in ("data.json", "checkpoint.json", ".", ".."):
                raise ValueError("Invalid checkpoint artifact name")
            writer(temporary / name)
        files = {}
        for path in temporary.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("Checkpoint writers must produce regular files only")
            sync_file(path)
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
        record = {"schema": SCHEMA, "stage": stage, "status": "COMPLETE",
                  "binding_sha256": self.binding_sha256, "parents": parents or {},
                  "completed_utc": utc_now(), "files": files}
        atomic_json(temporary / "checkpoint.json", record, exclusive=True)
        if destination.exists():
            raise FileExistsError("Another process committed this checkpoint")
        os.rename(temporary, destination)
        return self.load_required(stage, parents=parents or {})

    def data(self, stage: str, *, parents=None) -> dict:
        self.load_required(stage, parents=parents)
        return read_json(self.directory(stage) / "data.json")

    def completed_k(self, parents: dict, ranked_features: list[str]) -> list[dict]:
        rows, gap = [], False
        for k in range(1, 101):
            stage = f"k_search/k_{k:03d}"
            record = self.load(stage, parents=parents)
            if record is None:
                gap = True
                continue
            if gap:
                raise ValueError("Non-contiguous k checkpoint history needs manual review")
            row = self.data(stage, parents=parents)
            if row.get("k") != k or row.get("features") != ranked_features[:k]:
                raise ValueError(f"Wrong k/ranking prefix at {stage}")
            accuracy = row.get("metrics", {}).get("accuracy")
            if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                    or not 0 <= accuracy <= 1):
                raise ValueError("Invalid saved Validation Accuracy")
            rows.append(row)
        return rows

    def claim_test_once(self, final_reference: str, session_id: str) -> None:
        """Claim before the first Test load. A partial claim also blocks retry."""
        self.load_required("final")
        if self.reference("final") != final_reference:
            raise ValueError("Final-model checkpoint changed before Test")
        atomic_json(self.root / "results" / "test_evaluation_claim.json", {
            "schema": SCHEMA, "binding_sha256": self.binding_sha256,
            "final_checkpoint_sha256": final_reference,
            "session_id": session_id, "started_utc": utc_now(),
            "test_evaluation_number": 1,
            "incomplete_claim_requires_manual_review": True,
        }, exclusive=True)


def verified_copy(source: Path, target: Path) -> None:
    """For publication only; equal existing artifacts may be reused."""
    if target.exists():
        if file_sha(source) != file_sha(target):
            raise FileExistsError(target)
        return
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if file_sha(source) != file_sha(target):
        raise OSError("Publication checksum mismatch")
