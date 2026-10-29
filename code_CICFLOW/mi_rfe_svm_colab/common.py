"""Contracts and local atomic I/O; original scientific sources stay read only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "cicflow.svm.local-5h.v1"
DRIVE_CHECKPOINT_INTERVAL_SECONDS = 18000
LOCAL_PROGRESS_AFTER_EACH_K = True
DRIVE_CHECKPOINT_AFTER_EACH_K = False
SCIENTIFIC = dict(scientific_status="PROVISIONAL_RESEARCH", official_status="OFFICIAL_BLOCKED",
                  official_reproduction=False, capture_session_independence="UNKNOWN",
                  label_origin="SCENARIO_PROXY")
CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
SVM_PARAMS = dict(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced",
                  shrinking=True, probability=False, cache_size=10000, max_iter=20000, random_state=42)
RFE_PARAMS = dict(n_estimators=100, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42)


class Blocked(ValueError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024**2), b""):
            h.update(part)
    return h.hexdigest()


def info(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    if exclusive:
        # link is atomic no-clobber on the LOCAL filesystem; Drive uses xb below.
        os.link(tmp, path)
        tmp.unlink()
    else:
        os.replace(tmp, path)


def drive_json_once(path, value):
    # A partial remote marker is deliberately never accepted as committed.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    if read(path) != value:
        raise OSError("Drive JSON verification failed: " + str(path))


def check_files(root, records):
    for name, record in records.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or (Path(root) / name).is_symlink()
                or info(Path(root) / name) != record):
            raise Blocked("Checksum/path mismatch: " + name)


def core(source):
    path = Path(source).resolve() / "colab/cicflow_reference_pipeline.py"
    name = "_svm5h_reference_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    module.activate_source(source)
    if module.SVM_PARAMS != SVM_PARAMS or module.RFE_PARAMS["svm"] != RFE_PARAMS:
        raise Blocked("Existing scientific parameters differ; reconciliation required")
    return module


def snapshot(source):
    source = Path(source).resolve()
    root = source.parent
    files = sorted(p for p in (source / "mi_rfe_svm_colab").rglob("*")
                   if p.is_file() and p.suffix in (".py", ".json", ".ipynb", ".txt")
                   and not any(s in p.parts for s in ("__pycache__", "reports", "runs", ".ipynb_checkpoints")))
    files += [source / p for p in core(source).SOURCE_FILES]
    files += [root / p for p in ("huyen_code/ciciomt24_training.py", "colab/svm_resumable_training.py",
                                "colab/svm_resumable_store.py", "notebooks/COMPACT_MI_RFE_ColabPro.ipynb")]
    files += sorted((root / "CODE_ctu13").glob("*.py"))
    records = {p.relative_to(root).as_posix(): sha(p) for p in files}
    return {"files": records, "sha256": digest(records)}


def paths(request):
    config, tag = request["config"], request.get("run_tag", "run01")
    if config not in CONFIGS or not re.fullmatch(r"run[0-9]{2,6}", tag):
        raise Blocked("Invalid configuration/run tag")
    dataset = Path(request["dataset"]).resolve()
    output = Path(request["output"]).resolve()
    if dataset.name != "CICFLOWMETER_FEATURE_CSV" or output != dataset.parent / "experiment_results":
        raise Blocked("Use MyDrive/CICFLOWMETER_FEATURE_CSV and MyDrive/experiment_results")
    local = Path(request.get("local_root", "/content/mi_rfe_svm_5h")).resolve()
    if local == output or output in local.parents or dataset == local or dataset in local.parents:
        raise Blocked("Local cache must be outside Drive/dataset")
    if request.get("production", True) and (Path("/content") not in local.parents
            or Path("/content/drive") == local or Path("/content/drive") in local.parents):
        raise Blocked("Production local cache must be under /content, outside /content/drive")
    name = f"mi_rfe_svm_{config}_{tag}"
    return local / name, output / name


def contract(source):
    return {**core(source).contract("svm"), "adapter_format": FORMAT, **SCIENTIFIC,
            "drive_interval_seconds": DRIVE_CHECKPOINT_INTERVAL_SECONDS,
            "svc_input": "C-order float64 converted AFTER reference float32 scaling",
            "partial_svc_resume": False}


def redact(text):
    text = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", str(text))
    text = re.sub(r'''(?i)((?:token|password|passwd|secret|api[_-]?key)["']?\s*[=:]\s*)(["'])(.*?)\2''',
                  lambda m: m[1] + m[2] + "[REDACTED]" + m[2], text)
    text = re.sub(r"(?i)(token|password|passwd|secret|api[_-]?key)([\s=:]+)([^\s,;]+)", r"\1\2[REDACTED]", str(text))
    return re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
