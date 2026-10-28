"""Small file/identity helpers; no subprocess or fit on import."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

FORMAT = "cicflow.stage8.baseline.all4.v1"
CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
EXPECTED = dict(zip(CONFIGS, (
    {"train": 326157, "val": 104147, "test": 103494},
    {"train": 163203, "val": 52145, "test": 51824},
    {"train": 311423, "val": 96408, "test": 95865},
    {"train": 155814, "val": 48258, "test": 47991},
)))
SCIENTIFIC = {"scientific_status": "PROVISIONAL_RESEARCH", "official_status": "OFFICIAL_BLOCKED",
              "capture_session_independence": "UNKNOWN", "official_reproduction": False}

class Blocked(ValueError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    def reject(value):
        raise Blocked("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)

def write(path, value, *, exclusive=True):
    raw = json.dumps(value, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    if exclusive and path.exists():
        raise FileExistsError(path)
    os.replace(tmp, path)

def file_record(path):
    return {"sha256": hash_file(path), "bytes": Path(path).stat().st_size}

def check_files(root, records):
    root = Path(root).resolve()
    for name, record in records.items():
        path = (root / name).resolve()
        if root not in path.parents or file_record(path) != record:
            raise Blocked("Artifact missing/hash mismatch/path escape: " + name)

def load_core(source):
    path = Path(source).resolve() / "colab/cicflow_reference_pipeline.py"
    key = "_cicflow_stage8_data_core"
    if key not in sys.modules or Path(sys.modules[key].__file__).resolve() != path:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]

def snapshot(source):
    source = Path(source).resolve()
    area = source / "stage8_xgboost"
    records = {}
    for path in sorted(area.rglob("*")):
        rel = path.relative_to(source)
        if any(part in {"runs", "logs", "reports", "__pycache__", ".git"} for part in rel.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".json", ".ipynb", ".txt"}:
            records[rel.as_posix()] = hash_file(path)
    for name in ("colab/cicflow_reference_pipeline.py", "stage4_mi/mi_selection.py", "stage5_rfe/rfe_selection.py"):
        records[name] = hash_file(source / name)
    return {"files": records, "sha256": digest(records)}

def contract(source):
    return read(Path(source) / "stage8_xgboost/configs/xgboost_contract.json")

def output_path(request):
    if request.get("experiment") != "baseline_xgb":
        raise Blocked("MI-RFE-XGB is not approved: no RF ranking/best-k reuse")
    config, tag = request["config"], request["run_tag"]
    if config not in CONFIGS or not re.fullmatch(r"run[0-9]{2,6}", tag):
        raise Blocked("Invalid configuration/run tag")
    dataset, result = Path(request["dataset"]).resolve(), Path(request["result_root"]).resolve()
    if dataset.name != "CICFLOWMETER_FEATURE_CSV" or result != dataset.parent / "experiment_results":
        raise Blocked("Use MyDrive/CICFLOWMETER_FEATURE_CSV and MyDrive/experiment_results")
    name = f"baseline_xgb_{config}_{tag}"
    return result / name

def ensure_owned(request):
    directory = output_path(request)
    owner = {"format": FORMAT, "config": request["config"], "experiment": "baseline_xgb", "run_name": directory.name}
    if directory.exists():
        if not (directory / "OWNER.json").is_file() or read(directory / "OWNER.json") != owner:
            raise Blocked("Foreign/legacy/RF output is protected: " + str(directory))
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / "OWNER.json", owner)
    return directory

def clean_git(source, expected):
    # Called only inside a user-launched Colab worker, never during editing.
    import subprocess
    if not re.fullmatch("[0-9a-fA-F]{40}", expected or ""):
        raise Blocked("Supply the exact 40-hex approved Git commit")
    repo = Path(source).resolve().parent
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    if head.lower() != expected.lower() or dirty:
        raise Blocked("Git commit mismatch or modified checkout")
    return head

def confirmation(request):
    if tuple(request.get("configs", ())) != CONFIGS:
        raise Blocked("Batch must contain the exact four ordered configurations, without duplicates")
    action = "RESUME" if request.get("resume") else "RUN"
    return (f"{action} STAGE8 baseline_xgb ALL4 {request['run_tag']} {request['backend']} "
            f"{request['git_commit'][:8]} {digest(request['budget'])[:8]}")
