"""Stage9-only paths, immutable records and integrity; no selection mutation."""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
METHODS = {"A1": ("phuong_anA", "MI_RFE_SVM_A1_STEP5_TARGET1", 5),
           "B": ("phuong_anB_step2", "MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2", 2)}
FORMAT = "stage9.svm-final.v1"
SCALER_PROTOCOL = "new StandardScaler.fit_transform(C-float32 selected Train then Validation); no Test fit"
SVM_PARAMS = dict(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", shrinking=True,
                  probability=False, cache_size=10000, max_iter=20000, random_state=42)
SCIENTIFIC = dict(scientific_status="EXPERIMENTAL_NON_OFFICIAL", official_status="OFFICIAL_BLOCKED",
                  official_reproduction=False, label_origin="PROVISIONAL_FOLDER_PROXY_NOT_GROUND_TRUTH")


class Blocked(ValueError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def info(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def once(path, value):
    """Exclusive remote create + fsync + readback. A partial file stays BLOCKED."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    if read(path) != value:
        raise Blocked("JSON readback mismatch: " + str(path))


def mutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    once(tmp, value)
    os.replace(tmp, path)


def files_checked(root, files):
    root = Path(root).resolve()
    for name, meta in files.items():
        p = Path(name)
        target = root / p
        if (p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name
                or target.is_symlink() or root not in target.resolve().parents or info(target) != meta):
            raise Blocked("Artifact path/SHA/size mismatch: " + name)


def intact(value):
    if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise Blocked("Identity self-digest mismatch")
    return value


def paths(request):
    if request["method"] not in METHODS or request["config"] not in CONFIGS:
        raise Blocked("Unknown method/config")
    if not re.fullmatch(r"RUN_S9_[A-Za-z0-9_-]{1,54}", request["run_id"]):
        raise Blocked("Stage9 run_id must begin RUN_S9_")
    dataset, output, local, source = (Path(request[k]).resolve() for k in ("dataset", "output", "local_root", "source"))
    if dataset.name != "CICFLOWMETER_FEATURE_CSV" or output != dataset.parent / "experiment_results":
        raise Blocked("Use the original CICFLOW dataset and sibling experiment_results")
    if any(a == b or a in b.parents or b in a.parents for a, b in
           ((source, dataset), (local, dataset), (local, output))):
        raise Blocked("Source, dataset, local cache and output must remain separated")
    if request.get("production", True) and (dataset != Path("/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV")
            or local != Path("/content/stage9_svm_final")):
        raise Blocked("Production Drive/local roots differ")
    return (local / request["method"] / request["config"],
            output / "stage9_svm_final" / request["method"] / request["config"])


def selection_path(request):
    root = Path(request["output"]).resolve() / METHODS[request["method"]][0] / "mi_rfe_svm" / request["config"]
    run = request["selection_run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run):
        raise Blocked("Unsafe selection run_id")
    return root / run


def confirmation(request, phase):
    paths(request)
    return (f"{phase.upper()} EXPERIMENTAL SVM {request['method']} {request['config']}/{request['run_id']} "
            f"SELECTION {request['selection_run_id']}")
