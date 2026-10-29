"""Independent A1 identity and paths; immutable A0 helpers are reused."""
from __future__ import annotations
import json
import re
from pathlib import Path
from mi_rfe_svm_colab.common import (Blocked, atomic, check_files, core, digest,
    drive_json_once, info, read, redact, sha, utc)

ALGORITHM = "MI_RFE_SVM_A1_STEP5_TARGET1"
FORMAT = "phuong_anA.selection.v1"
STATUS = "EXPERIMENTAL_NON_OFFICIAL"
TIE_POLICY = "sklearn_rank_asc; removal_squared_importance_desc; original_MI_position_asc"
SCIENTIFIC = {"scientific_status": STATUS, "official_status": "OFFICIAL_BLOCKED",
              "official_reproduction": False, "test_values_read": False,
              "final_refit_authorized": False}

# Actual imports used by A1 or by the reused A0 modules at import time.
# Historical Stage3/29 generators are not executed by A1. Their frozen
# provenance is bound through evidence/cicflow_split_evidence.json instead.
RUNTIME_SOURCE_FILES = (
    "colab/cicflow_reference_pipeline.py",
    "mi_rfe_svm_colab/__init__.py", "mi_rfe_svm_colab/common.py",
    "mi_rfe_svm_colab/runner.py", "mi_rfe_svm_colab/dataset.py",
    "mi_rfe_svm_colab/local_progress.py", "mi_rfe_svm_colab/preflight.py",
    "mi_rfe_svm_colab/artifact_reuse.py", "mi_rfe_svm_colab/drive_checkpoint.py",
    "phuong_anA/evidence/cicflow_split_evidence.json",
)


def scientific(source):
    result = json.loads(json.dumps(core(source).contract("svm")))
    result.update(variant=ALGORITHM, scientific_status=STATUS, official_status="OFFICIAL_BLOCKED",
                  reference_status="EXPERIMENTAL_STEP5_NOT_EQUIVALENT_TO_A0_STEP1",
                  final_scaler_and_classifier_fit_role="NOT_AUTHORIZED_SELECTION_ONLY",
                  test_evaluation_calls=0)
    result["rfe"].update(step=5, n_features_to_select=1, tie_semantics=TIE_POLICY)
    return result


def paths(request):
    config, run_id = request["config"], request["run_id"]
    if not re.fullmatch(r"FW[1-9][0-9]*_FS[1-9][0-9]*", config):
        raise Blocked("Use the actual CICFLOW folder configuration, e.g. FW256_FS8")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise Blocked("Unsafe run ID")
    source, dataset, output, local = (Path(request[k]).resolve() for k in ("source", "dataset", "output", "local_root"))
    if dataset.name != "CICFLOWMETER_FEATURE_CSV" or output != dataset.parent / "experiment_results":
        raise Blocked("Require MyDrive/CICFLOWMETER_FEATURE_CSV and MyDrive/experiment_results")
    if request.get("production", True) and (local != Path("/content/mi_rfe_svm_5h")
            or dataset.parent != Path("/content/drive/MyDrive")):
        raise Blocked("Keep DRIVE=/content/drive/MyDrive and LOCAL_ROOT=/content/mi_rfe_svm_5h")
    if source == dataset or source in dataset.parents or dataset in source.parents:
        raise Blocked("Source and dataset must have separate roots")
    return (local / "phuong_anA" / "mi_rfe_svm" / config / run_id,
            output / "phuong_anA" / "mi_rfe_svm" / config / run_id)


def source_snapshot(source):
    source = Path(source).resolve()
    required = [source / p for p in RUNTIME_SOURCE_FILES]
    missing = [p.relative_to(source).as_posix() for p in required if not p.is_file()]
    if missing:
        raise Blocked("Missing required A1 runtime source: " + ", ".join(missing))
    files = [p for p in (source / "phuong_anA").rglob("*") if p.is_file()
             and p.suffix in (".py", ".json", ".ipynb", ".txt")
             and not any(part in p.parts for part in ("__pycache__", "reports", "runs", ".ipynb_checkpoints"))]
    for folder in ("mi_rfe_svm_colab",):
        files += [p for p in (source / folder).glob("*.py")]
    files += [source / p for p in core(source).SOURCE_FILES]
    files += required
    missing = [p.relative_to(source).as_posix() for p in files if not p.is_file()]
    if missing:
        raise Blocked("Missing required A1 runtime source: " + ", ".join(sorted(set(missing))))
    records = {p.relative_to(source).as_posix(): sha(p) for p in sorted(set(files))}
    return {"files": records, "sha256": digest(records)}


def confirmation(request):
    return f"{request['mode'].upper()} EXPERIMENTAL A1 {request['config']}/{request['run_id']}"
