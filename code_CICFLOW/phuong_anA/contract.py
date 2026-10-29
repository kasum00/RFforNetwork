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
    files = [p for p in (source / "phuong_anA").rglob("*") if p.is_file()
             and p.suffix in (".py", ".json", ".ipynb", ".txt")
             and not any(part in p.parts for part in ("__pycache__", "reports", "runs", ".ipynb_checkpoints"))]
    for folder in ("mi_rfe_svm_colab",):
        files += [p for p in (source / folder).glob("*.py")]
    files += [source / p for p in core(source).SOURCE_FILES]
    files += [source / p for p in ("stage3_v2_windows.py", "stage29_v2_allocation.py", "stage29_cplus_integrity.py")]
    records = {p.relative_to(source).as_posix(): sha(p) for p in sorted(set(files))}
    return {"files": records, "sha256": digest(records)}


def confirmation(request):
    return f"{request['mode'].upper()} EXPERIMENTAL A1 {request['config']}/{request['run_id']}"
