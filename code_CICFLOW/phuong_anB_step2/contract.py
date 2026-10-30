"""B namespace and exact source/scientific binding; A1 output is never imported."""
import json
import re
from pathlib import Path
from mi_rfe_svm_colab.common import (Blocked, atomic, check_files, core, digest,
    drive_json_once, info, read, redact, sha, utc)

ALGORITHM = "MI_RFE_SVM_B_STEP2_COARSE6_RADIUS5_V2"
FORMAT = "phuong_anB_step2.selection.v2"
RFE_STEP = 2
CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
NAMESPACE = "phuong_anB_step2"
TIE_POLICY = "sklearn_rank_asc; removal_squared_importance_desc; original_MI_position_asc"
SCIENTIFIC = {"scientific_status": "EXPERIMENTAL_NON_OFFICIAL", "official_status": "OFFICIAL_BLOCKED",
              "official_reproduction": False, "test_values_read": False,
              "final_refit_authorized": False}


def paths(request):
    config, run_id = request["config"], request["run_id"]
    if config not in CONFIGS or not re.fullmatch(r"RUN_B2_[A-Za-z0-9_-]{1,54}", run_id):
        raise Blocked("Use a CICFLOW config and a fresh B ID beginning RUN_B2_; A1 IDs are forbidden")
    source, dataset, output, local = (Path(request[k]).resolve() for k in ("source", "dataset", "output", "local_root"))
    if dataset.name != "CICFLOWMETER_FEATURE_CSV" or output != dataset.parent / "experiment_results":
        raise Blocked("B requires the same MyDrive/CICFLOWMETER_FEATURE_CSV and experiment_results as A1")
    if source == dataset or source in dataset.parents or dataset in source.parents:
        raise Blocked("Source and dataset must be separate")
    if local == output or local == dataset or output in local.parents or dataset in local.parents:
        raise Blocked("Local arrays must be separate from Drive/dataset")
    if request.get("production", True) and (dataset.parent != Path("/content/drive/MyDrive")
            or local != Path("/content/mi_rfe_svm_b_step2")):
        raise Blocked("Production requires mounted /content/drive/MyDrive and local /content/mi_rfe_svm_b_step2")
    return (local / NAMESPACE / "mi_rfe_svm" / config / run_id,
            output / NAMESPACE / "mi_rfe_svm" / config / run_id)


def scientific(source, policy):
    result = json.loads(json.dumps(core(source).contract("svm")))
    result.update(variant=ALGORITHM, **SCIENTIFIC,
        reference_status="EXPERIMENTAL_STEP2_COARSE_FINE_NOT_EQUIVALENT_TO_STEP1_OR_A1",
        final_scaler_and_classifier_fit_role="NOT_AUTHORIZED_SELECTION_ONLY",
        test_evaluation_calls=0, search_policy=policy)
    result["rfe"].update(step=RFE_STEP, n_features_to_select=1, tie_semantics=TIE_POLICY)
    result.pop("k_grid", None)
    result["k_criterion"] = "highest measured Validation Accuracy; ties smallest evaluated k"
    return result


def source_snapshot(source):
    source = Path(source).resolve()
    files = [p for p in (source / NAMESPACE).rglob("*") if p.is_file()
             and p.suffix in (".py", ".json", ".txt")
             and not any(part in p.parts for part in ("__pycache__", "reports", "runs", "outputs"))]
    files += list((source / "mi_rfe_svm_colab").glob("*.py"))
    files += [source / p for p in core(source).SOURCE_FILES]
    files += [source.parent / "notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb"]
    required = [source / NAMESPACE / p for p in ("contract.py", "dataset.py", "policy.py", "history.py",
        "rfe.py", "pipeline.py", "store.py", "telemetry.py", "preflight.py", "results.py",
        "controller.py", "cli.py", "selftest.py", "environment.py", "io_utils.py",
        "notebook_helpers.py", "search.py", "evidence/dataset_reference.json",
        "evidence/cicflow_split_evidence.json", "evidence/approved_helpers.json",
        "evidence/original_policy_excerpt.py")]
    files += required
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise Blocked("Missing B runtime/source evidence: " + ", ".join(missing))
    records = {p.relative_to(source.parent).as_posix(): sha(p) for p in sorted(set(files))}
    return {"files": records, "sha256": digest(records)}


def confirmation(request):
    return f"{request['mode'].upper()} EXPERIMENTAL B STEP2 {request['config']}/{request['run_id']}"
