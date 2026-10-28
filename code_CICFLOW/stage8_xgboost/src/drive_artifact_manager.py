"""Validate saved artifacts; no predictions or dataset values in verification."""
from pathlib import Path
from .common import Blocked, FORMAT, SCIENTIFIC, EXPECTED, check_files, digest, file_record, read, write

REQUIRED = ("run_manifest.json", "configuration.json", "preflight.json", "feature_selection.json",
            "models/final_model.ubj", "models/final_scaler.joblib", "models/booster_config.json",
            "metrics/validation_metrics.json", "metrics/test_metrics.json", "metrics/confusion_matrix.csv",
            "reports/EXPERIMENT_REPORT.md", "checkpoints/booster/protocol_state.json", "RUN_VALIDATED.json")

def verify(directory, *, require_complete=True):
    directory = Path(directory)
    for name in REQUIRED:
        if not (directory / name).is_file():
            raise Blocked("Missing artifact: " + name)
    manifest = read(directory / "run_manifest.json")
    identity = manifest["identity"]
    if manifest["format"] != FORMAT or digest(identity) != manifest["identity_sha256"]:
        raise Blocked("Run identity mismatch")
    validated = read(directory / "RUN_VALIDATED.json")
    check_files(directory, validated["files"])
    if not (set(REQUIRED) - {"RUN_VALIDATED.json"}) <= set(validated["files"]):
        raise Blocked("Incomplete artifact hash manifest")
    state = read(directory / "checkpoints/booster/protocol_state.json")
    test = read(directory / "metrics/test_metrics.json")
    features = read(directory / "feature_selection.json")
    names = identity["dataset"]["binding"]["names"]
    expected_context = validated["checkpoint_identity"]
    if any(state.get(k) != v for k, v in expected_context.items()):
        raise Blocked("Protocol scientific identity mismatch")
    if (state.get("test_state") != "TEST_COMPLETED" or state.get("test_evaluation_calls") != 1
            or state.get("official_status") != "COMPLETED"
            or state.get("test_metrics_sha256") != file_record(directory / "metrics/test_metrics.json")["sha256"]
            or state.get("final_refit_status") != "STAGE_COMPLETE" or state.get("validation_fit_status") != "STAGE_COMPLETE"
            or test["test_evaluation_calls"] != 1 or test["rows"] != EXPECTED[identity["config"]]["test"]
            or features["selected_features"] != names or len(names) != 380 or len(set(names)) != 380
            or features["best_k"] is not None or features["status"] != "NOT_APPLICABLE"
            or validated["completed_rounds"] != 200 or validated["identity_sha256"] != digest(identity)):
        raise Blocked("Baseline features, Test-once, rounds or final-refit evidence invalid")
    final = read(directory / "checkpoints/final_refit_artifacts.json")
    check_files(directory, final["files"])
    val = read(directory / "metrics/validation_metrics.json")
    if (final["completed_rounds"] != 200 or final["identity_sha256"] != digest(identity)
            or final["rows"] != EXPECTED[identity["config"]]["train"]+EXPECTED[identity["config"]]["val"]
            or final["frozen_features_sha256"] != file_record(directory / "feature_selection.json")["sha256"]
            or val["identity_sha256"] != digest(identity)
            or state.get("validation_metrics_sha256") != file_record(directory / "metrics/validation_metrics.json")["sha256"]):
        raise Blocked("Final refit artifact binding invalid")
    if require_complete:
        complete = read(directory / "RUN_COMPLETE.json")
        if (complete["status"] != "COMPLETE" or complete["validated_sha256"] != file_record(directory / "RUN_VALIDATED.json")["sha256"]
                or complete["identity_sha256"] != digest(identity) or complete["worker_exit_code"] != 0):
            raise Blocked("Completion marker invalid")
    return {"status": "PASS", "config": identity["config"], "run_name": directory.name,
            "directory": str(directory), "metrics": test["metrics"], "test_evaluation_calls": 1, **SCIENTIFIC}

def complete_after_worker_exit(directory, exit_code):
    if exit_code != 0:
        raise Blocked("Nonzero worker exit: cannot publish RUN_COMPLETE")
    result = verify(directory, require_complete=False)
    path = Path(directory) / "RUN_COMPLETE.json"
    if not path.exists():
        manifest = read(Path(directory) / "run_manifest.json")
        write(path, {"status": "COMPLETE", "worker_exit_code": 0, **SCIENTIFIC,
                     "identity_sha256": manifest["identity_sha256"],
                     "validated_sha256": file_record(Path(directory) / "RUN_VALIDATED.json")["sha256"]})
    return verify(directory)
