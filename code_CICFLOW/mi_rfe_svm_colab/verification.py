"""Metadata/hash verification and final flat publication; never fit or predict."""
import math
import os
import shutil
from pathlib import Path
from .common import (Blocked, FORMAT, SCIENTIFIC, atomic, check_files, contract, digest,
                     drive_json_once, info, read, sha, utc)
from .artifact_reuse import validate_selection
from .drive_checkpoint import latest, state_files
from .local_progress import KProgress, choose

REQUIRED = ("run_identity.json", "mi_top100.json", "mi_feature_scores.csv", "mi_state.json",
            "rfe_state.json", "rfe_feature_ranking.csv", "top100_scaler_state.json", "top100_scaler.joblib",
            "k_search_results.json", "selection_frozen.json", "features.json", "model.joblib",
            "final_scaler.joblib", "final_state.json", "TEST_CLAIM.json", "test_result.json", "result.json",
            "events.jsonl", "progress.json", "resource_usage.json", "console_snapshot.log")
REQUIRED += ("hyperparameters.json", "preflight_audit.json")


def validate_state(state, identity, *, final_required=False, test_required=False, with_chain=True):
    state = Path(state)
    if (read(state / "run_identity.json") != identity or identity.get("format") != FORMAT
            or identity.get("sha256") != digest({k: v for k, v in identity.items() if k != "sha256"})
            or identity.get("spec", {}).get("experiment") != "mi_rfe_svm"
            or identity.get("contract") != contract(Path(__file__).resolve().parents[1])):
        raise Blocked("Run identity/branch digest invalid")
    top, ranked = validate_selection(state, identity)
    selection, frozen = read(state / "k_search_results.json"), read(state / "selection_frozen.json")
    rows = selection["results"]
    best = choose(rows, ranked, complete=True)
    if with_chain:
        history = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
        if history.rows != rows:
            raise Blocked("Selection differs from local k history")
    if (selection["best_k"] != best["k"] or frozen["best_k"] != best["k"]
            or selection["features"] != ranked[:best["k"]] or frozen["features"] != selection["features"]
            or frozen["identity"] != identity["sha256"] or frozen.get("test_used_for_selection") is not False
            or read(state / "features.json") != frozen["features"]
            or selection["best_validation_accuracy"] != best["accuracy"]
            or frozen["best_validation_accuracy"] != best["accuracy"]
            or frozen["k_search_sha256"] != sha(state / "k_search_results.json")):
        raise Blocked("Frozen selection/tie rule changed")
    scaler = read(state / "top100_scaler_state.json")
    if (scaler["binding"] != {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top}
            or scaler["fit_role"] != "train" or scaler["rows"] != identity["dataset"]["splits"]["train"]["rows"]):
        raise Blocked("Train Top100 scaler binding differs")
    check_files(state, scaler["files"])
    if final_required:
        final = read(state / "final_state.json")
        if (final["binding"] != {"identity": identity["sha256"], "frozen": digest(frozen)}
                or final["fit_role"] != "train+val" or final["fit_rows"] != sum(
                    identity["dataset"]["splits"][r]["rows"] for r in ("train", "val"))
                or set(final["files"]) != {"model.joblib", "final_scaler.joblib"}):
            raise Blocked("Final refit binding/rows incomplete")
        check_files(state, final["files"])
    if test_required:
        test, claim = read(state / "test_result.json"), read(state / "TEST_CLAIM.json")
        count = identity["dataset"]["splits"]["test"]["rows"]
        if (claim["identity"] != identity["sha256"] or claim["evaluation_number"] != 1
                or claim["final_state"] != digest(read(state / "final_state.json"))
                or test["identity"] != identity["sha256"] or test["claim"] != digest(claim)
                or test["test_evaluation_calls"] != 1 or test["test_rows_evaluated"] != count):
            raise Blocked("Test-once claim/result invalid")
        matrix = test["metrics"]["confusion_matrix_labels_0_benign_1_attack"]
        if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
                or any(type(n) is not int or n < 0 for row in matrix for n in row)
                or sum(map(sum, matrix)) != count
                or not math.isclose((matrix[0][0]+matrix[1][1])/count, test["metrics"]["accuracy"], abs_tol=1e-12)):
            raise Blocked("Test metrics/counts inconsistent")
    return {"status": "PASS", "best_k": best["k"], **SCIENTIFIC}


def result_record(state, identity):
    state = Path(state)
    frozen, test = read(state / "selection_frozen.json"), read(state / "test_result.json")
    rows = read(state / "k_search_results.json")["results"]
    nonconverged = [r["k"] for r in rows if r["fit"].get("converged") is False]
    status = "COMPLETED_WITH_CONVERGENCE_WARNING" if nonconverged or read(state / "final_state.json")["fit"].get("converged") is False else "PASS"
    return {"status": status, "identity": identity["sha256"], "experiment": "mi_rfe_svm",
            "config": identity["spec"]["config"], "best_k": frozen["best_k"], "selected_features": frozen["features"],
            "validation_accuracy": frozen["best_validation_accuracy"], "metrics": test["metrics"],
            "nonconverged_selection_k": nonconverged, "test_evaluation_calls": 1,
            "test_rows_evaluated": test["test_rows_evaluated"],
            "rows_by_split": {r: v["rows"] for r, v in identity["dataset"]["splits"].items()}, **SCIENTIFIC}


def publish(state, remote, identity):
    state, remote = Path(state), Path(remote)
    validate_state(state, identity, final_required=True, test_required=True)
    result = result_record(state, identity)
    if (state / "result.json").exists():
        if read(state / "result.json") != result:
            raise Blocked("Stored result differs from verified selection/Test")
    else:
        atomic(state / "result.json", result, exclusive=True)
    for name in REQUIRED:
        source, target = state / name, remote / name
        if target.exists():
            if info(source) != info(target):
                raise Blocked("Never overwrite a different existing artifact: " + name)
            continue
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8*1024**2)
            dst.flush()
            os.fsync(dst.fileno())
        if info(source) != info(target):
            raise OSError("Final publication verification failed: " + name)
    manifest = {"format": FORMAT, "identity": identity["sha256"],
                "files": {name: info(remote / name) for name in REQUIRED}}
    path = remote / "artifact_manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise Blocked("Existing final artifact manifest differs")
    else:
        drive_json_once(path, manifest)
    checkpoint = latest(remote, identity["sha256"])
    if not checkpoint:
        raise Blocked("No COMMITTED final Drive checkpoint")
    marker = {"format": FORMAT, "identity": identity["sha256"], "status": "COMPLETE",
              "manifest": info(path), "checkpoint": checkpoint["id"], "test_evaluation_calls": 1, **SCIENTIFIC}
    marker_path = remote / "RUN_COMPLETE.json"
    if marker_path.exists():
        if read(marker_path) != marker:
            raise Blocked("Completed marker differs")
    else:
        drive_json_once(marker_path, marker)
    return verify(remote)


def verify(remote):
    remote = Path(remote)
    marker, manifest, identity = read(remote / "RUN_COMPLETE.json"), read(remote / "artifact_manifest.json"), read(remote / "run_identity.json")
    if (marker.get("format") != FORMAT or marker.get("status") != "COMPLETE"
            or marker.get("identity") != identity["sha256"] or marker.get("test_evaluation_calls") != 1
            or marker.get("manifest") != info(remote / "artifact_manifest.json")
            or manifest.get("identity") != identity["sha256"] or set(manifest.get("files", {})) != set(REQUIRED)
            or read(remote / "OWNER.json") != {"format": FORMAT, "identity": identity["sha256"]}):
        raise Blocked("Final manifest/completion/owner invalid")
    check_files(remote, manifest["files"])
    if (read(remote / "hyperparameters.json") != identity["contract"]
            or read(remote / "preflight_audit.json").get("status") != "PASS"
            or read(remote / "preflight_audit.json").get("identity") != identity
            or any(marker.get(k) != v for k, v in SCIENTIFIC.items())):
        raise Blocked("Scientific contract/preflight/completion status differs")
    checkpoint = latest(remote, identity["sha256"])
    if (not checkpoint or checkpoint["id"] != marker["checkpoint"] or checkpoint["completed_k"] != 100
            or checkpoint.get("final_state_sha256") != sha(remote / "final_state.json")
            or checkpoint.get("selection_sha256") != sha(remote / "selection_frozen.json")
            or checkpoint.get("test_result_sha256") != sha(remote / "test_result.json")
            or checkpoint.get("result_sha256") != sha(remote / "result.json")):
        raise Blocked("Completed run does not bind the latest verified Drive bundle")
    outcome = validate_state(remote, identity, final_required=True, test_required=True, with_chain=False)
    if read(remote / "result.json") != result_record(remote, identity):
        raise Blocked("Final result differs")
    return {**outcome, "directory": str(remote), "result_status": read(remote / "result.json")["status"],
            "metrics": read(remote / "test_result.json")["metrics"], "test_evaluation_calls": 1,
            "verification_scope": "saved artifacts/hashes only; no dataset values/model deserialization/prediction"}
