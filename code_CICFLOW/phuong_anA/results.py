"""Validate A1 selection, publish saved Validation metrics only, never Test."""
import csv
import json
import math
import shutil
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, SCIENTIFIC, TIE_POLICY, Blocked, atomic,
                       check_files, digest, drive_json_once, info, read, sha)
from .rfe import validate_result
from .store import ledger
from mi_rfe_svm_colab.local_progress import KProgress, choose

OUTPUTS = ("selection.json", "k_search_results.json", "features.json", "rfe_feature_ranking.csv",
           "mi_top100.json", "selection_report.md", "A0_A1_COMPARISON.json")


def validate_mi(state, identity):
    state = Path(state)
    mi, top = read(state / "mi_state.json"), read(state / "mi_top100.json")
    check_files(state, mi["files"])
    with (state / "mi_feature_scores.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    scores = [float(row[1]) for row in rows]
    if (mi["identity"] != identity["sha256"] or len(rows) != 380
            or set(r[0] for r in rows) != set(identity["dataset"]["names"])
            or [r[0] for r in rows[:100]] != top or len(top) != 100 or len(set(top)) != 100
            or scores != sorted(scores, reverse=True) or any(not math.isfinite(x) or x < 0 for x in scores)):
        raise Blocked("MI mapping/scores/identity invalid")
    return top


def validate_ranking(state, identity):
    state = Path(state)
    top = validate_mi(state, identity)
    scaler = read(state / "top100_scaler_state.json")
    if scaler["binding"] != {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top}:
        raise Blocked("A1 Train scaler identity differs")
    check_files(state, scaler["files"])
    if scaler["fit_role"] != "train" or scaler["rows"] != identity["dataset"]["splits"]["train"]["rows"]:
        raise Blocked("Scaler did not fit full Train only")
    rfe = read(state / "rfe_state.json")
    if rfe["binding"] != {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
                          "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}:
        raise Blocked("A1 ranking MI/scaler binding differs")
    check_files(state, rfe["files"])
    with (state / "rfe_feature_ranking.csv").open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rfe["result"]["records"]]
    if csv_rows != expected:
        raise Blocked("A1 CSV order/importance/rank mapping differs")
    return top, validate_result(rfe["result"], top)


def validate_selection(state, identity):
    state = Path(state)
    top, ranked = validate_ranking(state, identity)
    history = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
    validation_rows = identity["dataset"]["splits"]["val"]["rows"]
    for row in history.rows:
        for key in ("accuracy", "precision_attack", "recall_attack", "recall_benign", "f1_attack", "macro_f1", "balanced_accuracy"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise Blocked("Missing/invalid full Validation metrics: " + key)
        matrix = row.get("confusion_matrix_labels_0_benign_1_attack", [])
        if (len(matrix) != 2 or any(len(r) != 2 for r in matrix)
                or any(type(n) is not int or n < 0 for r in matrix for n in r)
                or sum(map(sum, matrix)) != validation_rows
                or not math.isclose((matrix[0][0]+matrix[1][1])/validation_rows, row["accuracy"], abs_tol=1e-12)):
            raise Blocked("Validation confusion matrix/count/accuracy mismatch")
    best = choose(history.rows, ranked, complete=True)
    frozen, search = read(state / "selection_frozen.json"), read(state / "k_search_results.json")
    selection = read(state / "selection.json")
    if (search["results"] != history.rows or frozen["identity"] != identity["sha256"]
            or frozen["k_search_sha256"] != sha(state / "k_search_results.json")
            or frozen["best_k"] != best["k"] or frozen["features"] != ranked[:best["k"]]
            or frozen["best_validation_accuracy"] != best["accuracy"]
            or read(state / "features.json") != frozen["features"]
            or search["best_k"] != best["k"] or selection.get("algorithm") != ALGORITHM
            or selection.get("identity") != identity["sha256"] or selection.get("best_k") != best["k"]
            or selection.get("selected_features") != frozen["features"]
            or selection.get("validation_metrics") != {k: v for k, v in best.items()
                if k not in ("k", "features", "fit", "predict_seconds", "started", "finished", "rss_before_fit_bytes", "rss_after_predict_bytes")}
            or selection.get("test_evaluation_calls") != 0
            or any(selection.get(k) != v for k, v in SCIENTIFIC.items())):
        raise Blocked("A1 full k/selection chain invalid")
    return {"status": "PASS", "best_k": best["k"], "validation_accuracy": best["accuracy"], **SCIENTIFIC}


def create_selection(state, identity, rows, ranked, frozen):
    state = Path(state)
    best = choose(rows, ranked, complete=True)
    metric_keys = ("k", "features", "fit", "predict_seconds", "started", "finished", "rss_before_fit_bytes", "rss_after_predict_bytes")
    selection = {"algorithm": ALGORITHM, "identity": identity["sha256"], "config": identity["config"],
                 "run_id": identity["run_id"], "best_k": best["k"], "selected_features": frozen["features"],
                 "validation_metrics": {k: v for k, v in best.items() if k not in metric_keys},
                 "tie_policy": TIE_POLICY, "test_evaluation_calls": 0,
                 "selection_nonconverged_k": [r["k"] for r in rows if r["fit"].get("converged") is False],
                 "k_fit_seconds_sum": sum(r["fit"]["seconds"] for r in rows), **SCIENTIFIC}
    rfe = read(state / "rfe_state.json")["result"]
    comparison = {"status": "NO_MEASURED_A0_COMPARISON", "A0": {"step": 1, "target": 1,
        "theoretical_rf_fits_for_top100": 100, "measured_rf_seconds": None,
        "best_k": None, "validation_accuracy": None, "test_metrics": None},
        "A1": {"step": 5, "target": 1, "theoretical_rf_fits_for_top100": 21,
            "completed_elimination_rounds": len(rfe["steps"]), "unique_sklearn_ranks": rfe["unique_sklearn_rank_count"],
            "recorded_elimination_fit_seconds_sum": sum(s["fit_seconds"] for s in rfe["steps"]),
            "recorded_final_rf_fit_seconds": rfe["final_estimator_fit_seconds"],
            "best_k": best["k"], "validation_accuracy": best["accuracy"], "test_metrics": None},
        "speedup_ratio": None, "not_a_claim_of_equal_ranking_or_metrics": True, **SCIENTIFIC}
    for name, value in (("selection.json", selection), ("A0_A1_COMPARISON.json", comparison)):
        if (state / name).exists():
            if read(state / name) != value:
                raise Blocked("Existing selection/report differs")
        else:
            atomic(state / name, value, exclusive=True)
    report = (f"# A1 selection\n\nStatus: EXPERIMENTAL_NON_OFFICIAL\n\nConfig: {identity['config']}\n"
              f"Run: {identity['run_id']}\nBest k: {best['k']}\nValidation Accuracy: {best['accuracy']}\n"
              f"RFE step=5, target=1; sklearn rank groups={rfe['unique_sklearn_rank_count']}.\n"
              f"Tie policy: {TIE_POLICY}\n\nAll k=1..100 completed. Test evaluations: 0. Final refit not authorized.\n"
              "No measured A0 runtime/accuracy comparison is available. Proxy labels/capture limitations remain.\n")
    if (state / "selection_report.md").exists():
        if (state / "selection_report.md").read_text(encoding="utf-8") != report:
            raise Blocked("Existing selection report differs; do not overwrite")
    else:
        (state / "selection_report.md").write_text(report, encoding="utf-8")
    validate_selection(state, identity)


def publish(store):
    outcome = validate_selection(store.state, store.identity)
    selection_cp = store.saved("selection")
    if selection_cp is None:
        raise Blocked("Selection must be COMMITTED before publication")
    result_root = store.remote / "results"
    result_root.mkdir(exist_ok=True)
    for name in OUTPUTS:
        source, target = store.state / name, result_root / name
        if target.exists():
            if info(target) != info(source):
                raise Blocked("Existing result differs; do not overwrite: " + name)
        else:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*1024**2)
            if info(target) != info(source):
                raise Blocked("Result copy verification failed")
    manifest = {"identity": store.identity["sha256"], "algorithm": ALGORITHM,
                "files": {name: info(result_root / name) for name in OUTPUTS}}
    path = result_root / "artifact_manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise Blocked("Publication manifest differs")
    else:
        drive_json_once(path, manifest)
    marker = {"format": FORMAT, "identity": store.identity["sha256"], "status": "SELECTION_COMPLETE",
              "checkpoint": selection_cp["sha256"], "manifest": info(path), "test_evaluation_calls": 0, **SCIENTIFIC}
    marker_path = result_root / "SELECTION_COMPLETE.json"
    if marker_path.exists():
        if read(marker_path) != marker:
            raise Blocked("Completed result differs")
    else:
        drive_json_once(marker_path, marker)
    return {**outcome, "directory": str(store.remote), "phase_status": "SELECTION_COMPLETE"}
