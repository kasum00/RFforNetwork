"""Verify and publish B's measured Validation selection; no final model or Test."""
import csv
import math
import shutil
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, SCIENTIFIC, Blocked, atomic, check_files,
                       digest, drive_json_once, info, read, sha)
from .history import History, METRICS, validate_rows, check_committed_search
from .policy import winner
from .rfe import validate_result

OUTPUTS = ("selection.json", "k_search_results.json", "selection_frozen.json", "features.json",
           "rfe_feature_ranking.csv", "mi_top100.json", "mi_feature_scores.csv",
           "fine_plan.json", "statistics.json", "selection_report.md")


def validate_mi(state, identity):
    state = Path(state)
    mi, top = read(state / "mi_state.json"), read(state / "mi_top100.json")
    check_files(state, mi["files"])
    with (state / "mi_feature_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    scores = [float(row[1]) for row in rows]
    if (mi["identity"] != identity["sha256"] or len(rows) != 380
            or set(r[0] for r in rows) != set(identity["dataset"]["names"])
            or [r[0] for r in rows[:100]] != top or len(top) != 100 or len(set(top)) != 100
            or scores != sorted(scores, reverse=True) or any(not math.isfinite(x) or x < 0 for x in scores)):
        raise Blocked("MI mapping/scores/identity differs")
    return top


def validate_ranking(state, identity):
    state = Path(state)
    top = validate_mi(state, identity)
    scaler = read(state / "top100_scaler_state.json")
    if scaler["binding"] != {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top}:
        raise Blocked("Train scaler binding differs")
    check_files(state, scaler["files"])
    if scaler["fit_role"] != "train" or scaler["rows"] != identity["dataset"]["splits"]["train"]["rows"]:
        raise Blocked("Scaler did not fit full Train only")
    rfe = read(state / "rfe_state.json")
    if rfe["binding"] != {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
                          "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}:
        raise Blocked("B ranking MI/scaler binding differs")
    check_files(state, rfe["files"])
    with (state / "rfe_feature_ranking.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rfe["result"]["records"]]
    if saved != expected:
        raise Blocked("RFE CSV order/rank/importance mapping differs")
    return top, validate_result(rfe["result"], top)


def json_once(path, value):
    if path.exists():
        if read(path) != value:
            raise Blocked("Existing B artifact differs; no overwrite: " + str(path))
    else:
        atomic(path, value, exclusive=True)


def search_payload(history):
    best = winner(history.rows)
    fine_plan = history.fine_plan()
    return {"identity": history.identity["sha256"], "search_policy": history.policy,
            "coarse_grid": history.policy["coarse_grid"], "coarse_best_k": fine_plan["coarse_best_k"],
            "fine_radius": history.policy["fine_radius"], "fine_grid": fine_plan["fine_grid"],
            "evaluated_k": [r["k"] for r in history.rows], "results": history.rows,
            "best_k": best["k"], "features": history.ranked[:best["k"]],
            "best_validation_accuracy": best["accuracy"],
            "nonconverged_k": [r["k"] for r in history.rows if not r["fit"]["converged"]],
            "all_1_to_100_evaluated": set(r["k"] for r in history.rows) == set(range(1, 101))}


def measurements(state, identity, history):
    state = Path(state)
    rfe = read(state / "rfe_state.json")["result"]
    mi_seconds = read(state / "mi_state.json")["seconds"]
    rf_seconds = sum(row["fit_seconds"] for row in rfe["steps"]) + rfe["final_estimator_fit_seconds"]
    coarse = [row for row in history.rows if row["phase"] == "COARSE"]
    fine = [row for row in history.rows if row["phase"] == "FINE"]
    all_receipts = {}
    for path in sorted((state / "sessions").glob("session_*.json")):
        value = read(path)
        if (value["identity"] != identity["sha256"] or path.stem != "session_" + value["attempt"]
                or value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"})
                or type(value["wall_seconds"]) not in (int, float)
                or not math.isfinite(value["wall_seconds"]) or value["wall_seconds"] < 0):
            raise Blocked("Recorded session timing identity/duration differs")
        for key in ("rf_started", "rf_completed", "svm_started", "svm_completed"):
            if type(value[key]) is not int or value[key] < 0:
                raise Blocked("Session fit counter invalid")
        all_receipts[path.name] = value
    completed = [name for name, row in all_receipts.items() if row["outcome"] == "SELECTION_WORK_COMPLETE"]
    if len(completed) != 1:
        raise Blocked("Missing persisted timing receipt for completed selection")
    cutoff = all_receipts[completed[0]]
    check_files(state / "sessions", cutoff["prior_receipts"])
    if set(cutoff["prior_receipts"]) - set(all_receipts):
        raise Blocked("Frozen timing receipt refers to absent session")
    receipts = [all_receipts[name] for name in sorted(cutoff["prior_receipts"])] + [cutoff]
    if type(mi_seconds) not in (int, float) or not math.isfinite(mi_seconds) or mi_seconds < 0:
        raise Blocked("Invalid measured MI duration")
    coarse_seconds = sum(row["unit_seconds"] for row in coarse)
    fine_seconds = sum(row["unit_seconds"] for row in fine)
    return {"identity": identity["sha256"], "rf_fits": rfe["forest_fits_in_completed_rfe"],
        "svm_fits": len(history.rows), "svm_coarse_fits": len(coarse), "svm_fine_fits": len(fine),
        "fit_count_scope": "completed ranking/candidates, excludes interrupted unrecorded attempts",
        "rf_fit_attempts_recorded": sum(r["rf_started"] for r in receipts),
        "svm_fit_attempts_recorded": sum(r["svm_started"] for r in receipts),
        "mi_seconds": mi_seconds, "rf_rfe_fit_seconds": rf_seconds,
        "svm_coarse_seconds": coarse_seconds, "coarse_seconds": rf_seconds+coarse_seconds,
        "fine_seconds": fine_seconds,
        "svm_fit_seconds_sum": sum(row["fit"]["seconds"] for row in history.rows),
        "selection_completed_work_seconds": mi_seconds+rf_seconds+coarse_seconds+fine_seconds,
        "total_selection_seconds": sum(r["wall_seconds"] for r in receipts),
        "total_selection_time_scope": "sum persisted selection-session wall receipts up to freeze; excludes preflight/final publication/verify and unrecorded hard-interruption time",
        "session_receipts": len(receipts), "hard_interruption_unrecorded_time": "UNKNOWN_NOT_RECONSTRUCTED",
        "peak_rss_bytes_recorded": max(r["peak_rss_bytes"] for r in receipts),
        "benchmark_speedup": None}


def selection_payload(identity, history):
    best = winner(history.rows)
    plan = history.fine_plan()
    return {"algorithm": ALGORITHM, "identity": identity["sha256"],
            "config": identity["config"], "run_id": identity["run_id"], "rfe_step": 2,
            "search_policy": history.policy, "best_k": best["k"],
            "coarse": history.policy["coarse_grid"], "coarse_best_k": plan["coarse_best_k"],
            "fine_radius": history.policy["fine_radius"], "fine": plan["fine_grid"],
            "selected_features": history.ranked[:best["k"]],
            "validation_metrics": {key: best[key] for key in
                (*METRICS, "confusion_matrix_labels_0_benign_1_attack", "classification_report", "label_origin")},
            "evaluated_k": [row["k"] for row in history.rows],
            "selection_nonconverged_k": [row["k"] for row in history.rows if not row["fit"]["converged"]],
            "test_evaluation_calls": 0, **SCIENTIFIC}


def validate_selection(state, identity):
    state = Path(state)
    _, ranked = validate_ranking(state, identity)
    history = History(state, identity, ranked, sha(state / "rfe_state.json"))
    validate_rows(history.rows, ranked, history.policy,
                  validation_count=identity["dataset"]["splits"]["val"]["rows"], complete=True)
    if read(state / "fine_plan.json") != history.fine_plan() or read(state / "k_search_results.json") != search_payload(history):
        raise Blocked("Coarse/fine plan, evaluated grid or measured candidate scores differ")
    best = winner(history.rows)
    frozen = {"identity": identity["sha256"], "best_k": best["k"], "features": ranked[:best["k"]],
              "best_validation_accuracy": best["accuracy"], "k_search_sha256": sha(state / "k_search_results.json"),
              "test_used_for_selection": False, "selection_fit_role": "Train; score on Validation only"}
    if (read(state / "selection_frozen.json") != frozen
            or read(state / "features.json") != frozen["features"]
            or read(state / "selection.json") != selection_payload(identity, history)
            or read(state / "statistics.json") != measurements(state, identity, history)):
        raise Blocked("Frozen B selection/statistics differs from measured, policy-complete history")
    return {"status": "PASS", "best_k": best["k"], "validation_accuracy": best["accuracy"],
            "rf_fits": read(state / "statistics.json")["rf_fits"], "svm_fits": len(history.rows), **SCIENTIFIC}


def create_selection(state, identity, history, module):
    state = Path(state)
    validate_rows(history.rows, history.ranked, history.policy,
                  validation_count=identity["dataset"]["splits"]["val"]["rows"], complete=True)
    payload = search_payload(history)
    json_once(state / "k_search_results.json", payload)
    module.frozen_selection(state, identity, payload)  # Same A1 frozen-selection bytes/meaning.
    json_once(state / "selection.json", selection_payload(identity, history))
    stats = measurements(state, identity, history)
    json_once(state / "statistics.json", stats)
    report = (f"# Phương án B — selection only\n\nConfig: {identity['config']}\nRun: {identity['run_id']}\n"
              f"Dataset: CICFLOWMETER_FEATURE_CSV\nRFE step: 2\nPolicy: {history.policy['name']}\n"
              f"Coarse k: {history.policy['coarse_grid']}\nCoarse best k: {payload['coarse_best_k']}\n"
              f"Fine radius: {history.policy['fine_radius']}\nFine k: {payload['fine_grid']}\n"
              f"Best k: {payload['best_k']}\nValidation Accuracy: {payload['best_validation_accuracy']}\n"
              f"RF fits in completed ranking: {stats['rf_fits']}\nSVM candidates: {stats['svm_fits']}\n"
              f"Coarse seconds (RF + SVM units): {stats['coarse_seconds']}\nFine seconds: {stats['fine_seconds']}\n"
              f"Recorded selection-session wall seconds: {stats['total_selection_seconds']}\n\n"
              "Time excludes unrecorded hard interruptions, preflight and final publication/verification.\n"
              "Selection covers exactly the approved coarse/fine grid; no guarantee of the exhaustive k=1..100 optimum.\n"
              "RFE step2 can change ranking relative to A1 step5 and reference step1.\n"
              "EXPERIMENTAL_NON_OFFICIAL. Proxy labels remain provisional. No final refit or Test evaluation.\n"
              "No measured speedup or equality to A1 is claimed.\n")
    path = state / "selection_report.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != report:
            raise Blocked("Existing selection report differs")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(report)
    validate_selection(state, identity)


def publish(store):
    outcome = validate_selection(store.state, store.identity)
    _, ranked = validate_ranking(store.state, store.identity)
    history = History(store.state, store.identity, ranked, sha(store.state / "rfe_state.json"))
    check_committed_search(store, history)
    selected = store.saved("selection")
    if selected is None:
        raise Blocked("Selection must be COMMITTED before publication")
    root = store.remote / "results"
    root.mkdir(exist_ok=True)
    for name in OUTPUTS:
        source, target = store.state / name, root / name
        if target.exists():
            if info(source) != info(target):
                raise Blocked("Existing published result differs: " + name)
        else:
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*1024**2)
            if info(source) != info(target):
                raise Blocked("Publication readback SHA differs")
    manifest = {"identity": store.identity["sha256"], "algorithm": ALGORITHM,
                "files": {name: info(root / name) for name in OUTPUTS}}
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.exists():
        if read(manifest_path) != manifest:
            raise Blocked("Published manifest differs")
    else:
        drive_json_once(manifest_path, manifest)
    marker = {"format": FORMAT, "identity": store.identity["sha256"], "status": "SELECTION_COMPLETE",
              "checkpoint": selected["sha256"], "manifest": info(manifest_path),
              "test_evaluation_calls": 0, **SCIENTIFIC}
    marker_path = root / "SELECTION_COMPLETE.json"
    if marker_path.exists():
        if read(marker_path) != marker:
            raise Blocked("Completed marker differs")
    else:
        drive_json_once(marker_path, marker)
    return {**outcome, "directory": str(store.remote), "phase_status": "SELECTION_COMPLETE",
            "test_evaluation_calls": 0}
