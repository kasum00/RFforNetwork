"""Summarize completed result.json files without importing models or reading Test CSVs."""
from __future__ import annotations

import json
from pathlib import Path


def previous_artifacts_status(root: Path) -> dict:
    """The two Stage 5 FS16 results are not fabricated if absent from Drive."""
    expected = {"FW_512_FS_16": 8, "FW_256_FS_16": 18}
    found = {}
    for config, best_k in expected.items():
        candidates = []
        for path in (root / "experiment_results").glob("*/result.json"):
            if not all((path.parent / name).is_file() for name in
                       ("model.joblib", "final_scaler.joblib", "features.json")):
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (result.get("experiment") == "mi_rfe_rf" and result.get("config") == config
                    and result.get("selection_measurements", {}).get("best_k") == best_k
                    and result.get("status") in ("PASS", "COMPLETED_WITH_CONVERGENCE_WARNING")):
                candidates.append(str(path))
        found[config] = {"status": "FOUND" if candidates else "PREVIOUS_ARTIFACTS_MISSING",
                         "result_json": candidates}
    return found


def completed_results(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "experiment_results").glob("*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if result.get("status") not in ("PASS", "COMPLETED_WITH_CONVERGENCE_WARNING"):
            continue
        m = result.get("metrics", {})
        report = m.get("classification_report", {})
        rows.append({"run": path.parent.name, "experiment": result.get("experiment"),
                     "config": result.get("config"), "status": result.get("status"),
                     "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
                     "balanced_accuracy": m.get("balanced_accuracy"),
                     "benign_precision": report.get("Benign_proxy", {}).get("precision"),
                     "benign_recall": report.get("Benign_proxy", {}).get("recall"),
                     "benign_f1": report.get("Benign_proxy", {}).get("f1-score"),
                     "attack_precision": report.get("Attack_proxy", {}).get("precision"),
                     "attack_recall": report.get("Attack_proxy", {}).get("recall"),
                     "attack_f1": report.get("Attack_proxy", {}).get("f1-score"),
                     "selected_features": len(result.get("features", [])),
                     "elapsed_seconds": result.get("elapsed_seconds"),
                     "peak_rss_bytes_sampled": result.get("peak_rss_bytes_sampled"),
                     "confusion_matrix_0_benign_1_attack": m.get("confusion_matrix_labels_0_benign_1_attack"),
                     "label_origin": result.get("label_origin"),
                     "result_json": str(path)})
    return rows
