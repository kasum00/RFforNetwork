"""Test-only tables; missing or unverified metrics remain blank, never zero-filled."""
import csv
import io
from pathlib import Path
from .common import Blocked, CONFIGS, METHODS, SCIENTIFIC, info, intact, once, read, utc
from .pipeline import verify_test

COLUMNS = ("Config", "Classifier", "Feature set", "Method", "RFE step", "k", "Acc (%)",
           "F1 attack (%)", "Macro-F1 (%)", "Precision attack (%)", "Recall attack (%)", "Balanced Acc (%)",
           "Verification status", "Run ID")


def collect(output):
    rows, records = [], []
    for config in CONFIGS:
        for method in METHODS:
            root = Path(output) / "stage9_svm_final" / method / config
            row = dict.fromkeys(COLUMNS, "")
            row.update(Config=config, Classifier="SVM", Method=method,
                       **{"Feature set": "MI-RFE " + method, "RFE step": METHODS[method][2], "Verification status": "PENDING"})
            evidence = dict(method=method, config=config, directory=str(root), status="PENDING", test_metrics=None)
            if root.exists():
                try:
                    identity = intact(read(root / "identity_manifest.json"))
                    if identity["method"] != method or identity["config"] != config:
                        raise Blocked("Report identity pair differs")
                    row["k"], row["Run ID"] = identity["best_k"], identity["run_id"]
                    outcome = verify_test(root, identity)
                    result = outcome["result"]
                    m = result["metrics"]
                    row.update({"Acc (%)": m["accuracy"]*100, "F1 attack (%)": m["f1_attack"]*100,
                                "Macro-F1 (%)": m["macro_f1"]*100, "Precision attack (%)": m["precision_attack"]*100,
                                "Recall attack (%)": m["recall_attack"]*100, "Balanced Acc (%)": m["balanced_accuracy"]*100,
                                "Verification status": "VERIFIED_TEST"})
                    evidence.update(status="VERIFIED_TEST", test_metrics=result,
                                    test_checkpoint=info(root / "test_checkpoint.json"))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["Verification status"] = "BLOCKED"
                    evidence.update(status="BLOCKED", reason=str(exc))
            rows.append(row)
            records.append(evidence)
    return rows, records


def export(output):
    from datetime import datetime, timezone
    import uuid
    root = Path(output) / "stage9_svm_final/reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    rows, records = collect(output)
    for name, subset in (("SVM_A1_TEST_RESULTS.csv", [r for r in rows if r["Method"] == "A1"]),
                         ("SVM_B_TEST_RESULTS.csv", [r for r in rows if r["Method"] == "B"]),
                         ("SVM_A1_B_COMPARISON.csv", rows), ("SVM_A1_B_FINAL_RESULTS.csv", rows)):
        with (root / name).open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(subset)
    once(root / "SVM_A1_B_FINAL_RESULTS.json", dict(created=utc(), rows=rows, evidence=records, **SCIENTIFIC))
    text = ["# Stage9 Test verification", "", "Only VERIFIED_TEST cells contain verified Test metrics.",
            "Raw metrics (0–1) and all requested timings/hashes/features are in the JSON evidence.",
            "A1: RF-RFE step5/exhaustive Validation k; B: step2/coarse6/fine ±5. Rankings may differ.",
            "No method or k is chosen using this Test comparison. Proxy labels remain provisional.", "",
            "| Config | Method | k | Test Acc (%) | F1 attack (%) | Macro-F1 (%) | Status |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        text.append("| " + " | ".join(str(r[c]) for c in ("Config", "Method", "k", "Acc (%)", "F1 attack (%)", "Macro-F1 (%)", "Verification status")) + " |")
    for name, body in (("STAGE9_VERIFICATION_REPORT.md", "\n".join(text)),
                       ("STAGE9_EXECUTION_SUMMARY.md", "# Stage9 execution summary\n\n" +
                        "\n".join(f"{r['method']} {r['config']}: {r['status']} {r.get('reason', '')}" for r in records))):
        with (root / name).open("x", encoding="utf-8") as f:
            f.write(body + "\n")
    return dict(status="EXPORTED", directory=str(root), rows=rows, metrics_inferred=False)
