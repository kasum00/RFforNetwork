import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from phuong_anA.contract import ALGORITHM, FORMAT, atomic, digest, info, sha
from phuong_anA.dataset import stable

SOURCE = Path(__file__).resolve().parents[2]


def small_identity(**changes):
    value = {"format": FORMAT, "algorithm": ALGORITHM, "config": "FW256_FS8", "run_id": "RUN_A1_UNIT",
             "source": {"sha256": "s"*64}, "git_commit": "a"*40, "dataset_fingerprint": "d"*64,
             "contract": {"rfe": {"step": 5, "n_features_to_select": 1}, "svm": {"C": 10.0}}}
    value.update(changes)
    value["sha256"] = digest(value)
    return value


def csv_file(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def dataset_fixture(root):
    dataset = Path(root) / "CICFLOWMETER_FEATURE_CSV"
    folder = dataset / "FW256_FS8"
    folder.mkdir(parents=True)
    names = [f"feature_{i}__mean" for i in range(380)]
    sizes = {"train": 8, "val": 4, "test": 4}
    for role in ("train", "val"):
        n = sizes[role]
        frame = {name: pa.array(np.arange(n, dtype=np.float32)+i) for i, name in enumerate(names)}
        frame["Window_Label"] = pa.array(np.array([0, 1]*(n//2), dtype=np.int8))
        pq.write_table(pa.table(frame), folder / (role + "_windows.parquet"))
    # Intentionally NOT a Parquet file: metadata inspection must never open Test.
    (folder / "test_windows.parquet").write_bytes(b"TEST_MUST_NOT_BE_OPENED")
    flows = dict.fromkeys(sizes, 1000)
    segments = [{"segment_id": "S_"+role, "split": role, "source_id": str(i+1),
                 "first_sorted_rank": "0", "last_sorted_rank": "999", "selected_flows": "1000"}
                for i, role in enumerate(sizes)]
    evidence_hash = lambda tag: hashlib.sha256(tag.encode()).hexdigest()
    proof = {k: {"sha256": evidence_hash(k)} for k in ("validation", "v2_preflight", "manifest", "v2_binding", "label_producer")}
    producer = 'proxy = 0 if source["category"] == "BENIGN" else 1'
    proof["label_producer"]["sha256"] = hashlib.sha256(producer.encode()).hexdigest()
    parent = {"validation": {"status": "PASS_GUARDED_EXPORT", "role_flows": flows,
                "exported_flows": 3000, "provenance_rows": 3000, "output_file_count": 3,
                "v2_preflight_sha256": proof["v2_preflight"]["sha256"],
                "v2_manifest_sha256": proof["manifest"]["sha256"], "provenance_sha256": evidence_hash("provenance")},
        "v2_preflight": {"ready_for_materialize": True, "v2_binding_sha256": proof["v2_binding"]["sha256"],
            "label_policy": "RAW_NO_LABEL; BINARY_PROXY_FOLDER_DERIVED_PROVISIONAL",
            "checks": {key: "PASS" for key in ("frozen_stage28_membership", "no_embargo_or_role_migration",
                "source_row_identity_unique", "exact_full_record_cross_split_duplicate")}},
        "manifest": segments, "v2_binding": {"v2_code_sha256": proof["label_producer"]["sha256"]},
        "label_producer": producer}
    schema = {"status": "PASS", "aggregate_features": names, "aggregate_feature_count": 380,
              "model_table_columns": names+["Window_Label"], "output_label": "Window_Label",
              "binary_label_status": "PROVISIONAL_FOLDER"}
    atomic(folder / "feature_schema.json", schema)
    atomic(folder / "window_feature_schema.json", schema)
    atomic(folder / "temporal_gap_audit.json", {"policy": "TRAIN_ONLY"})
    csv_file(folder / "window_counts_by_segment.csv", [{"segment_id": "S_"+role,
        "split": role, "source_id": str(i+1), "input_rows": "1000", "expected_windows": str(sizes[role])}
        for i, role in enumerate(sizes)])
    binding = {"config": {"fw": 256, "fs": 8}, "input_sha256": {**{k: proof[k]["sha256"] for k in
        ("validation", "v2_preflight", "manifest", "v2_binding")}, "provenance": evidence_hash("provenance")}}
    binding["binding_sha256"] = stable(binding)
    pre = {"status": "PASS", "binding": binding, "binding_sha256": binding["binding_sha256"],
           "feature_schema_sha256": sha(folder / "feature_schema.json"),
           "gap_audit_sha256": sha(folder / "temporal_gap_audit.json"),
           "counts_sha256": sha(folder / "window_counts_by_segment.csv"),
           "mandatory_gates": {k: "PASS" for k in ("v2_export", "source_csv_hashes", "chronological_row_identity",
               "frozen_bucket_role_and_embargo", "schema_and_numeric_conversion", "train_only_gap_fit")}}
    atomic(folder / ".stage3_preflight.json", pre)
    checks = {"v2_flow_totals_and_manifest": "PASS", "source_csv_and_index_sha256": "PASS",
              "frozen_split_embargo_membership": "PASS_PRECHECKED_EVERY_EXPORTED_ROW",
              "rank_timestamp_identity": "PASS_PRECHECKED_EVERY_EXPORTED_ROW", "window_boundaries_and_counts": "PASS",
              "window_any_attack_label": "PASS", "aggregate_schema_and_alignment": "PASS", "segment_checkpoint_hashes": "PASS"}
    integrity = {"status": "PASS", "configuration": {"fw": 256, "fs": 8}, "binding_sha256": binding["binding_sha256"],
        "input_flows": flows, "windows_by_split": sizes,
        "window_labels_by_split": {role: {"0": n//2, "1": n//2} for role, n in sizes.items()},
        "output_sha256": {role: {"model_sha256": sha(folder / (role+"_windows.parquet")) if role != "test" else evidence_hash("unopened_test")}
                          for role in sizes}, "checks": checks}
    atomic(folder / "window_integrity.json", integrity)
    atomic(folder / ".stage3_complete.json", {"status": "PASS", "binding_sha256": binding["binding_sha256"],
        "integrity_sha256": sha(folder / "window_integrity.json"), "total_windows": sum(sizes.values())})
    return {"source": str(SOURCE), "dataset": str(dataset), "config": "FW256_FS8"}, parent, proof, folder
