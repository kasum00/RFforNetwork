"""Manifest-driven CICFLOW adapter. Selection never opens Test, even its footer."""
import base64
import csv
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from .contract import Blocked, core, digest, info, read, sha
from mi_rfe_svm_colab.dataset import cache_arrays

SIDECARS = (".stage3_preflight.json", ".stage3_complete.json", "feature_schema.json",
            "window_feature_schema.json", "window_integrity.json", "temporal_gap_audit.json",
            "window_counts_by_segment.csv")


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def discover(dataset):
    """Only Stage3-completed folder names/config manifests, no inherited row counts."""
    found, rejected = [], {}
    for folder in sorted(Path(dataset).iterdir()):
        match = re.fullmatch(r"FW([1-9][0-9]*)_FS([1-9][0-9]*)", folder.name)
        if not folder.is_dir() or not match:
            continue
        try:
            manifest, complete = read(folder / "window_integrity.json"), read(folder / ".stage3_complete.json")
            if (manifest.get("status") != "PASS" or complete.get("status") != "PASS"
                    or manifest.get("configuration") != dict(fw=int(match[1]), fs=int(match[2]))):
                raise Blocked("Not a completed matching FW/FS manifest")
            found.append(folder.name)
        except (OSError, ValueError, TypeError) as exc:
            rejected[folder.name] = str(exc)
    return {"configs": found, "rejected": rejected}


def evidence(request, pre):
    """Original small Stage29 manifests, re-bound by each config's Stage3 hashes."""
    root = request.get("evidence_root")
    bundled = read(Path(__file__).parent / "evidence/cicflow_split_evidence.json") if not root else None
    values, records = {}, {}
    for key in ("validation", "v2_preflight", "manifest", "v2_binding", "label_producer"):
        expected = values["v2_binding"]["v2_code_sha256"] if key == "label_producer" else pre["binding"]["input_sha256"][key]
        relative = "code_CICFLOW/stage29_v2_allocation.py" if key == "label_producer" else pre["binding"]["input_paths"][key].replace("\\", "/")
        if root:
            path = Path(root) / Path(relative).name
            if not path.is_file() or path.stat().st_size > 8*1024**2:
                raise Blocked("Missing/oversized frozen split evidence: " + str(path))
            raw = path.read_bytes()
        else:
            entry = bundled["files"].get(key)
            if not entry or entry["sha256"] != expected or entry["source_relative_path"] != relative:
                raise Blocked("Bundled evidence belongs to another split/dataset; supply its original evidence_root")
            raw = gzip.decompress(base64.b64decode(entry["gzip_base64"], validate=True))
            if len(raw) != entry["bytes"]:
                raise Blocked("Evidence snapshot byte count differs")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise Blocked("Frozen split evidence SHA256 mismatch: " + key)
        text = raw.decode("utf-8-sig")
        values[key] = text if key == "label_producer" else (list(csv.DictReader(io.StringIO(text))) if key == "manifest" else json.loads(text))
        records[key] = {"sha256": expected, "source_relative_path": relative, "bytes": len(raw)}
    return values, records


def validate_manifest(folder, source, parent, records):
    pre, complete, integrity, schema = (read(folder / name) for name in
        (".stage3_preflight.json", ".stage3_complete.json", "window_integrity.json", "window_feature_schema.json"))
    binding = pre["binding"]
    bound = {k: v for k, v in binding.items() if k != "binding_sha256"}
    if (stable(bound) != binding.get("binding_sha256") or pre.get("binding_sha256") != binding["binding_sha256"]
            or complete.get("binding_sha256") != binding["binding_sha256"]
            or integrity.get("binding_sha256") != binding["binding_sha256"]):
        raise Blocked("Stage3 binding chain differs")
    expected_config = {k: binding["config"][k] for k in ("fw", "fs")}
    if folder.name != f"FW{expected_config['fw']}_FS{expected_config['fs']}" or integrity["configuration"] != expected_config:
        raise Blocked("FW/FS folder/manifest identity mismatch")
    if any(x.get("status") != "PASS" for x in (pre, complete, integrity, schema)):
        raise Blocked("Stage3 completion/schema/integrity must PASS")
    for name, expected in (("window_integrity.json", complete["integrity_sha256"]),
                           ("feature_schema.json", pre["feature_schema_sha256"]),
                           ("window_feature_schema.json", pre["feature_schema_sha256"]),
                           ("temporal_gap_audit.json", pre["gap_audit_sha256"]),
                           ("window_counts_by_segment.csv", pre["counts_sha256"])):
        if sha(folder / name) != expected:
            raise Blocked("Stage3 manifest/sidecar changed: " + name)
    required = ("v2_export", "source_csv_hashes", "chronological_row_identity",
                "frozen_bucket_role_and_embargo", "schema_and_numeric_conversion", "train_only_gap_fit")
    if any(pre.get("mandatory_gates", {}).get(g) != "PASS" for g in required):
        raise Blocked("Missing Stage3 split/schema admission evidence")
    checks = {"v2_flow_totals_and_manifest": "PASS", "source_csv_and_index_sha256": "PASS",
              "frozen_split_embargo_membership": "PASS_PRECHECKED_EVERY_EXPORTED_ROW",
              "rank_timestamp_identity": "PASS_PRECHECKED_EVERY_EXPORTED_ROW",
              "window_boundaries_and_counts": "PASS", "window_any_attack_label": "PASS",
              "aggregate_schema_and_alignment": "PASS", "segment_checkpoint_hashes": "PASS"}
    if any(integrity.get("checks", {}).get(k) != v for k, v in checks.items()):
        raise Blocked("Required frozen split/window checks missing or failed")
    validation, prior, segments = parent["validation"], parent["v2_preflight"], parent["manifest"]
    if (validation.get("status") != "PASS_GUARDED_EXPORT" or prior.get("ready_for_materialize") is not True
            or validation["v2_preflight_sha256"] != records["v2_preflight"]["sha256"]
            or validation["v2_manifest_sha256"] != records["manifest"]["sha256"]
            or validation["provenance_sha256"] != binding["input_sha256"]["provenance"]):
        raise Blocked("Parent split provenance hashes/status differ")
    for key in ("frozen_stage28_membership", "no_embargo_or_role_migration",
                "source_row_identity_unique", "exact_full_record_cross_split_duplicate"):
        if not prior.get("checks", {}).get(key, "").startswith("PASS"):
            raise Blocked("Parent Train/Validation/Test isolation check failed: " + key)
    if prior["v2_binding_sha256"] != records["v2_binding"]["sha256"]:
        raise Blocked("Parent V2 binding differs")
    if (records["label_producer"]["sha256"] != parent["v2_binding"]["v2_code_sha256"]
            or prior.get("label_policy") != "RAW_NO_LABEL; BINARY_PROXY_FOLDER_DERIVED_PROVISIONAL"
            or schema.get("binary_label_status") != "PROVISIONAL_FOLDER"
            or 'proxy = 0 if source["category"] == "BENIGN" else 1' not in parent["label_producer"]):
        raise Blocked("Missing/changed provenance of the 0/1 folder-derived proxy labels")
    by_id, flows, ranges = {}, dict.fromkeys(("train", "val", "test"), 0), defaultdict(list)
    for row in segments:
        sid, role = row["segment_id"], row["split"]
        if sid in by_id or role not in flows:
            raise Blocked("Duplicate segment or unknown role in split manifest")
        by_id[sid] = row
        flows[role] += int(row["selected_flows"])
        ranges[row["source_id"]].append((int(row["first_sorted_rank"]), int(row["last_sorted_rank"]), role))
    for intervals in ranges.values():
        for i, (lo, hi, role) in enumerate(intervals):
            if lo < 0 or hi < lo:
                raise Blocked("Invalid frozen source-rank range")
            for other_lo, other_hi, other_role in intervals[i+1:]:
                if role != other_role and max(lo, other_lo) <= min(hi, other_hi):
                    raise Blocked("Cross-split source rank ranges overlap")
    if (flows != validation["role_flows"] or flows != integrity["input_flows"]
            or sum(flows.values()) != validation["exported_flows"]
            or validation["provenance_rows"] != validation["exported_flows"]
            or len(segments) != validation["output_file_count"]):
        raise Blocked("Parent split totals/segments differ")
    with (folder / "window_counts_by_segment.csv").open(encoding="utf-8-sig", newline="") as f:
        counts = list(csv.DictReader(f))
    totals, seen = dict.fromkeys(flows, 0), set()
    for row in counts:
        sid = row["segment_id"]
        if sid in seen or sid not in by_id or row["split"] != by_id[sid]["split"] or row["source_id"] != by_id[sid]["source_id"]:
            raise Blocked("Window segment/split source mapping changed")
        seen.add(sid)
        if int(row["input_rows"]) != int(by_id[sid]["selected_flows"]):
            raise Blocked("Window segment input count changed")
        totals[row["split"]] += int(row["expected_windows"])
    if set(by_id) != seen or totals != integrity["windows_by_split"] or sum(totals.values()) != complete["total_windows"]:
        raise Blocked("Window counts are not bound to the frozen split manifest")
    return integrity, schema, {"0": "Benign_folder_proxy", "1": "Attack_folder_proxy",
        "status": "PROVISIONAL_FOLDER", "ground_truth": "NOT_VERIFIED", "producer_sha256": records["label_producer"]["sha256"]}


def inspect(request):
    """Train/Val bytes and footers only. Test is stat + archived manifest SHA only."""
    root, source = Path(request["dataset"]).resolve(), Path(request["source"]).resolve()
    available = discover(root)
    if request["config"] not in available["configs"]:
        raise Blocked("Invalid/missing config manifest: " + str(available))
    folder = root / request["config"]
    for name in SIDECARS:
        if not (folder / name).is_file() or (folder / name).stat().st_size > 8*1024**2:
            raise Blocked("Upload existing small CICFLOW sidecar (do not regenerate): " + str(folder / name))
    pre = read(folder / ".stage3_preflight.json")
    parent, proof = evidence(request, pre)
    integrity, schema, labels = validate_manifest(folder, source, parent, proof)
    module = core(source)
    from stage4_mi.mi_selection import load_feature_schema, validate_feature_names, PROVENANCE_TOKENS
    names = validate_feature_names(load_feature_schema(folder))
    if (len(names) != 380 or schema.get("aggregate_feature_count") != 380
            or schema.get("output_label") != "Window_Label" or names != schema["aggregate_features"]
            or schema["model_table_columns"] != names + ["Window_Label"]
            or any(any(t in n.lower() for t in (*PROVENANCE_TOKENS, "label", "target")) for n in names)):
        raise Blocked("Feature order/schema contains label/provenance or is not the 380-feature contract")
    import pyarrow as pa
    import pyarrow.parquet as pq
    splits, identities = {}, []
    for role in ("train", "val", "test"):
        path = module.resolve_split(folder, role)
        st = path.stat()  # No Test open/hash/read_table call.
        identity = (st.st_dev, st.st_ino)
        if identity[1] and identity in identities:
            raise Blocked("Split files alias the same filesystem object")
        identities.append(identity)
        n = integrity["windows_by_split"][role]
        histogram = integrity["window_labels_by_split"][role]
        if (type(n) is not int or n <= 0 or set(histogram) != {"0", "1"}
                or any(type(v) is not int or v <= 0 for v in histogram.values()) or sum(histogram.values()) != n):
            raise Blocked("Invalid manifest row counts/binary label histogram")
        row = {"path": str(path), "rows": n, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
               "declared_sha256": integrity["output_sha256"][role]["model_sha256"]}
        if role != "test":
            if sha(path) != row["declared_sha256"]:
                raise Blocked("Train/Validation Parquet SHA256 differs: " + role)
            parquet = pq.ParquetFile(path)
            if (parquet.schema_arrow.names != names + ["Window_Label"] or parquet.metadata.num_rows != n
                    or not all(pa.types.is_integer(f.type) or pa.types.is_floating(f.type) for f in parquet.schema_arrow)):
                raise Blocked("Train/Validation Parquet schema/order/count mismatch")
            row.update(schema_sha256=digest(str(parquet.schema_arrow)), sha256_verified=True)
        else:
            row["sha256_verified"] = False
        splits[role] = row
    if len({s["declared_sha256"] for s in splits.values()}) != 3:
        raise Blocked("Split manifests declare identical model files")
    result = {"binding": {"config": request["config"], "names": names, "splits": splits},
              "sidecars": {name: info(folder / name) for name in SIDECARS}, "split_evidence": proof,
              "label_histograms": integrity["window_labels_by_split"], "label_mapping": labels,
              "isolation_scope": "Frozen source-row membership/embargo/exact-duplicate attestations and nonoverlapping rank ranges",
              "limitations": {k: v for k, v in integrity["checks"].items() if not v.startswith("PASS")},
              "test_access": "STAT_ONLY_NO_OPEN_OR_HASH", "test_values_read": False}
    result["fingerprint"] = digest(result)
    return result


def load_selection(request, audit, local, identity, role):
    if role not in ("train", "val"):
        raise Blocked("A1 is selection-only: opening Test is forbidden")
    def build():
        import numpy as np
        X, y, cleaning = core(request["source"]).read_values(audit, role, selection=True)
        values, counts = np.unique(y, return_counts=True)
        if {str(int(v)): int(c) for v, c in zip(values, counts)} != audit["label_histograms"][role]:
            raise Blocked("Actual Train/Validation proxy labels differ from manifest")
        return {"X": X, "y": y}, {"cleaning": cleaning}
    arrays, record = cache_arrays(Path(local) / "cache" / role,
        {"algorithm": "MI_RFE_SVM_A1_STEP5_TARGET1", "identity": identity["sha256"],
         "split": audit["binding"]["splits"][role]}, build)
    if arrays["X"].shape != (audit["binding"]["splits"][role]["rows"], 380) or len(arrays["y"]) != len(arrays["X"]):
        raise Blocked("Local array cache shape differs")
    return arrays["X"], arrays["y"]
