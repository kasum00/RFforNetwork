"""Read-only CICFLOW Parquet adapter. Test requires a durable protocol claim."""
from pathlib import Path
from .common import Blocked, EXPECTED, digest, hash_file, load_core, read

def inspect(request):
    core = load_core(request["source"])
    audit = core.metadata_audit(request["source"], request["dataset"], request["config"])
    if {role: row["rows"] for role, row in audit["binding"]["splits"].items()} != EXPECTED[request["config"]]:
        raise Blocked("Unexpected split counts")
    # Bind small manifests/schema/label sidecars; never load Parquet values.
    evidence = {}
    root = Path(request["dataset"])
    for folder in (root, root / request["config"]):
        for path in sorted(folder.iterdir()):
            if (path.is_file() and path.suffix.lower() in {".json", ".csv"}
                    and any(token in path.name.lower() for token in ("manifest", "schema", "integrity", "label"))):
                if path.stat().st_size > 8*1024**2:
                    raise Blocked("Manifest too large for metadata-only inspection: " + str(path))
                evidence[path.relative_to(root).as_posix()] = {"sha256": hash_file(path), "bytes": path.stat().st_size}
    folder = root / request["config"]
    integrity_path, schema_path = folder / "window_integrity.json", folder / "window_feature_schema.json"
    if not integrity_path.is_file() or not schema_path.is_file():
        raise Blocked("Upload the existing window_integrity.json and window_feature_schema.json sidecars for this config; do not generate new splits")
    integrity, schema = read(integrity_path), read(schema_path)
    if (integrity.get("status") != "PASS" or integrity.get("windows_by_split") != EXPECTED[request["config"]]
            or schema.get("aggregate_feature_count") != 380 or schema.get("output_label") != "Window_Label"):
        raise Blocked("Split/schema integrity sidecar does not match current dataset")
    labels = integrity.get("window_labels_by_split", {})
    for role, rows in EXPECTED[request["config"]].items():
        histogram = labels.get(role, {})
        if set(histogram) != {"0", "1"} or any(not isinstance(v, int) or v < 0 for v in histogram.values()) or sum(histogram.values()) != rows:
            raise Blocked("Invalid bound label histogram for " + role)
    audit["label_histograms"] = labels
    audit["binary_label_status"] = schema.get("binary_label_status", "UNKNOWN")
    audit["manifest_evidence"] = evidence
    audit["fingerprint"] = digest({"binding": audit["binding"], "manifest_evidence": evidence})
    audit["label_mapping"] = {"0": "Benign_proxy", "1": "Attack_proxy"}
    audit["label_mapping_provenance"] = "UNKNOWN: bound sidecars available for review; binary values checked when the role is opened"
    audit["split_manifest_status"] = "BOUND_FOR_REVIEW" if evidence else "NOT_PRESENT_OFFICIAL_BLOCKED"
    return audit

def load(request, audit, role, *, protocol=None):
    if role == "test":
        if protocol is None:
            raise Blocked("Test cannot be opened during selection/validation/final refit")
        from .checkpoint_manager import TEST_EVALUATION_REQUESTED, _read_strict_json
        state = _read_strict_json(protocol.path)
        if state.get("test_state") != TEST_EVALUATION_REQUESTED or state.get("final_refit_status") != "STAGE_COMPLETE":
            raise Blocked("Missing durable one-time Test claim after final refit")
    matrix, labels, cleaning = load_core(request["source"]).read_values(audit, role,
        selection=role != "test", test_authorized=role == "test")
    import numpy as np
    values, counts = np.unique(labels, return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts)}
    if histogram != audit["label_histograms"][role]:
        raise Blocked("Actual labels do not match integrity sidecar for " + role)
    return matrix, labels, cleaning
