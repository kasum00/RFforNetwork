"""Read-only dataset and local numpy caches, never part of a Drive bundle."""
from pathlib import Path
import os
import uuid
from .common import Blocked, FORMAT, atomic, check_files, core, digest, info, paths, read


def inspect(request):
    root = Path(request["dataset"])
    audit = core(request["source"]).metadata_audit(request["source"], root, request["config"])
    sidecars = {}
    for folder in (root, root / request["config"]):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix == ".json" and any(s in p.name.lower() for s in ("schema", "integrity", "manifest", "label")):
                if p.stat().st_size > 8*1024**2:
                    raise Blocked("Oversized metadata sidecar")
                sidecars[p.relative_to(root).as_posix()] = info(p)
    folder = root / request["config"]
    integrity, schema = read(folder / "window_integrity.json"), read(folder / "window_feature_schema.json")
    expected = core(request["source"]).EXPECTED[request["config"]]
    if (integrity.get("status") != "PASS" or integrity.get("windows_by_split") != expected
            or schema.get("aggregate_feature_count") != 380 or schema.get("output_label") != "Window_Label"):
        raise Blocked("Dataset integrity/schema sidecars differ")
    labels = integrity.get("window_labels_by_split", {})
    for role, n in expected.items():
        counts = labels.get(role, {})
        if (set(counts) != {"0", "1"} or any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts.values()) != n):
            raise Blocked("Invalid bound label counts: " + role)
    audit.update(manifest_evidence=sidecars, label_histograms=labels,
                 fingerprint=digest({"binding": audit["binding"], "sidecars": sidecars}),
                 full_content_hash=False, label_provenance=schema.get("binary_label_status", "UNKNOWN"))
    return audit


def cache_arrays(directory, binding, build):
    """Checksummed cache, loaded once per attempt; build returns arrays + metadata."""
    import numpy as np
    directory = Path(directory)
    if directory.exists():
        record = read(directory / "cache.json")
        if record.get("binding") != binding:
            raise Blocked("Local cache binding changed; do not overwrite")
        check_files(directory, record["files"])
    else:
        stage = directory.with_name(directory.name + ".partial-" + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        arrays, meta = build()
        for name, value in arrays.items():
            if not name.isidentifier():
                raise Blocked("Invalid cache name")
            np.save(stage / (name + ".npy"), value, allow_pickle=False)
        record = {"binding": binding, "metadata": meta,
                  "files": {p.name: info(p) for p in stage.glob("*.npy")}}
        atomic(stage / "cache.json", record, exclusive=True)
        os.rename(stage, directory)
    arrays = {Path(name).stem: np.load(directory / name, mmap_mode="r", allow_pickle=False)
              for name in record["files"]}
    return arrays, record


def raw_split(request, audit, local, identity, role, *, test_claim=None):
    if role not in ("train", "val", "test") or (role == "test" and test_claim is None):
        raise Blocked("Test is inaccessible until a verified durable claim after final refit")
    if role == "test":
        if (request.get("mode") != "finalize" or request.get("phase", "final_refit") not in ("test", "all_final")
                or test_claim.get("format") != FORMAT or test_claim.get("identity") != identity["sha256"]
                or test_claim.get("evaluation_number") != 1):
            raise Blocked("Invalid Test phase/claim identity")
        local_claim = read(Path(local) / "state/TEST_CLAIM.json")
        remote_claim = read(paths(request)[1] / "TEST_CLAIM.json")
        if (local_claim != test_claim or remote_claim != test_claim
                or test_claim.get("final_state") != digest(read(Path(local) / "state/final_state.json"))):
            raise Blocked("Test requires matching durable/local intent after completed final refit")
    def build():
        X, y, cleaning = core(request["source"]).read_values(audit, role,
            selection=role != "test", test_authorized=role == "test")
        import numpy as np
        values, counts = np.unique(y, return_counts=True)
        if {str(int(v)): int(c) for v, c in zip(values, counts)} != audit["label_histograms"][role]:
            raise Blocked("Actual labels differ from bound histogram")
        return {"X": X, "y": y}, {"cleaning": cleaning}
    arrays, record = cache_arrays(Path(local) / "cache" / role,
        {"identity": identity["sha256"], "role": role, "split": audit["binding"]["splits"][role]}, build)
    if arrays["X"].shape != (audit["binding"]["splits"][role]["rows"], 380):
        raise Blocked("Raw cache shape differs")
    return arrays["X"], arrays["y"]


def svc_prefix(ranked, k):
    import numpy as np
    if not 1 <= k <= 100 or ranked.dtype != np.float32:
        raise Blocked("Expected reference-scaled float32 Top100")
    # Matches libsvm's own input validation conversion. Values are not re-scaled.
    return np.ascontiguousarray(ranked[:, :k], dtype=np.float64)
