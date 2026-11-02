"""The same reference cleaning and row order; Test bytes require a verified claim."""
from pathlib import Path
from .common import Blocked, files_checked, read, sha


def load(request, identity, role, *, claim=None):
    if role not in ("train", "val", "test"):
        raise Blocked("Unknown split")
    if role == "test":
        from .common import paths
        _, remote = paths(request)
        if claim is None or not (remote / "TEST_CLAIM.json").is_file() or read(remote / "TEST_CLAIM.json") != claim:
            raise Blocked("No verified durable claim: Test must remain unopened")
        if claim["identity"] != identity["sha256"]:
            raise Blocked("Test claim identity differs")
        from .common import confirmation
        if not request.get("enable_test_evaluation") or request.get("test_confirmation") != confirmation(request, "TEST_ONCE"):
            raise Blocked("Test loader requires explicit user authorization")
        from .pipeline import validate_claim, verify_final
        verify_final(remote, identity)
        validate_claim(claim, remote, identity)
    audit = identity["dataset"]
    split = audit["binding"]["splits"][role]
    if sha(split["path"]) != split["declared_sha256"]:
        raise Blocked("Dataset file changed: " + role)
    if role == "test":
        # First footer open is AFTER the durable claim and full Test SHA check.
        import pyarrow.parquet as pq
        if pq.ParquetFile(split["path"]).schema_arrow.names != audit["binding"]["names"] + ["Window_Label"]:
            raise Blocked("Test schema/order mismatch")
    from mi_rfe_svm_colab.common import core
    X, y, cleaning = core(request["source"]).read_values(audit, role,
        selection=role != "test", names=identity["features"], test_authorized=role == "test")
    import numpy as np
    values, counts = np.unique(y, return_counts=True)
    if ({str(int(v)): int(n) for v, n in zip(values, counts)} != audit["label_histograms"][role]
            or X.shape != (split["rows"], identity["best_k"])):
        raise Blocked("Actual split labels/counts/features differ from verified manifest")
    return X, y
