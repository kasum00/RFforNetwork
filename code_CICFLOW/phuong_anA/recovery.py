"""Read-only assessment of existing A1 artifacts. No Store/cache/lease creation.

This module never loads a model or opens the dataset. Its PASS checks saved
artifacts, not compatibility with new source or current dataset bytes. Strict
preflight remains mandatory before any resume. No migration is performed.
"""
import csv
import math
from pathlib import Path
from .contract import ALGORITHM, FORMAT, Blocked, check_files, digest, info, read, sha
from .rfe import validate_result
from .store import ledger
from .telemetry import durable_completed_k
from mi_rfe_svm_colab.local_progress import choose, validate_rows


def records(paths, identity, ranked, rfe_sha, validation_rows):
    """Validate complete records in memory; unlike KProgress, never mkdir."""
    rows, previous = [], None
    for k, path in enumerate(paths, 1):
        value = read(path)
        if (Path(path).name != f"k_{k:03d}.json"
                or value["binding"] != {"identity": identity["sha256"], "rfe": rfe_sha, "ranked": digest(ranked)}
                or value["previous"] != previous
                or value["sha256"] != digest({key: v for key, v in value.items() if key != "sha256"})):
            raise Blocked("K record identity/order/hash chain differs")
        row = value["row"]
        validate_rows(rows + [row], ranked)
        for key in ("precision_attack", "recall_attack", "recall_benign", "f1_attack", "macro_f1", "balanced_accuracy"):
            metric = row[key]
            if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric) or not 0 <= metric <= 1:
                raise Blocked("Invalid saved Validation metric: " + key)
        matrix = row["confusion_matrix_labels_0_benign_1_attack"]
        if (len(matrix) != 2 or any(len(r) != 2 for r in matrix)
                or any(type(v) is not int or v < 0 for r in matrix for v in r)
                or sum(map(sum, matrix)) != validation_rows
                or not math.isclose((matrix[0][0]+matrix[1][1])/validation_rows, row["accuracy"], abs_tol=1e-12)):
            raise Blocked("Saved Validation confusion matrix/counts differ")
        for seconds in (row["fit"]["seconds"], row["predict_seconds"]):
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise Blocked("Invalid measured runtime in k checkpoint")
        rows.append(row)
        previous = value["sha256"]
    return rows


def inspect_run(remote, *, local=None, expected_identity=None):
    """No writes anywhere. expected_identity, if supplied, must match in full."""
    remote = Path(remote)
    manifest = read(remote / "run_manifest.json")
    identity = manifest["identity"]
    if (manifest["format"] != FORMAT or manifest["algorithm"] != ALGORITHM
            or identity["format"] != FORMAT or identity["algorithm"] != ALGORITHM
            or identity["sha256"] != digest({k: v for k, v in identity.items() if k != "sha256"})
            or identity["source"]["sha256"] != digest(identity["source"]["files"])
            or identity["dataset_fingerprint"] != identity["audit"]["fingerprint"]
            or identity["audit"]["fingerprint"] != digest({k: v for k, v in identity["audit"].items() if k != "fingerprint"})
            or identity["dataset"] != identity["audit"]["binding"]):
        raise Blocked("Run/source/dataset identity is not intact A1")
    if expected_identity is not None and identity != expected_identity:
        raise Blocked("Resume BLOCKED: source/Git/dataset/config/parameters identity mismatch; no migration authorized")
    checkpoints = ledger(remote, identity)  # Checks COMMITTED markers, bytes, hashes and parent chain.
    artifacts, owners = {}, {}
    for checkpoint in checkpoints:
        for name, meta in checkpoint["record"]["files"].items():
            if name in artifacts and info(artifacts[name]) != meta:
                raise Blocked("Committed checkpoints disagree on immutable artifact: " + name)
            artifacts[name] = checkpoint["path"] / name
            owners[name] = checkpoint
    required = ("mi_state.json", "mi_top100.json", "mi_feature_scores.csv", "top100_scaler_state.json",
                "top100_scaler.joblib", "rfe_state.json", "rfe_feature_ranking.csv")
    if not set(required) <= set(artifacts):
        raise Blocked("Verified MI/scaler/RFE chain is incomplete; do not assume it is reusable")
    names = identity["dataset"]["names"]
    top, mi = read(artifacts["mi_top100.json"]), read(artifacts["mi_state.json"])
    with artifacts["mi_feature_scores.csv"].open(encoding="utf-8", newline="") as stream:
        scores = list(csv.reader(stream))[1:]
    values = [float(row[1]) for row in scores]
    if (mi["identity"] != identity["sha256"] or len(names) != 380 or len(set(names)) != 380
            or len(scores) != 380 or set(r[0] for r in scores) != set(names)
            or len(top) != 100 or len(set(top)) != 100 or top != [r[0] for r in scores[:100]]
            or values != sorted(values, reverse=True) or any(not math.isfinite(v) or v < 0 for v in values)):
        raise Blocked("Saved MI identity/order/scores differ")
    for name, meta in mi["files"].items():
        if name not in artifacts or info(artifacts[name]) != meta:
            raise Blocked("MI artifact SHA differs: " + name)
    scaler = read(artifacts["top100_scaler_state.json"])
    if (scaler["binding"] != {"identity": identity["sha256"], "mi": sha(artifacts["mi_state.json"]), "top": top}
            or scaler["fit_role"] != "train" or scaler["rows"] != identity["dataset"]["splits"]["train"]["rows"]):
        raise Blocked("Saved Train scaler binding differs")
    for name, meta in scaler["files"].items():
        if name not in artifacts or info(artifacts[name]) != meta:
            raise Blocked("Scaler artifact SHA differs: " + name)
    rfe = read(artifacts["rfe_state.json"])
    if rfe["binding"] != {"identity": identity["sha256"], "mi": sha(artifacts["mi_state.json"]),
                           "scaler": sha(artifacts["top100_scaler_state.json"]), "top": digest(top)}:
        raise Blocked("Saved A1 RFE MI/scaler binding differs")
    for name, meta in rfe["files"].items():
        if name not in artifacts or info(artifacts[name]) != meta:
            raise Blocked("RFE artifact SHA differs: " + name)
    ranked = validate_result(rfe["result"], top)
    with artifacts["rfe_feature_ranking.csv"].open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if csv_rows != [{k: "" if v is None else str(v) for k, v in row.items()} for row in rfe["result"]["records"]]:
        raise Blocked("Saved RFE CSV mapping differs")
    count = durable_completed_k(checkpoints)
    paths = []
    for k in range(1, count+1):
        stage, name = f"k_{k:03d}", f"k/k_{k:03d}.json"
        checkpoint = next(r for r in checkpoints if r["record"]["stage"] == stage)
        if name not in checkpoint["record"]["files"]:
            raise Blocked("Committed k stage lacks its actual k record")
        paths.append(checkpoint["path"] / name)
    rows = records(paths, identity, ranked, sha(artifacts["rfe_state.json"]), identity["dataset"]["splits"]["val"]["rows"])
    best = choose(rows, ranked) if rows else None
    local_status = {"status": "NOT_INSPECTED", "completed_k": None}
    if local is not None:
        local = Path(local)
        state = local / "state"
        if state.is_dir():
            if read(local / "run_manifest.json") != manifest or read(state / "run_identity.json") != identity:
                raise Blocked("Local run identity differs; do not merge")
            for name in required:
                if info(state / name) != info(artifacts[name]):
                    raise Blocked("Local MI/scaler/RFE differs from committed artifacts")
            for name, path in artifacts.items():
                if name.startswith("k/") and owners[name]["record"]["stage"].startswith("k_"):
                    if info(state / name) != info(path):
                        raise Blocked("Local k ancestry differs from committed checkpoint")
            local_rows = records(sorted((state / "k").glob("k_*.json")), identity, ranked,
                                 sha(artifacts["rfe_state.json"]), identity["dataset"]["splits"]["val"]["rows"])
            if len(local_rows) < count:
                raise Blocked("Local prefix is behind committed checkpoint; restore required")
            local_status = {"status": "VERIFIED_LOCAL_PREFIX_NOT_AUTOMATICALLY_COMMITTED", "completed_k": len(local_rows),
                            "pending_commit_k": list(range(count+1, len(local_rows)+1)), "next_fit_k_if_identity_matches": len(local_rows)+1}
        else:
            local_status = {"status": "LOCAL_STATE_ABSENT", "completed_k": None}
    return {"status": "PASS", "scope": "saved checkpoint chain only; no current source/dataset compatibility claimed",
            "identity_sha256": identity["sha256"], "saved_git_commit": identity["git_commit"],
            "saved_source_sha256": identity["source"]["sha256"], "dataset_fingerprint": identity["dataset_fingerprint"],
            "config": identity["config"], "run_id": identity["run_id"], "committed_k": list(range(1, count+1)),
            "k1_committed_and_verified": count >= 1, "next_fit_k_from_drive": count+1,
            "best_k_from_committed_validation": best["k"] if best is not None else None,
            "best_accuracy_from_committed_validation": best["accuracy"] if best is not None else None,
            "last_checkpoint_stage": checkpoints[-1]["record"]["stage"] if checkpoints else None,
            "failed_checkpoint_count": sum(r["record"]["stage"].startswith("failed_") for r in checkpoints),
            "mi_scaler_rfe_saved_artifacts_verified": True, "local": local_status,
            "incomplete_directories_ignored_not_deleted": [p.name for p in (remote / "checkpoints").glob(".incomplete_*")],
            "remote_writes": 0, "model_deserializations": 0, "dataset_reads": 0, "test_values_read": False,
            "migration_performed": False}
