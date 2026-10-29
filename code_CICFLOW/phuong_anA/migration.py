"""Explicit, no-fit migration of completed A1 MI/scaler/RFE and k1.

The origin run and recovery copy are READ ONLY. Plan is read only. Apply writes
only a new run and staging directories, and publishes run_manifest.json last.
Existing output is never overwritten. Standard resume identity gates are intact.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, SCIENTIFIC, Blocked, atomic, check_files,
                       core, digest, drive_json_once, info, paths, read, redact, sha)
from .recovery import inspect_run, records
from .results import validate_ranking
from .store import Store, ledger, released
from mi_rfe_svm_colab.local_progress import KProgress

MIGRATION_FORMAT = "phuong_anA.k1-keyerror-migration.v1"
# Byte SHA256 of the actual files read before/after the reviewed correction.
# An unknown transition requires a new review, not an expanded wildcard rule.
REVIEWED_TRANSITIONS = {
    "phuong_anA/telemetry.py": (
        "15f56ab7046c313471968a220608ee539175f5cbde097128a50ccb0e7e95eb5c",
        "d8837bfa45b8882d9964ca7d452cf30bbf24bf90a23f4722cba13f52c0b07107"),
    "phuong_anA/pipeline.py": (
        "1dc9f88e53b972cebafb279eebccaa7c7746478b7b283a06ae42149fc821d7e4",
        "df8317ad76d7e119382702ea3b2ac05d04b065313f0586f8ff4e9a040391c805"),
}
REVIEWED_ADDITIONS = {
    "phuong_anA/recovery.py", "phuong_anA/migration.py",
    "phuong_anA/tests/test_best_accuracy_recovery.py", "phuong_anA/tests/test_migration.py",
}
PHASE_FILES = {
    "mi": ("mi_top100.json", "mi_feature_scores.csv", "mi_state.json"),
    "scaler": ("top100_scaler.joblib", "top100_scaler_state.json"),
    "rfe": ("rfe_feature_ranking.csv", "rfe_state.json"),
}


def source_compatibility(old, new):
    for snapshot in (old, new):
        if snapshot["sha256"] != digest(snapshot["files"]):
            raise Blocked("Source snapshot digest differs")
    before, after = old["files"], new["files"]
    removed, added = set(before)-set(after), set(after)-set(before)
    if removed or not added <= REVIEWED_ADDITIONS:
        raise Blocked("Unreviewed source additions/removals: " + str(sorted(removed | (added-REVIEWED_ADDITIONS))))
    changed = []
    for name in sorted(set(before) & set(after)):
        if before[name] == after[name]:
            continue
        if name not in REVIEWED_TRANSITIONS or (before[name], after[name]) != REVIEWED_TRANSITIONS[name]:
            raise Blocked("Unreviewed source change; numerical compatibility not established: " + name)
        changed.append(name)
    if set(changed) != set(REVIEWED_TRANSITIONS):
        raise Blocked("Migration only supports the reviewed best_accuracy KeyError correction")
    return {"changed": changed, "added": sorted(added), "removed": [],
            "all_other_source_hashes_identical": True, "policy": MIGRATION_FORMAT}


def artifacts(checkpoints):
    result = {}
    for checkpoint in checkpoints:
        for name, meta in checkpoint["record"]["files"].items():
            if name in result and info(result[name]) != meta:
                raise Blocked("Old checkpoints disagree on immutable artifact: " + name)
            result[name] = checkpoint["path"] / name
    return result


def prepare(request, origin, recovery):
    """Read-only plan; no directories, caches, leases, model loads or fits."""
    from .preflight import review
    request = {**request, "mode": "preflight"}
    local, remote = paths(request)
    origin, recovery = Path(origin).resolve(), Path(recovery).resolve()
    if remote.exists() or local.exists():
        raise Blocked("Migration target already exists; preserve it and select a fresh explicit run ID")
    origin_manifest = read(origin / "run_manifest.json")
    old = origin_manifest["identity"]
    expected_origin = Path(request["output"]).resolve() / "phuong_anA/mi_rfe_svm" / old["config"] / old["run_id"]
    expected_backup = Path(request["output"]).resolve() / "phuong_anA/recovery" / old["config"] / old["run_id"]
    if (origin != expected_origin or origin == remote or old["config"] != request["config"]
            or old["run_id"] == request["run_id"] or expected_backup not in recovery.parents):
        raise Blocked("Origin/recovery/target paths or config do not match isolated A1 namespaces")
    audit = inspect_run(origin)
    checkpoints = ledger(origin, old)
    if audit["committed_k"] not in ([], [1]) or (origin / "results/SELECTION_COMPLETE.json").exists():
        raise Blocked("This migration supports the observed failed run with zero or one committed k only")
    leases = sorted((origin / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    if not leases or not released(origin, read(leases[-1])):
        raise Blocked("Origin worker not proven released; never migrate a running session")
    original = artifacts(checkpoints)
    backup_manifest_path = recovery / "recovery_manifest.json"
    backup = read(backup_manifest_path)
    expected_names = {"k_001.json", "original_run_manifest.json", "original_rfe_state.json"}
    if (backup["purpose"] != "RECOVERY_COPY_ONLY_NOT_RUN_CHECKPOINT"
            or Path(backup["original_run"]).resolve() != origin or backup["original_identity"] != old["sha256"]
            or backup["original_git_commit"] != old["git_commit"]
            or backup["dataset_fingerprint"] != old["dataset_fingerprint"] or set(backup["files"]) != expected_names):
        raise Blocked("Recovery copy provenance/identity differs")
    check_files(recovery, backup["files"])
    if (info(recovery / "original_run_manifest.json") != info(origin / "run_manifest.json")
            or info(recovery / "original_rfe_state.json") != info(original["rfe_state.json"])):
        raise Blocked("Recovery copy is not byte-identical to the verified origin run/RFE")
    old_rfe = read(original["rfe_state.json"])
    ranked = old_rfe["result"]["names"]
    recovered_rows = records([recovery / "k_001.json"], old, ranked, sha(original["rfe_state.json"]),
                             old["dataset"]["splits"]["val"]["rows"])
    if audit["committed_k"]:
        committed = next(r for r in checkpoints if r["record"]["stage"] == "k_001")
        if info(committed["path"] / "k/k_001.json") != info(recovery / "k_001.json"):
            raise Blocked("Recovery k1 conflicts with committed k1")
    plan = review(request, require_tests=True)  # Exact current Git/source/env/data/resource/test evidence.
    if plan["status"] != "PASS":
        raise Blocked("New-run preflight BLOCKED: " + "; ".join(plan["reasons"]))
    new = plan["identity"]
    if (new["sha256"] != digest({k: v for k, v in new.items() if k != "sha256"})
            or new["config"] != request["config"] or new["run_id"] != request["run_id"]
            or new["algorithm"] != ALGORITHM or new["format"] != FORMAT):
        raise Blocked("New-run preflight returned a different/incomplete identity")
    allowed_identity_changes = {"sha256", "run_id", "git_commit", "source"}
    if set(old) != set(new) or any(old[k] != new[k] for k in set(old)-allowed_identity_changes):
        raise Blocked("Dataset/splits/environment/defaults/parameters/resource budget differ; migration forbidden")
    compatibility = source_compatibility(old["source"], new["source"])
    result = {"format": MIGRATION_FORMAT, "algorithm": ALGORITHM, "origin": str(origin),
        "recovery": str(recovery), "target": str(remote), "local_target": str(local),
        "original_identity": old, "new_identity": new, "source_compatibility": compatibility,
        "origin_manifest": info(origin / "run_manifest.json"), "origin_ledger_head": checkpoints[-1]["sha256"],
        "original_artifacts": {name: info(original[name]) for names in PHASE_FILES.values() for name in names},
        "recovery_manifest": info(backup_manifest_path), "recovered_k1": info(recovery / "k_001.json"),
        "original_committed_k": audit["committed_k"], "transferred_k": [1], "next_fit_k": 2,
        "validation_accuracy_k1": recovered_rows[0]["accuracy"], "training_performed": False, **SCIENTIFIC}
    result["plan_sha256"] = digest(result)
    return result


def phrase(plan):
    new = plan["new_identity"]
    return f"MIGRATE EXPERIMENTAL A1 {new['config']}/{new['run_id']} {plan['plan_sha256'][:12]}"


def copy_once(source, target):
    source, target = Path(source), Path(target)
    expected = info(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8*1024**2)
        dst.flush(); os.fsync(dst.fileno())
    if info(target) != expected:
        raise Blocked("Migration copy SHA256/size differs: " + str(target))


def copy_tree_once(source, destination, *, exclude=()):
    for path in sorted(Path(source).rglob("*")):
        if path.is_symlink():
            raise Blocked("Migration staging contains a symlink")
        if path.is_file() and path.relative_to(source).as_posix() not in exclude:
            copy_once(path, Path(destination) / path.relative_to(source))


def rebind(store, plan, original, k1_path):
    """No estimator load/fit: rewrite dependency identities on new copies only."""
    state, identity = store.state, store.identity
    atomic(state / "run_identity.json", identity, exclusive=True)
    atomic(state / "MIGRATION_MANIFEST.json", plan, exclusive=True)
    store.commit("init", {n: state / n for n in ("run_identity.json", "MIGRATION_MANIFEST.json")},
                 metadata={"origin_ledger_head": plan["origin_ledger_head"], "migration_plan_sha256": plan["plan_sha256"]})
    for name in ("mi_top100.json", "mi_feature_scores.csv", "top100_scaler.joblib", "rfe_feature_ranking.csv"):
        copy_once(original[name], state / name)
    mi = read(original["mi_state.json"])
    mi["identity"] = identity["sha256"]
    atomic(state / "mi_state.json", mi, exclusive=True)
    top = read(state / "mi_top100.json")
    scaler = read(original["top100_scaler_state.json"])
    scaler["binding"] = {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top}
    atomic(state / "top100_scaler_state.json", scaler, exclusive=True)
    rfe = read(original["rfe_state.json"])
    rfe["binding"] = {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
                       "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}
    atomic(state / "rfe_state.json", rfe, exclusive=True)
    _, ranked = validate_ranking(state, identity)
    metadata = {"migration_plan_sha256": plan["plan_sha256"], "reused_without_fit": True,
        "feature_names_order": ranked, "feature_order_sha256": digest(ranked),
        "scaler_identity": sha(state / "top100_scaler_state.json"), "completed_k": 0,
        "train_val_manifest_identity": digest({r: identity["dataset"]["splits"][r] for r in ("train", "val")})}
    for stage, names in PHASE_FILES.items():
        store.commit(stage, {name: state / name for name in names}, metadata=metadata)
    row = read(k1_path)["row"]
    history = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
    history.append(row)  # Same measured metrics, features, timing, warnings; new dependency binding.
    validated = records([state / "k/k_001.json"], identity, ranked, sha(state / "rfe_state.json"),
                        identity["dataset"]["splits"]["val"]["rows"])
    if validated != [row]:
        raise Blocked("Rebound k1 measured data changed")
    store.commit("k_001", {"k/k_001.json": state / "k/k_001.json"}, metadata={**metadata,
        "completed_k": 1, "completed_k_values": [1], "original_k1_sha256": plan["recovered_k1"]["sha256"]})
    return {name: info(state / name) for name in
            ("run_identity.json", "MIGRATION_MANIFEST.json", *[n for ns in PHASE_FILES.values() for n in ns], "k/k_001.json")}


def verify_transfer(origin_artifacts, migrated, plan):
    """Prove only identity links changed; numerical/measurement payloads are equal."""
    state = Path(migrated)
    for name in ("mi_top100.json", "mi_feature_scores.csv", "top100_scaler.joblib", "rfe_feature_ranking.csv"):
        if info(state / name) != plan["original_artifacts"][name]:
            raise Blocked("Byte-identical numerical artifact changed: " + name)
    for name, removed in (("mi_state.json", "identity"), ("top100_scaler_state.json", "binding"), ("rfe_state.json", "binding")):
        before, after = read(origin_artifacts[name]), read(state / name)
        if {k: v for k, v in before.items() if k != removed} != {k: v for k, v in after.items() if k != removed}:
            raise Blocked("Migration changed scientific state beyond identity: " + name)
    if read(state / "k/k_001.json")["row"] != read(Path(plan["recovery"]) / "k_001.json")["row"]:
        raise Blocked("Migration changed k1 metrics/features/timing")


def apply(request, origin, recovery, *, plan_sha256, authorization):
    plan = prepare(request, origin, recovery)  # Recheck, do not trust a caller-supplied identity/plan.
    if plan["plan_sha256"] != plan_sha256 or authorization != phrase(plan):
        raise Blocked("Migration plan changed or exact run-specific authorization is missing")
    local, remote = paths(request)
    original_identity = plan["original_identity"]
    old_checkpoints = ledger(Path(origin), original_identity)
    original = artifacts(old_checkpoints)
    # Reserve the destination. Standard preflight sees no run_manifest and blocks
    # every worker until the entire verified migration has been published.
    remote.mkdir(parents=True, exist_ok=False)
    drive_json_once(remote / "MIGRATION_INTENT.json", {"format": MIGRATION_FORMAT,
        "plan_sha256": plan_sha256, "status": "RESERVATION_ONLY_NOT_COMPLETION"})
    local.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    local_stage = local.with_name(".incomplete_migration_" + local.name + "_" + token)
    remote_stage = remote.with_name(".incomplete_migration_" + remote.name + "_" + token)
    module = core(request["source"])
    with module.run_lock(local):
        store = Store(local_stage, remote_stage, plan["new_identity"], resume=False)
        new_files = rebind(store, plan, original, Path(recovery) / "k_001.json")
        verify_transfer(original, store.state, plan)
        staged = inspect_run(remote_stage, expected_identity=plan["new_identity"])
        if staged["committed_k"] != [1] or staged["best_accuracy_from_committed_validation"] != plan["validation_accuracy_k1"]:
            raise Blocked("New committed prefix does not equal verified recovered k1")
        store.release("PAUSED")
        receipt = {"format": MIGRATION_FORMAT, "status": "MIGRATION_COMPLETE_NOT_TRAINING",
            "plan_sha256": plan_sha256, "identity": plan["new_identity"]["sha256"], "files": new_files,
            "ledger_head": ledger(remote_stage, plan["new_identity"])[-1]["sha256"],
            "transferred_k": [1], "next_fit_k": 2, "training_performed": False, **SCIENTIFIC}
        drive_json_once(remote_stage / "MIGRATION_MANIFEST.json", plan)
        drive_json_once(remote_stage / "MIGRATION_COMPLETE.json", receipt)
        # A complete local restore is ready before the remote admission manifest.
        copy_tree_once(local_stage, local)
        copy_tree_once(remote_stage, remote, exclude=("run_manifest.json",))
        check_files(local / "state", new_files)
        # Check staged publication bytes, then publish run manifest LAST.
        for path in remote_stage.rglob("*"):
            if path.is_file() and path.name != "run_manifest.json":
                if info(remote / path.relative_to(remote_stage)) != info(path):
                    raise Blocked("Destination publication checksum differs")
        # Recheck the origin/recovery; never modify them or let drift go unnoticed.
        if (ledger(Path(origin), original_identity)[-1]["sha256"] != plan["origin_ledger_head"]
                or info(Path(origin) / "run_manifest.json") != plan["origin_manifest"]
                or info(Path(recovery) / "recovery_manifest.json") != plan["recovery_manifest"]
                or info(Path(recovery) / "k_001.json") != plan["recovered_k1"]):
            raise Blocked("Origin/recovery changed during migration; new run remains inadmissible")
        copy_once(remote_stage / "run_manifest.json", remote / "run_manifest.json")
    result = verify_migration(remote)
    return {**result, "local": str(local), "origin_preserved": str(Path(origin)), "training_started": False}


def verify_migration(remote):
    """Read-only verification of a fully published new migration run."""
    remote = Path(remote)
    plan, receipt = read(remote / "MIGRATION_MANIFEST.json"), read(remote / "MIGRATION_COMPLETE.json")
    if (plan["plan_sha256"] != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan["format"] != MIGRATION_FORMAT or plan["algorithm"] != ALGORITHM
            or receipt["format"] != MIGRATION_FORMAT or receipt["plan_sha256"] != plan["plan_sha256"]
            or receipt["status"] != "MIGRATION_COMPLETE_NOT_TRAINING"
            or receipt["identity"] != plan["new_identity"]["sha256"] or receipt["transferred_k"] != [1]
            or receipt["training_performed"] is not False or read(remote / "run_manifest.json")["identity"] != plan["new_identity"]):
        raise Blocked("Published migration provenance/identity differs")
    if source_compatibility(plan["original_identity"]["source"], plan["new_identity"]["source"]) != plan["source_compatibility"]:
        raise Blocked("Published source compatibility proof differs")
    for key in set(plan["original_identity"])-{"sha256", "run_id", "git_commit", "source"}:
        if plan["original_identity"][key] != plan["new_identity"][key]:
            raise Blocked("Published migration changed scientific/data identity")
    current = inspect_run(remote, expected_identity=plan["new_identity"])
    rows = ledger(remote, plan["new_identity"])
    initial = next(r for r in rows if r["record"]["stage"] == "init")
    if info(initial["path"] / "MIGRATION_MANIFEST.json") != info(remote / "MIGRATION_MANIFEST.json"):
        raise Blocked("Root migration manifest differs from committed proof")
    mapped = artifacts(rows)
    for name, expected in receipt["files"].items():
        if name not in mapped or info(mapped[name]) != expected:
            raise Blocked("Published migration artifact differs: " + name)
    k1_checkpoint = next(r for r in rows if r["record"]["stage"] == "k_001")
    if k1_checkpoint["sha256"] != receipt["ledger_head"]:
        raise Blocked("Migration receipt does not bind k1 commit")
    # A later normal resume can append k2...; k1/provenance must remain immutable.
    return {"status": "PASS", "phase_status": "MIGRATION_VERIFIED", "directory": str(remote),
        "migration_transferred_k": [1], "current_committed_k": current["committed_k"],
        "next_fit_k": current["next_fit_k_from_drive"], "original_accuracy_k1": plan["validation_accuracy_k1"],
        "training_started": False, **SCIENTIFIC}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "verify"))
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        if args.action == "plan":
            plan = prepare(payload["request"], payload["origin"], payload["recovery"])
            result = {"status": "PASS", "plan": plan, "authorization_phrase": phrase(plan), "training_started": False}
        elif args.action == "apply":
            result = apply(payload["request"], payload["origin"], payload["recovery"],
                           plan_sha256=payload["plan_sha256"], authorization=payload["authorization"])
        else:
            result = verify_migration(payload["remote"])
        print("__A1_MIGRATION_RESULT__" + json.dumps(result, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print("__A1_MIGRATION_RESULT__" + json.dumps({"status": "BLOCKED", "message": redact(exc),
            "training_started": False, **SCIENTIFIC}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
