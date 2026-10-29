"""A1 preflight: actual CICFLOW manifests, exact source, versions and CPU budget."""
import inspect as pyinspect
import os
import shutil
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, SCIENTIFIC, Blocked, confirmation, core, digest,
                       paths, read, scientific, source_snapshot)
from .dataset import inspect
from .store import ledger, released
from mi_rfe_svm_colab.preflight import environment, git_identity, resource_gate


def runtime_policy(runtime):
    seconds, margin, heartbeat = (runtime[k] for k in ("session_budget_seconds", "stop_margin_seconds", "heartbeat_seconds"))
    if (any(type(x) is not int for x in (seconds, margin, heartbeat)) or seconds < 0 or margin < 0
            or seconds > 0 and margin >= seconds or not 5 <= heartbeat <= 300):
        raise Blocked("Invalid runtime policy: budget0=unlimited, margin<positive budget, heartbeat5..300 seconds")


def review(request, *, require_tests=True):
    import psutil
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    runtime_policy(request["runtime"])
    local, remote = paths(request)
    if request.get("mode", "preflight") not in ("metadata", "preflight", "run", "resume", "verify"):
        raise Blocked("A1 only permits metadata/preflight/run/resume/verify; final refit/Test are not authorized")
    source, env = source_snapshot(request["source"]), environment()
    git = git_identity(request)
    audit = inspect(request)
    module = core(request["source"])
    signature = pyinspect.signature(mutual_info_classif)
    if signature.parameters["discrete_features"].default != "auto" or signature.parameters["n_neighbors"].default != 3:
        raise Blocked("MI default contract differs from A0")
    identity = {"format": FORMAT, "algorithm": ALGORITHM, "config": request["config"], "run_id": request["run_id"],
                "source": source, "git_commit": git, "environment": env, "contract": scientific(request["source"]),
                "dataset_fingerprint": audit["fingerprint"], "dataset": audit["binding"], "audit": audit,
                "resolved_estimators": {"svm": module.make_model("svm").get_params(),
                    "rfe": RandomForestClassifier(**module.RFE_PARAMS["svm"]).get_params()},
                "mi_signature": str(signature), "resource_budget": request["budget"]}
    identity["sha256"] = digest(identity)
    reasons, existing, active = list(env["errors"]), [], None
    if remote.exists():
        if not (remote / "run_manifest.json").is_file() or read(remote / "run_manifest.json").get("identity") != identity:
            reasons.append("Existing run is foreign or has different dataset/source/config/algorithm identity")
        else:
            existing = ledger(remote, identity)
            if request.get("mode") == "run":
                reasons.append("Run exists: explicitly resume/verify; do not overwrite")
            claims = sorted((remote / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
            if claims and not (remote / "results/SELECTION_COMPLETE.json").exists():
                prior = read(claims[-1])
                if not released(remote, prior):
                    active = prior["id"]
                    if request.get("takeover_confirmation") != "PREVIOUS RUNTIME STOPPED " + active:
                        reasons.append("Unreleased A1 worker: " + active)
    elif request.get("mode") in ("resume", "verify"):
        reasons.append("No existing A1 run to resume/verify")
    ancestor = local
    while not ancestor.exists():
        ancestor = ancestor.parent
    counts = {role: row["rows"] for role, row in audit["binding"]["splits"].items()}
    gate = resource_gate(counts, request["budget"], sample={"available_ram_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(ancestor).free, "logical_cpus": os.cpu_count(),
        "rss_bytes": psutil.Process().memory_info().rss})
    selection_saved = any(r["record"]["stage"] == "selection" for r in existing)
    if not selection_saved and request.get("mode") not in ("metadata", "verify"):
        reasons += gate["reasons"]
        tests = request.get("unit_tests", {})
        if require_tests and (tests.get("status") != "PASS" or tests.get("source_sha256") != source["sha256"]
                or tests.get("environment") != env or tests.get("tests_run", 0) <= 0 or tests.get("skipped", 0)):
            reasons.append("Missing/stale/failed A1 synthetic-test evidence")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons, "identity": identity,
            "resource_gate": gate, "local": str(local), "remote": str(remote), "audit": audit,
            "runtime_policy": request["runtime"], "last_checkpoint": existing[-1]["record"]["stage"] if existing else None,
            "unreleased_lease": active, "confirmation": confirmation(request), **SCIENTIFIC}
