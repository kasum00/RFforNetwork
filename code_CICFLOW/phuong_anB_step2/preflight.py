"""No fit: exact policy/source/environment/CICFLOW identities and resource gate."""
import inspect as pyinspect
import os
import shutil
from pathlib import Path
from .contract import (ALGORITHM, FORMAT, RFE_STEP, SCIENTIFIC, Blocked, confirmation,
                       core, digest, paths, read, scientific, source_snapshot, check_files)
from .dataset import inspect
from .policy import approved_policy
from .store import ledger, released
from .environment import environment
from mi_rfe_svm_colab.preflight import git_identity, resource_gate


def runtime_policy(runtime):
    seconds, margin, heartbeat = (runtime[k] for k in ("session_budget_seconds", "stop_margin_seconds", "heartbeat_seconds"))
    if (any(type(x) is not int for x in (seconds, margin, heartbeat)) or seconds < 0 or margin < 0
            or seconds > 0 and margin >= seconds or not 5 <= heartbeat <= 300):
        raise Blocked("Invalid runtime policy: budget0=unlimited, margin<budget, heartbeat5..300 seconds")


def review(request, *, require_tests=True):
    import psutil
    runtime_policy(request["runtime"])
    local, remote = paths(request)
    mode = request.get("mode", "preflight")
    if mode not in ("metadata", "preflight", "run", "resume", "verify"):
        raise Blocked("B is selection only; no Test or final-refit command")
    policy, env = approved_policy(request), environment()
    if env["errors"]:
        return {"status": "BLOCKED", "reasons": env["errors"], "environment": env,
                "worker_fit_started": False, "resource_gate": None, **SCIENTIFIC}
    approved = read(Path(__file__).parent / "evidence/approved_helpers.json")
    check_files(Path(request["source"]), approved["files"])
    source, git = source_snapshot(request["source"]), git_identity(request)
    audit = inspect(request)
    module = core(request["source"])
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    signature = pyinspect.signature(mutual_info_classif)
    if signature.parameters["discrete_features"].default != "auto" or signature.parameters["n_neighbors"].default != 3:
        raise Blocked("MI defaults differ from approved A1")
    identity = {"format": FORMAT, "algorithm": ALGORITHM, "rfe_step": RFE_STEP,
        "config": request["config"], "run_id": request["run_id"], "search_policy": policy,
        "source": source, "git_commit": git, "environment": env,
        "contract": scientific(request["source"], policy), "dataset_fingerprint": audit["fingerprint"],
        "dataset": audit["binding"], "audit": audit, "mi_signature": str(signature),
        "resolved_estimators": {"svm": module.make_model("svm").get_params(),
            "rfe": RandomForestClassifier(**module.RFE_PARAMS["svm"]).get_params()},
        "resource_budget": request["budget"]}
    identity["sha256"] = digest(identity)
    reasons, existing, active = [], [], None
    if remote.exists():
        if not (remote / "run_manifest.json").is_file() or read(remote / "run_manifest.json").get("identity") != identity:
            reasons.append("Existing B run has different source/dataset/policy/step/environment/config identity; no overwrite")
        else:
            existing = ledger(remote, identity)
            if mode == "run":
                reasons.append("B run already exists; explicitly resume/verify")
            claims = sorted((remote / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
            if claims and not (remote / "results/SELECTION_COMPLETE.json").is_file():
                prior = read(claims[-1])
                if not released(remote, prior):
                    active = prior["id"]
                    if request.get("takeover_confirmation") != "PREVIOUS RUNTIME STOPPED " + active:
                        reasons.append("Unreleased B worker: " + active)
    elif mode in ("resume", "verify"):
        reasons.append("No existing B run for resume/verify")
    if mode == "verify" and not (remote / "results/SELECTION_COMPLETE.json").is_file():
        reasons.append("B selection is not published complete")
    if (local / "run_manifest.json").is_file() and read(local / "run_manifest.json")["identity"] != identity:
        reasons.append("Local B state has a different identity; do not merge")
    ancestor = local
    while not ancestor.exists():
        ancestor = ancestor.parent
    counts = {r: row["rows"] for r, row in audit["binding"]["splits"].items()}
    gate = resource_gate(counts, request["budget"], sample={"available_ram_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(ancestor).free, "logical_cpus": os.cpu_count(),
        "rss_bytes": psutil.Process().memory_info().rss})
    completed = any(r["record"]["stage"] == "selection" for r in existing)
    if not completed and mode not in ("metadata", "verify"):
        reasons += gate["reasons"]
        tests = request.get("unit_tests", {})
        if require_tests and (tests.get("status") != "PASS" or tests.get("source_sha256") != source["sha256"]
                or tests.get("environment") != env or tests.get("tests_run", 0) <= 0 or tests.get("skipped", 0)):
            reasons.append("Missing/stale/failed B synthetic-test evidence")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons, "identity": identity,
            "resource_gate": gate, "local": str(local), "remote": str(remote), "audit": audit,
            "runtime_policy": request["runtime"], "last_checkpoint": existing[-1]["record"]["stage"] if existing else None,
            "unreleased_lease": active, "confirmation": confirmation(request), **SCIENTIFIC}
