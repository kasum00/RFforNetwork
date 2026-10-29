"""Metadata/resource admission; estimates are not a timing benchmark."""
import inspect as pyinspect
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from .common import (Blocked, FORMAT, SCIENTIFIC, SVM_PARAMS, RFE_PARAMS, contract,
                     core, digest, paths, read, snapshot)
from . import dataset
from .artifact_reuse import inspect_import


def environment():
    import importlib.metadata as im
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "psutil", "pyarrow")
    versions = {name: im.version(name) for name in packages}
    numpy_version = "1.26.4" if tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13) else "2.2.6"
    expected = {"numpy": numpy_version, "pandas": "2.3.3", "scipy": "1.15.3",
                "scikit-learn": "1.7.2", "joblib": "1.5.3"}
    errors = [f"{k}: expected {v}, installed {versions[k]}" for k, v in expected.items() if versions[k] != v]
    if not (3, 10) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 14):
        errors.append("Reference dependency profile supports Python 3.10..3.13")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "errors": errors}


def git_identity(request):
    repo = Path(request["source"]).resolve().parent
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              text=True, capture_output=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if (not re.fullmatch(r"[0-9a-f]{40}", str(request.get("git_commit", "")))
            or request["git_commit"] != commit or git("status", "--porcelain")):
        raise Blocked("Require the exact approved 40-character commit and clean source checkout")
    return commit


def estimate(rows):
    train, val = rows["train"], rows["val"]
    # MI dense float64 copies + RF + raw/local mappings + current SVC prefixes
    # + unchanged 10000 MiB libsvm cache + 2 GiB headroom. No NxN kernel matrix.
    ram = train*380*24 + (train+val)*380*4 + (train+val)*100*24 + 10000*1024**2 + 2*1024**3
    disk = (train+val)*380*4 + (train+val)*100*8 + 4*1024**3
    return {"required_ram_bytes": ram, "required_local_disk_bytes": disk,
            "cache_size_mib": 10000, "runtime_prediction": None, "is_estimate_not_guarantee": True}


def resource_gate(rows, budget, *, sample=None):
    import psutil
    expected = estimate(rows)
    sample = sample or {"available_ram_bytes": psutil.virtual_memory().available,
                        "disk_free_bytes": shutil.disk_usage("/content").free,
                        "logical_cpus": os.cpu_count(), "rss_bytes": psutil.Process().memory_info().rss}
    reasons = []
    if budget.get("min_available_ram_bytes", 0) < 1024**3 or budget.get("max_process_rss_bytes", 0) <= 0:
        reasons.append("Explicit positive RAM/RSS budget required")
    if sample["available_ram_bytes"] < expected["required_ram_bytes"]:
        reasons.append("RAM below admission estimate; no automatic sampling/cache-size reduction")
    if sample["disk_free_bytes"] < expected["required_local_disk_bytes"]:
        reasons.append("Local disk below full-cache/bundle estimate")
    if budget.get("max_process_rss_bytes", 0) < expected["required_ram_bytes"]:
        reasons.append("RSS budget below conservative admission estimate")
    if not sample.get("logical_cpus"):
        reasons.append("CPU information unavailable")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons,
            "estimate": expected, "observed": sample, "gpu_accelerates_sklearn_svc": False}


def review(request, *, need_fit=True):
    import psutil
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    source = Path(request["source"]).resolve()
    module = core(source)
    local, remote = paths(request)
    env, source_info = environment(), snapshot(source)
    signature = pyinspect.signature(mutual_info_classif)
    if (signature.parameters["discrete_features"].default != "auto"
            or signature.parameters["n_neighbors"].default != 3):
        raise Blocked("MI defaults differ from reference")
    git = git_identity(request)
    audit = dataset.inspect(request)
    reuse = inspect_import(request, audit, env)
    identity = {"format": FORMAT, "spec": {"experiment": "mi_rfe_svm", "config": request["config"], "run_name": remote.name},
                "source": source_info, "git_commit": git, "environment": env,
                "contract": contract(source), "dataset": audit["binding"],
                "dataset_audit": audit, "dataset_fingerprint": audit["fingerprint"],
                "reuse": reuse, "budget": request["budget"],
                "resolved_estimators": {"svm": SVC(**SVM_PARAMS).get_params(),
                                        "rfe": RandomForestClassifier(**RFE_PARAMS).get_params()},
                "mi_signature": str(signature)}
    identity["sha256"] = digest(identity)
    reasons = list(env["errors"])
    checkpoint = None
    active_lease = None
    legacy_status = None
    if remote.exists():
        if not (remote / "OWNER.json").exists() or not (remote / "run_identity.json").exists():
            reasons.append("Existing/legacy run is protected; choose run02 or verify with its original code")
        elif read(remote / "run_identity.json") != identity:
            reasons.append("Existing run has a different identity; never overwrite")
        elif request.get("mode") == "run":
            reasons.append("Run exists: explicitly choose resume/verify")
        if (remote / "COMPLETE.json").exists():
            try:
                legacy_status = module.verify_directory(remote)
            except Exception as exc:
                legacy_status = {"status": "BLOCKED", "error": str(exc)}
        if (remote / "OWNER.json").exists() and (remote / "run_identity.json").exists() and read(remote / "run_identity.json") == identity:
            from .drive_checkpoint import latest, lease_released
            checkpoint = latest(remote, identity["sha256"])
            claims = sorted((remote / "leases").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
            if claims:
                last_lease = read(claims[-1])
                if not lease_released(remote, last_lease):
                    active_lease = last_lease["id"]
                    if request.get("takeover_confirmation") != "PREVIOUS RUNTIME STOPPED " + active_lease:
                        reasons.append("Unreleased job lease: " + active_lease + "; explicitly confirm previous runtime stopped")
    elif request.get("mode") in ("resume", "finalize", "verify"):
        reasons.append("No existing run for this mode")
    if local.exists() and (local / "state/run_identity.json").exists() and read(local / "state/run_identity.json") != identity:
        reasons.append("Local state identity differs")
    local_parent = local
    while not local_parent.exists():
        local_parent = local_parent.parent
    sample = {"available_ram_bytes": psutil.virtual_memory().available,
              "disk_free_bytes": shutil.disk_usage(local_parent).free, "logical_cpus": os.cpu_count(),
              "rss_bytes": psutil.Process().memory_info().rss}
    gate = resource_gate(module.EXPECTED[request["config"]], request["budget"], sample=sample)
    if need_fit:
        reasons += gate["reasons"]
        evidence = request.get("unit_tests", {})
        if (evidence.get("status") != "PASS" or evidence.get("source_sha256") != source_info["sha256"]
                or evidence.get("environment") != env or evidence.get("tests_run", 0) <= 0
                or evidence.get("skipped", 0) != 0):
            reasons.append("Run new synthetic tests: missing/stale/skipped evidence")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons, "resource_gate": gate,
            "identity": identity, "audit": audit, "local": str(local), "remote": str(remote),
            "values_read": False,
            "test_claim_present": (remote / "TEST_CLAIM.json").exists(),
            "latest_committed_checkpoint": checkpoint, "unreleased_lease": active_lease,
            "legacy_reference_layout_status": legacy_status,
            "test_claim_policy": "Saved result may be published; an ambiguous claim always blocks re-evaluation",
            **SCIENTIFIC}
