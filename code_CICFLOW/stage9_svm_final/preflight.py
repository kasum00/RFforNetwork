"""Read-only scientific admission. No model fit or Test file open."""
import os
import shutil
import subprocess
from pathlib import Path
from .common import (Blocked, FORMAT, METHODS, SCALER_PROTOCOL, SVM_PARAMS, digest,
                     files_checked, intact, paths, read, sha)
from .selection import verify_selection
from .environment import environment


def source_snapshot(source):
    source = Path(source).resolve()
    package = source / "stage9_svm_final"
    files = [p for p in package.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".txt")
             and not any(part in p.parts for part in ("reports", "__pycache__", "runs"))]
    files += [source.parent / "notebooks/STAGE9_SVM_A1_B_FINAL_EVALUATION_COLAB.ipynb"]
    records = {p.relative_to(source.parent).as_posix(): sha(p) for p in sorted(files)}
    return dict(files=records, sha256=digest(records))


def check_source(request):
    approved_commit = request.get("git_commit")
    if not isinstance(approved_commit, str) or len(approved_commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in approved_commit):
        raise Blocked("Missing or invalid approved git_commit in Stage9 request")
    repo = Path(request["source"]).resolve().parent
    for args, expected in ((("rev-parse", "HEAD"), request["git_commit"]), (("status", "--porcelain"), "")):
        actual = subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()
        if actual != expected:
            raise Blocked("Stage9 checkout is not clean at the approved commit")
    files_checked(request["source"], read(Path(__file__).with_name("reference_helpers.json")))
    return source_snapshot(request["source"])


def positive_labels(audit):
    mapping = audit["label_mapping"]
    if (mapping["0"] != "Benign_folder_proxy" or mapping["1"] != "Attack_folder_proxy"
            or mapping["status"] != "PROVISIONAL_FOLDER" or not mapping["producer_sha256"]
            or mapping["producer_sha256"] != audit["split_evidence"]["label_producer"]["sha256"]):
        raise Blocked("Positive Attack label has no verified producer/manifest mapping")
    return dict(benign=0, attack=1, mapping=mapping)


def resource_gate(audit, budget, local, sample=None):
    import psutil
    train, val = (audit["binding"]["splits"][role]["rows"] for role in ("train", "val"))
    # Full 380-column raw inputs + float32 combined/scaler copies + float64 libsvm + cache.
    ram = (train+val)*(380*4 + 100*32) + SVM_PARAMS["cache_size"]*1024**2 + 2*1024**3
    disk = (train+val)*380*4 + 4*1024**3
    ancestor = Path(local)
    while not ancestor.exists():
        ancestor = ancestor.parent
    sample = sample or dict(available_ram_bytes=psutil.virtual_memory().available,
        disk_free_bytes=shutil.disk_usage(ancestor).free, logical_cpus=os.cpu_count(),
        rss_bytes=psutil.Process().memory_info().rss)
    reasons = []
    if sample["available_ram_bytes"] < ram or budget["max_process_rss_bytes"] < ram:
        reasons.append("RAM/RSS budget below conservative full-data estimate; no auto reduction")
    if budget["min_available_ram_bytes"] < 1024**3 or sample["disk_free_bytes"] < disk:
        reasons.append("Invalid RAM reserve or insufficient local disk")
    return dict(status="BLOCKED" if reasons else "PASS", reasons=reasons, observed=sample,
                estimate=dict(required_ram_bytes=ram, required_local_disk_bytes=disk, is_estimate_not_guarantee=True))


def review(request, *, need_fit=False):
    local, remote = paths(request)
    env = environment()
    if env["errors"]:
        raise Blocked("Environment: " + "; ".join(env["errors"]))
    source = check_source(request)
    selected = verify_selection(request)
    for name in ("colab/cicflow_reference_pipeline.py", "mi_rfe_svm_colab/common.py"):
        if selected["verified_helpers"][name] != sha(Path(request["source"]) / name):
            raise Blocked("Archived/current final preprocessing helper differs; reconcile source before final fit: " + name)
    from mi_rfe_svm_colab.common import core
    resolved = core(request["source"]).make_model("svm").get_params()
    if (selected["reference_final_role"] != "Train+Validation after frozen selection"
            or selected["identity"]["resolved_estimators"]["svm"] != resolved
            or any(resolved[k] != v for k, v in SVM_PARAMS.items())):
        raise Blocked("BLOCKED_SCIENTIFIC_CONTRACT: final scaler/SVC source mismatch")
    labels = positive_labels(selected["audit"])
    approved_commit = request.get("git_commit")
    if not isinstance(approved_commit, str) or len(approved_commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in approved_commit):
        raise Blocked("Missing or invalid approved git_commit in Stage9 request")
    identity = dict(format=FORMAT, method=request["method"], config=request["config"],
        run_id=request["run_id"], selection_run_id=request["selection_run_id"],
        selection_identity=selected["identity"]["sha256"], selection_checkpoint_sha256=selected["checkpoint_sha256"],
        selection_publication=selected["publication"], verified_helpers=selected["verified_helpers"], dataset=selected["audit"],
        features=selected["features"], best_k=selected["best_k"], frozen_digest=digest(selected["frozen"]),
        rfe_step=METHODS[request["method"]][2],
        source=source, git_commit=request["git_commit"], selection_commit=selected["identity"]["git_commit"],
        environment=env, svm=SVM_PARAMS, svm_resolved=resolved, scaler_protocol=SCALER_PROTOCOL, labels=labels)
    identity["sha256"] = digest(identity)
    if remote.exists():
        if not (remote / "identity_manifest.json").is_file() or read(remote / "identity_manifest.json") != identity:
            raise Blocked("Existing Stage9 method/config has another identity; preserve it")
        action = request.get("mode", "preflight")
        if need_fit or action in ("test-readiness", "test"):
            for lease in sorted((remote / "leases").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json")):
                old = read(lease)
                released = lease.with_name(lease.stem + ".RELEASED.json")
                if not released.is_file():
                    expected = "PREVIOUS STAGE9 WORKER STOPPED " + old["token"]
                    if old["identity"] != identity["sha256"] or request.get("previous_worker_stopped") != expected:
                        raise Blocked("Unreleased Stage9 worker; do not assume stopped: " + old["token"])
    gate = resource_gate(selected["audit"], request["budget"], local)
    if need_fit:
        if gate["status"] != "PASS":
            raise Blocked("Resource gate: " + "; ".join(gate["reasons"]))
        tests = request.get("unit_tests", {})
        if (tests.get("status") != "PASS" or tests.get("source_sha256") != source["sha256"]
                or tests.get("environment") != env or tests.get("tests_run", 0) <= 0
                or tests.get("skipped", 0) or tests.get("failures", 1) or tests.get("errors", 1)):
            raise Blocked("Missing/current-source synthetic unit-test PASS evidence")
    return dict(status="PASS", identity=identity, selected=selected, resource_gate=gate,
                local=str(local), directory=str(remote), test_values_read=False, worker_fit_started=False)
