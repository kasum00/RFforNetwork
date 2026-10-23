"""Windows migration/resource review. Never fits or reads Test CSV contents."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import struct
import sys
from pathlib import Path

import psutil

from colab.experiment_launcher import _training
from colab.svm_resumable_store import digest, file_sha, read_json
from colab.svm_run_authorization import _object_sha256, _validate_technical_evidence
from colab.svm_official_readiness import git_checkout
from .checkpoint_migration import blocked, normalized_bytes, scan_checkpoints, verify_sources
from .paths import CONFIG, logical_path

PINS = {"numpy": "2.2.6", "pandas": "2.3.3", "scipy": "1.15.3",
        "scikit-learn": "1.7.2", "joblib": "1.5.3"}
BUDGET_FIELDS = ("max_elapsed_seconds", "max_tree_rss_bytes", "min_start_available_ram_bytes",
                 "min_runtime_available_ram_bytes", "min_free_disk_bytes", "poll_interval_seconds")
LOCAL_MODULES = ("__init__.py", "paths.py", "checkpoint_migration.py", "windows_preflight.py",
                 "local_authorization.py", "local_supervisor.py", "progress_console.py", "windows_job.py", "run_mi_rfe_svm.py")
REQUIRED_TESTS = ("local.test_windows_local", "local.test_local_supervisor",
                  "colab.test_svm_resumable_store", "colab.test_svm_resumable_training",
                  "colab.test_svm_resumable_execution", "colab.test_stage73b_authorization",
                  "colab.test_stage73b_scientific_integrity")


def implementation_hashes(repo: Path) -> dict:
    names = ["local/" + name for name in LOCAL_MODULES]
    names += ["local/test_windows_local.py", "local/test_local_supervisor.py",
              "colab/svm_resumable_training.py", "colab/test_svm_resumable_training.py"]
    return {name: normalized_bytes((repo / name).read_bytes()) for name in names}


def inventory_fingerprint(manifest_sha, validation_sha, summary_sha, inventory) -> str:
    payload = {"manifest_sha256": manifest_sha, "validation_sha256": validation_sha,
               "summary_sha256": summary_sha,
               "inventory_sizes": sorted((logical_path(name), size) for name, size in inventory)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def environment() -> dict:
    versions = {name: metadata.version(name) for name in (*PINS, "psutil")}
    return {"os_name": os.name, "platform": platform.platform(), "machine": platform.machine(),
            "python_major_minor": list(sys.version_info[:2]), "python": sys.version,
            "bits": struct.calcsize("P") * 8, "versions": versions,
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False)}


def check_environment(profile: dict) -> None:
    if profile["os_name"] != "nt" or profile["bits"] != 64 or profile["python_major_minor"] != [3, 13]:
        blocked("Windows x64 Python 3.13 required for this Colab migration profile")
    if any(profile["versions"].get(k) != v for k, v in PINS.items()):
        blocked(f"Scientific packages must match exactly: {PINS}")


def validate_budgets(budgets: dict) -> dict:
    if not isinstance(budgets, dict) or set(budgets) != set(BUDGET_FIELDS):
        blocked("Explicit local resource policy required; no inherited Colab 42-GiB threshold")
    if any(type(budgets[k]) is not int or budgets[k] <= 0 for k in BUDGET_FIELDS):
        blocked("Every local budget must be an explicit positive integer; no automatic defaults")
    if not 1 <= budgets["poll_interval_seconds"] <= 60 or budgets["max_elapsed_seconds"] <= 300:
        blocked("Poll must be 1..60s and session budget must exceed the 300s safety margin")
    return budgets.copy()


def disk_anchor(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def resources(paths, budgets: dict, estimated_peak: int, *, vm=None, disk_usage=None) -> dict:
    vm = vm or psutil.virtual_memory()
    disk_usage = disk_usage or psutil.disk_usage
    disks = {str(disk_anchor(p)): disk_usage(str(disk_anchor(p))).free
             for p in (paths.checkpoint_root, paths.output_root)}
    failed = []
    if vm.available < budgets["min_start_available_ram_bytes"]:
        failed.append("AVAILABLE_RAM_BELOW_LOCAL_START_THRESHOLD")
    if budgets["max_tree_rss_bytes"] + budgets["min_runtime_available_ram_bytes"] > vm.available:
        failed.append("RSS_LIMIT_PLUS_RAM_RESERVE_EXCEEDS_AVAILABLE")
    if estimated_peak > budgets["max_tree_rss_bytes"] or estimated_peak > int(.85 * vm.available):
        failed.append("CONSERVATIVE_SVM_ARRAY_CACHE_SCENARIO_NOT_ADMITTED")
    if any(free < budgets["min_free_disk_bytes"] for free in disks.values()):
        failed.append("LOCAL_DISK_BELOW_APPROVED_RESERVE")
    return {"status": "PASS" if not failed else "RESOURCE_ADMISSION_FAIL", "reasons": failed,
            "ram_total_bytes": vm.total, "ram_available_bytes": vm.available,
            "estimated_array_cache_scenario_bytes": estimated_peak, "disk_free_bytes": disks,
            "unknown_support_vector_peak": True, "budgets": budgets}


def checksum_index(payload: dict) -> dict:
    if payload.get("schema") != "stage73b.ssd.file-sha256.v1":
        blocked("Original Colab SSD checksum manifest schema missing/mismatched")
    indexed = {}
    for row in payload.get("files", []):
        if not isinstance(row, dict) or row.get("area") not in ("windows", "manifest"):
            blocked("Invalid dataset checksum manifest record")
        key = (row["area"], logical_path(row.get("path")))
        if key in indexed or type(row.get("bytes")) is not int or row["bytes"] < 0:
            blocked("Duplicate logical dataset path or invalid size")
        value = row.get("sha256")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            blocked("Malformed source checksum")
        indexed[key] = row
    return indexed


def verify_content(paths, payload: dict, *, roles: tuple[str, ...]) -> dict:
    """Test bytes are hashed ONLY after a final-model Test claim (roles=('test',))."""
    index = checksum_index(payload)
    verified = 0
    for (area, relative), row in index.items():
        if area != "windows" or not relative.endswith("_windows.csv"):
            continue
        parts = relative.split("/")
        if parts[0] != CONFIG or len(parts) < 3 or parts[1] not in roles:
            continue
        path = paths.data_root / relative
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            blocked(f"Dataset content checksum differs: {relative}")
        verified += 1
        if verified % 10 == 0:
            print(f"CONTENT VERIFIED {verified} files roles={roles}", flush=True)
    if not verified:
        blocked(f"No source checksum records for selected roles: {roles}")
    return {"files_content_verified": verified, "roles": list(roles), "test_contents_read": "test" in roles}


def dataset_identity(paths, binding: dict, checksum_payload: dict) -> tuple[dict, object]:
    training = _training(paths.repo)
    sw = training.sw
    segments = sw.load_manifest(paths.manifest_root)
    sw.verify_split_artifacts(paths.manifest_root, segments)
    sw.expected_window_table(segments)
    store = training.WindowStore(paths.data_root, split_root=paths.manifest_root)
    index = checksum_index(checksum_payload)
    # EXACT Colab/Linux fingerprint construction; physical Windows paths never enter it.
    inventory = sorted((f.path.relative_to(paths.data_root).as_posix(), f.path.stat().st_size)
                       for files in store.files.values() for f in files)
    if len(inventory) != 322 or len(set(p for p, _ in inventory)) != 322:
        blocked("Full 322-file window inventory required (not just one selected FW/FS directory)")
    fields = {"manifest_sha256": file_sha(paths.manifest_root / "split_manifest.csv"),
              "validation_sha256": file_sha(paths.manifest_root / "validation.json"),
              "window_summary_sha256": file_sha(paths.data_root / "window_run_summary.json")}
    fields["dataset_fingerprint_sha256"] = inventory_fingerprint(
        fields["manifest_sha256"], fields["validation_sha256"], fields["window_summary_sha256"], inventory)
    scope = "manifest+summary+CSV relative paths/sizes; not full CSV content hash"
    fields["dataset_fingerprint_scope_sha256"] = _object_sha256(scope)
    origin = binding["identity"]
    if any(origin.get(name) != value for name, value in fields.items()):
        blocked("Logical dataset fingerprint or approved metadata differ from origin")
    if binding.get("split_identity") != {k: origin[k] for k in fields}:
        blocked("Origin split identity is inconsistent")
    for relative, size in inventory:
        row = index.get(("windows", relative))
        if row is None or row["bytes"] != size:
            blocked(f"Source content manifest missing/size mismatch: {relative}")
    for area, name, digest_key in (("manifest", "split_manifest.csv", "manifest_sha256"),
                                  ("manifest", "validation.json", "validation_sha256"),
                                  ("windows", "window_run_summary.json", "window_summary_sha256")):
        if index.get((area, name), {}).get("sha256") != fields[digest_key]:
            blocked("Source checksum manifest is not anchored to the same split/window metadata")
    for file in store.files[CONFIG]:
        if file.role in ("train", "val"):
            store.validate_header(file.path)
    return {**fields, "fingerprint_scope": scope, "test_headers_or_rows_read": 0,
            "snapshot": store.snapshot(), "source_content_manifest": "REQUIRED_TRUSTED_COLAB_EXPORT"}, store


def verify_origin_authorization(root, binding, approval_path, technical_path) -> dict:
    saved = read_json(root / "sessions" / "session01" / "authorization.json")
    old = read_json(approval_path)
    if (saved.get("binding_sha256") != digest(binding)
            or saved.get("base", {}).get("identity") != binding["identity"]
            or saved["base"].get("approval_sha256") != file_sha(approval_path)
            or old.get("identity") != binding["identity"]
            or old.get("effective") is not True or old.get("decision") != "APPROVED_STAGE73B"):
        blocked("Origin approval/session authorization provenance differs")
    audit = read_json(root / "sessions" / "session01" / "preflight.json")
    profile = audit.get("identity_evidence", {}).get("environment_profile")
    if (audit.get("identity") != binding["identity"] or not isinstance(profile, dict)
            or _object_sha256(profile) != binding["identity"].get("environment_profile_sha256")):
        blocked("Original Colab preflight/environment profile is absent or unverified")
    if profile.get("selected_scientific_profile") != PINS or profile.get("python_major_minor") != [3, 13]:
        blocked("Original scientific Python profile not supported by this migration")
    technical = _validate_technical_evidence(old, technical_path, binding["identity"])
    return {"approval_sha256": file_sha(approval_path), "technical_sha256": technical["sha256"],
            "origin_environment_profile": profile, "colab_approval_is_local_authorization": False}


def preflight(paths, args, *, hash_train_val=True) -> dict:
    profile = environment()
    check_environment(profile)
    checkout = git_checkout(paths.repo)
    if not checkout.get("clean") or not checkout.get("commit"):
        blocked("Clean identifiable checkout required; commit/push source manually before VM preflight")
    training = _training(paths.repo)
    if Path(training.__file__).resolve() != (paths.repo / "huyen_code" / "ciciomt24_training.py").resolve():
        blocked("Imported scientific module belongs to another checkout; restart Python with the correct PYTHONPATH")
    original_hashes = training.source_hashes(paths.repo / "CODE_ctu13")
    training.source_audit(paths.repo / "CODE_ctu13")
    scan = scan_checkpoints(paths.checkpoint_root, expected_names=training.FEATURE_NAMES,
                            expected_contract=training.official_hyperparameters("mi_rfe_svm"))
    if _object_sha256({name: value.lower() for name, value in original_hashes.items()}) != scan["binding"]["identity"].get("original_script_hashes_sha256"):
        blocked("Origin CTU-13 original-script identity differs")
    bridge = verify_sources(paths.repo, scan["binding"])
    origin = verify_origin_authorization(paths.checkpoint_root, scan["binding"],
                                         args.origin_approval, args.technical_evidence)
    checksums = read_json(args.dataset_checksums)
    dataset, store = dataset_identity(paths, scan["binding"], checksums)
    content = verify_content(paths, checksums, roles=("train", "val")) if hash_train_val else None
    unit = read_json(args.unit_test_evidence)
    hashes = implementation_hashes(paths.repo)
    if (unit.get("passed") is not True or unit.get("returncode") != 0
            or unit.get("git_commit") != checkout["commit"] or unit.get("source_hashes") != hashes
            or not set(REQUIRED_TESTS) <= set(unit.get("modules", []))):
        blocked("Fresh local/legacy unit test evidence for this exact commit/source required")
    budgets = validate_budgets(read_json(args.budget_file))
    plan = training.resource_plan(store)[CONFIG]
    nfit = 897932
    # MI/RFE are already completed: retain conservative SVM/array/cache scenarios.
    estimated = plan["raw_float32_all_splits_bytes"] + nfit * 100 * (4 + 8 + 8) + 10000 * 1024**2 + 512 * 1024**2
    admission = resources(paths, budgets, estimated)
    return {"status": "PASS_LOCAL_PREFLIGHT" if admission["status"] == "PASS" else "RESOURCE_ADMISSION_FAIL",
            "full_training_authorized": False, "paths": paths.record(), "git": checkout,
            "environment": profile, "source_hashes": hashes, "source_bridge": bridge,
            "scan": scan, "origin_authorization": origin, "dataset": dataset,
            "content_verification": content, "resources": admission,
            "input_evidence_sha256": {name: file_sha(getattr(args, name)) for name in (
                "origin_approval", "technical_evidence", "dataset_checksums", "unit_test_evidence", "budget_file")},
            "model_fit_calls": 0, "test_csv_contents_read": 0,
            "windows_vs_linux_numerical_bitwise_equivalence": "NOT_ESTABLISHED"}
