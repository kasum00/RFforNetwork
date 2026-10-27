"""Drive-only CICFLOW notebook V2 orchestration; no scientific algorithm changes.

Importing this module uses the standard library only. Dataset checks, tests and
fits run only when the notebook user calls the corresponding worker action.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import unittest
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
BRANCHES = ("rf", "svm")
SCIENTIFIC_STATUS = "PROVISIONAL_RESEARCH"
REFERENCE_STATUS = "PROVISIONAL_REFERENCE_UNVERIFIED"
FORMAT = "cicflow-colab-v2.1"
RESULT_MARKER = "__CICFLOW_V2_RESULT__"
REQUIRED = ("colab/colab_pipeline.py", "colab/colab_pipeline_v2.py",
            "stage4_mi/mi_selection.py", "stage5_rfe/rfe_selection.py")
PACKAGES = ("numpy", "scipy", "scikit-learn", "pyarrow", "joblib", "psutil")
EXCLUDED = {"runs", "outputs", "logs", "reports", "__pycache__", ".git", ".venv"}


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    """CICFLOW equivalent of the reference launcher's explicit experiment spec."""

    experiment: str
    config: str
    run_name: str

    def validate(self):
        if self.experiment not in ("mi_rfe_rf", "mi_rfe_svm") or self.config not in CONFIGS:
            raise ContractError("Unknown CICFLOW experiment/configuration")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name):
            raise ContractError("Run name must use letters, digits, underscore and hyphen")

    @property
    def branch(self):
        self.validate()
        return "rf" if self.experiment == "mi_rfe_rf" else "svm"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_once(path, value):
    """Publish complete JSON once; reject collisions, including partial attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Preserve existing artifact: {path}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Single-writer experiment; never replace the named final artifact.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(temporary.read_text(encoding="utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.unlink()


def save_report(root, name, value):
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    path = Path(root) / "reports" / f"{name}_{token}.json"
    write_json_once(path, value)
    return str(path)


def source_snapshot(source):
    source = Path(source).resolve()
    files = {}
    for area in ("colab", "stage4_mi", "stage5_rfe"):
        for path in sorted((source / area).rglob("*")):
            relative = path.relative_to(source)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if path.is_file() and path.suffix in {".py", ".json"}:
                files[relative.as_posix()] = file_hash(path)
    missing = [name for name in REQUIRED if name not in files]
    if missing:
        raise ContractError(f"SOURCE_MISSING: {missing}; upload these files to {source}")
    return {"files": files, "sha256": digest(files)}


def activate_source(source):
    """Resolve competing 'colab' packages without touching google.colab."""
    source = Path(source).resolve()
    sys.path[:] = [str(source)] + [p for p in sys.path if p != str(source)]
    for package in ("colab", "stage4_mi", "stage5_rfe"):
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]
        folder = source / package
        if not folder.is_dir():
            raise ContractError(f"Missing source package: {folder}")
        # Explicit namespace works with or without an uploaded __init__.py.
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        module.__package__ = package
        sys.modules[package] = module
    importlib.invalidate_caches()
    loaded = importlib.import_module("colab.colab_pipeline")
    expected = (source / "colab/colab_pipeline.py").resolve()
    if Path(loaded.__file__).resolve() != expected:
        raise ContractError(f"Wrong pipeline import: expected {expected}; loaded {loaded.__file__}")
    return loaded


def science_modules(source):
    activate_source(source)
    mi = importlib.import_module("stage4_mi.mi_selection")
    rfe = importlib.import_module("stage5_rfe.rfe_selection")
    for module, relative in ((mi, "stage4_mi/mi_selection.py"), (rfe, "stage5_rfe/rfe_selection.py")):
        if Path(module.__file__).resolve() != (Path(source) / relative).resolve():
            raise ContractError(f"Wrong scientific module: {module.__file__}")
    return mi, rfe


def initialize_run(drive_root, source_root, dataset_root, mode, existing="", session_root=""):
    if mode not in {"dry_run", "full", "resume", "verify"}:
        raise ContractError(f"Invalid mode: {mode}")
    drive, source, dataset = map(lambda p: Path(p).resolve(), (drive_root, source_root, dataset_root))
    for label, path in (("Drive", drive), ("Source", source), ("Dataset", dataset)):
        if not path.is_dir():
            raise ContractError(f"{label} missing: {path}")
    if source != drive / "code_CICFLOW" or dataset != drive / "CICFLOWMETER_FEATURE_CSV":
        raise ContractError("Use the supplied MyDrive/code_CICFLOW and MyDrive/CICFLOWMETER_FEATURE_CSV layout")
    base = drive / "COMPACT_MI_RFE_FRESH_RUNS"
    if mode in {"resume", "verify"}:
        if not existing:
            raise ContractError("EXISTING_RESULT_ROOT is required; resume/verify never generate a new Run ID")
        root = Path(existing).resolve()
    elif session_root:
        root = Path(session_root).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        root = base / ("RUN_MI_" + stamp + "_" + uuid.uuid4().hex[:6])
        root.mkdir(parents=True, exist_ok=False)
        run_ids = {config: root.name + "_" + config for config in CONFIGS}
        write_json_once(root / "RUN_INFO.json", {
            "format": FORMAT, "experiment_id": root.name, "created_at": utc(),
            "source_root": str(source), "dataset_root": str(dataset), "run_ids": run_ids,
            "scientific_status": SCIENTIFIC_STATUS, "reference_status": REFERENCE_STATUS,
            "prior_artifacts_reused": False,
        })
    if root.parent != base:
        raise ContractError(f"Result root must be a direct child of {base}")
    if not (root / "RUN_INFO.json").is_file():
        raise ContractError("UNSUPPORTED: missing V2 RUN_INFO; no new ID will be generated")
    info = read_json(root / "RUN_INFO.json")
    if info.get("format") != FORMAT or info.get("experiment_id") != root.name:
        raise ContractError("UNSUPPORTED: this experiment is not a V2 run; preserve it and start a fresh V2 experiment")
    expected_ids = {config: root.name + "_" + config for config in CONFIGS}
    if info.get("run_ids") != expected_ids:
        raise ContractError("Stored configuration Run IDs are missing or inconsistent; no replacement IDs are generated")
    if info["source_root"] != str(source) or info["dataset_root"] != str(dataset):
        raise ContractError("Stored source/dataset paths differ from the selected experiment")
    for area in ("stage4_mi/runs", "stage5_rfe/runs", "logs", "reports", "task_records"):
        (root / area).mkdir(parents=True, exist_ok=True)
    return {"root": str(root), "info": info}


def dependency_report(source):
    """Fresh-worker import evidence, not a fabricated compatibility profile."""
    versions, errors = {}, []
    imports = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn",
               "pyarrow": "pyarrow", "joblib": "joblib", "psutil": "psutil"}
    for distribution, module_name in imports.items():
        try:
            module = importlib.import_module(module_name)
            versions[distribution] = metadata.version(distribution)
            if module is None:
                raise ImportError(module_name)
        except Exception as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
    result = {"status": "PASS" if not errors else "FAIL", "python": platform.python_version(),
              "versions": versions, "errors": errors, "pins_applied": False}
    if not errors:
        mi, rfe = science_modules(source)
        result["mi_parameters"] = mi._run_parameters()
        result["rfe_contracts"] = {b: rfe.branch_contract(b) for b in BRANCHES}
        result["resolved_rfe_estimators"] = {b: rfe.make_estimator(b).get_params() for b in BRANCHES}
    return result


def discover_test_files(source):
    result = {}
    for area in ("colab", "stage4_mi", "stage5_rfe"):
        found = set()
        for path in (Path(source) / area).rglob("test_*.py"):
            if not any(part in EXCLUDED for part in path.relative_to(source).parts):
                found.add(path.resolve())
        result[area] = sorted(found)
    return result


def tests_status(tests_run, failures=0, errors=0, skipped=0, expected_failures=0, unexpected_successes=0):
    if errors or failures or unexpected_successes or expected_failures:
        return "FAIL"
    if not tests_run:
        return "NO_TESTS_FOUND"
    if skipped:
        return "SKIPPED"
    return "PASS"


def run_test_files(source):
    """Load actual files once. Does not require tests/ to be an importable package."""
    activate_source(source)
    found = discover_test_files(source)
    rows = {}
    seen_ids = set()
    for paths in found.values():
        for path in paths:
            if str(path.parent) not in sys.path:
                sys.path.insert(1, str(path.parent))

    def leaves(suite):
        for case in suite:
            if isinstance(case, unittest.TestSuite):
                yield from leaves(case)
            else:
                yield case

    for area, paths in found.items():
        suite = unittest.TestSuite()
        import_errors = []
        for path in paths:
            name = "_cicflow_test_" + hashlib.sha256(str(path).encode()).hexdigest()
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                for case in leaves(unittest.defaultTestLoader.loadTestsFromModule(module)):
                    if case.id() not in seen_ids:
                        seen_ids.add(case.id())
                        suite.addTest(case)
            except Exception:
                import_errors.append({"file": str(path), "error": traceback.format_exc()})
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        row = {"files": [str(p) for p in paths], "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors) + len(import_errors),
               "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
               "unexpected_successes": len(result.unexpectedSuccesses), "import_errors": import_errors}
        row["status"] = tests_status(**{k: row[k] for k in (
            "tests_run", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")})
        rows[area] = row
    return {"status": "PASS" if all(r["status"] == "PASS" for r in rows.values()) else "BLOCKED",
            "areas": rows, "source_snapshot": source_snapshot(source)}


def dataset_structure(dataset):
    result = {}
    candidates = ("train_windows.parquet", "train.parquet", "train_features.parquet")
    for config in CONFIGS:
        folder = Path(dataset) / config
        found = [name for name in candidates if (folder / name).is_file()]
        result[config] = {"status": "PASS" if folder.is_dir() and len(found) == 1 else "BLOCKED",
                          "directory": str(folder), "train_candidates": found,
                          "reason": "" if len(found) == 1 else "Require exactly one supported Train Parquet"}
    return {"status": "PASS" if all(r["status"] == "PASS" for r in result.values()) else "BLOCKED",
            "configs": result, "validation_test_values_read": False}


def metadata_audit(source, dataset, evidence_files=None):
    mi, _ = science_modules(source)
    import pyarrow as pa
    results = {}
    for config in CONFIGS:
        try:
            folder = Path(dataset) / config
            names = mi.load_feature_schema(folder)
            train = mi.resolve_train_path(folder)
            footer = mi.inspect_train_footer(train, names)
            arrow_schema = mi._parquet_file(train).schema_arrow
            invalid = [f.name for f in list(arrow_schema)[:-1]
                       if not (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))]
            errors = []
            if any(name.lower() in {mi.LABEL_COLUMN.lower(), "label", "target"} for name in names):
                errors.append("Label/target column cannot appear in the 380 feature names")
            if invalid:
                errors.append(f"Non-numeric feature columns: {invalid}")
            if footer["rows"] != mi.CONFIG_EXPECTED_ROWS[config]:
                errors.append(f"Wrong Train count: {footer['rows']} != {mi.CONFIG_EXPECTED_ROWS[config]}")
            sidecars = []
            selected = [folder / name for name in ("feature_schema.json", "window_feature_schema.json",
                        "stage3_integrity_report.json", "integrity_report.json") if (folder / name).is_file()]
            for extra in (evidence_files or {}).get(config, []):
                path = Path(extra).resolve()
                if path.suffix.lower() != ".json":
                    raise ContractError("Evidence must be a JSON report, never a split Parquet")
                selected.append(path)
            for path in sorted(set(selected)):
                value = read_json(path)
                sidecars.append({"path": str(path), "sha256": file_hash(path), "evidence": value})
                # Explicit technical failures must never be ignored.
                if isinstance(value, dict) and value.get("status") in {"FAIL", "BLOCKED"}:
                    errors.append(f"Evidence reports {value['status']}: {path}")
            identity = mi.source_identity(config, folder, names, footer)
            results[config] = {"status": "PASS" if not errors else "BLOCKED", "errors": errors,
                "feature_names": names, "train_footer": footer, "source_identity": identity,
                "evidence_files": sidecars,
                "fingerprint_scope": "path+size+mtime+footer+sidecar hashes; not a full Train content hash",
                "train_values_status": "NOT_READ_METADATA_ONLY", "split_provenance_status": "NOT_VERIFIED",
                "leakage_claim": False, "validation_test_values_read": False}
        except Exception as exc:
            results[config] = {"status": "BLOCKED", "errors": [f"{type(exc).__name__}: {exc}"],
                               "validation_test_values_read": False}
    return {"status": "PASS" if all(r["status"] == "PASS" for r in results.values()) else "BLOCKED",
            "configs": results, "scientific_status": SCIENTIFIC_STATUS, "leakage_claim": False}


def resources(source, dataset, root, config):
    mi, rfe = science_modules(source)
    folder = Path(dataset) / config
    rows = mi.CONFIG_EXPECTED_ROWS[config]
    mi_snapshot = mi.collect_resource_snapshot(folder)
    usage = shutil.disk_usage(root)
    mi_snapshot["disk"] = {"path": str(root), "free_bytes": usage.free, "total_bytes": usage.total}
    rf_snapshot = rfe.collect_resource_snapshot(folder, Path(root) / "stage5_rfe/runs")
    rf_snapshot["disk"] = {"path": str(root), "free_bytes": usage.free, "total_bytes": usage.total}
    return {"mi": mi.resource_gate(mi_snapshot, rows), "rfe": rfe.resource_gate(rf_snapshot, rows),
            "mi_snapshot": mi_snapshot, "rfe_snapshot": rf_snapshot,
            "policy": "Existing source RAM estimates and 2 GiB disk threshold, measured at output root; estimate is not a fit guarantee"}


def ensure_selection(configs, branches):
    if not configs or len(set(configs)) != len(configs) or any(c not in CONFIGS for c in configs):
        raise ContractError("Select one or more distinct supported configurations")
    if not branches or len(set(branches)) != len(branches) or any(b not in BRANCHES for b in branches):
        raise ContractError("Select rf, svm, or both; XGBoost is BLOCKED")


def binding_identity(source, dataset, audit, dependencies, git_commit):
    return {"source_snapshot": source_snapshot(source), "dataset_root": str(Path(dataset).resolve()),
            "dataset_identities": {c: r.get("source_identity") for c, r in audit["configs"].items()},
            "evidence_hashes": {c: [{"path": f["path"], "sha256": f["sha256"]}
                for f in r.get("evidence_files", [])] for c, r in audit["configs"].items()},
            "python": dependencies["python"], "versions": dependencies["versions"],
            "mi_parameters": dependencies["mi_parameters"],
            "rfe_contracts": dependencies["rfe_contracts"],
            "resolved_rfe_estimators": dependencies["resolved_rfe_estimators"], "git_commit": git_commit}


def preflight(payload):
    source, dataset, root = (Path(payload[k]) for k in ("source", "dataset", "root"))
    ensure_selection(payload["configs"], payload["branches"])
    dep = dependency_report(source)
    audit = metadata_audit(source, dataset, payload.get("evidence_files")) if dep["status"] == "PASS" else {"status": "BLOCKED", "configs": {}}
    tests = payload.get("test_report", {})
    checks = {"dependencies": dep["status"] == "PASS", "metadata": audit["status"] == "PASS",
              "unit_tests": tests.get("status") == "PASS",
              "test_source_matches": tests.get("source_snapshot") == source_snapshot(source)}
    source_binding = root / "SOURCE_BINDING.json"
    checks["source_binding_current"] = source_binding.is_file() and read_json(source_binding) == source_snapshot(source)
    errors = []
    if not checks["unit_tests"]:
        errors.append("Unit tests must report PASS for every area; SKIPPED/NO_TESTS_FOUND cannot authorize training")
    resource_results = {}
    for config in payload["configs"]:
        try:
            resource_results[config] = resources(source, dataset, root, config)
            checks["resources_" + config] = all(resource_results[config][s]["status"] == "PASS" for s in ("mi", "rfe"))
        except Exception as exc:
            checks["resources_" + config] = False
            errors.append(f"{config}: {exc}")
    probe = root / "reports" / (".write_probe_" + uuid.uuid4().hex)
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("write probe")
        probe.unlink()
        checks["output_writable"] = True
    except OSError as exc:
        checks["output_writable"] = False
        errors.append(str(exc))
    if payload["mode"] in {"full", "resume"}:
        checks["provisional_confirmation"] = payload.get("confirm_provisional") is True
        checks["exact_confirmation"] = payload.get("confirmation") == "RUN PROVISIONAL " + root.name
    locked_path = root / "EXECUTION_BINDING.json"
    current = None
    if dep["status"] == "PASS" and audit["status"] == "PASS":
        current = binding_identity(source, dataset, audit, dep, payload.get("git_commit", "NOT_CAPTURED"))
        if locked_path.exists():
            checks["locked_identity"] = read_json(locked_path) == current
        elif payload["mode"] in {"resume", "verify"}:
            checks["locked_identity"] = False
            errors.append("UNSUPPORTED: selected run has no V2 execution binding")
        else:
            checks["locked_identity"] = True
    if payload["mode"] == "full":
        info = read_json(root / "RUN_INFO.json")
        occupied = []
        for c in payload["configs"]:
            rid = info["run_ids"][c]
            mi_dir = root / "stage4_mi/runs" / c / rid
            if mi_dir.exists():
                occupied.append(str(mi_dir))
            occupied.extend(str(root / "stage5_rfe/runs" / c / rid / b) for b in payload["branches"]
                            if (root / "stage5_rfe/runs" / c / rid / b).exists())
        checks["fresh_outputs"] = not occupied
        if occupied:
            errors.append("Outputs exist; select resume/verify for this experiment: " + ", ".join(occupied))
    passed = all(checks.values())
    if passed and current is not None and payload["mode"] in {"full", "resume"} and not locked_path.exists():
        write_json_once(locked_path, current)
    errors.extend("Check failed: " + k for k, v in checks.items() if not v)
    request = {key: payload.get(key) for key in ("source", "dataset", "root", "mode", "configs", "branches", "git_commit")}
    return {"status": "PASS" if passed else "BLOCKED", "checks": checks, "errors": errors,
            "request": request,
            "resources": resource_results, "audit": audit,
            "scientific_status": SCIENTIFIC_STATUS, "reference_status": REFERENCE_STATUS,
            "reference_confirmation_recorded": payload.get("reference_confirmed", False),
            "official_reproduction": False, "validation_test_values_read": False}


def require_launch_context(payload, preflight_result, *, dataset_ready, unit_test_evidence):
    """Reference Cell 10 prerequisites, rechecked for the current notebook selection."""
    if payload.get("mode") not in {"full", "resume"}:
        raise ContractError("Fitting requires full/resume mode")
    if not dataset_ready:
        raise ContractError("Dataset validation has not passed; run Cells 4 and 6 first")
    if not unit_test_evidence.get("passed"):
        raise ContractError("Unit tests have not passed; run Cell 5 first")
    if preflight_result.get("status") != "PASS":
        raise ContractError("Preflight has not passed; run Cell 9 first")
    keys = ("source", "dataset", "root", "mode", "configs", "branches", "git_commit")
    current = {key: payload.get(key) for key in keys}
    if preflight_result.get("request") != current:
        raise ContractError("Experiment/mode/selection changed after preflight; rerun Cell 9")
    if payload.get("confirm_provisional") is not True:
        raise ContractError("Explicit provisional research confirmation is required")
    required = "RUN PROVISIONAL " + Path(payload["root"]).name
    if payload.get("confirmation") != required:
        raise ContractError(f"Incorrect confirmation. Expected: {required}")
    if unit_test_evidence.get("source_sha256") != source_snapshot(payload["source"])["sha256"]:
        raise ContractError("Source changed after the unit tests; rerun Cell 5")


def verify_artifacts(source, dataset, root, config, stage, branch=None):
    """Verify scores, order, binding and completion evidence; never read row values."""
    mi, rfe = science_modules(source)
    root = Path(root)
    info = read_json(root / "RUN_INFO.json")
    rid = info["run_ids"][config]
    mi_dir = root / "stage4_mi/runs" / config / rid
    errors = []
    try:
        binding = rfe.load_stage4_binding(dataset_root=Path(dataset), config=config, mi_run_dir=mi_dir)
        rank = read_json(mi_dir / "mi_ranking.json")["ranking"]
        names = binding["feature_names"]
        if len(rank) != 380 or {r["feature_index"] for r in rank} != set(range(380)):
            raise ContractError("MI ranking does not cover each of 380 original indices once")
        scores = {r["feature_index"]: r["mi_score"] for r in rank}
        if rank != mi.rank_scores(scores, names):
            raise ContractError("MI score/name/index/ranking/tie semantics mismatch")
        manifest = read_json(mi_dir / "mi_manifest.json")
        if manifest.get("status") != "COMPLETE" or manifest.get("parameters") != mi._run_parameters():
            raise ContractError("MI manifest status or scientific parameters mismatch")
        if manifest.get("train_rows_observed") != binding["train_footer"]["rows"]:
            raise ContractError("MI observed Train row count mismatch")
        if manifest.get("source_identity") != binding["source_identity"]:
            raise ContractError("MI stored source identity contents mismatch")
        with (mi_dir / "mi_scores.csv").open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        converted = [{"rank": int(r["rank"]), "feature_index": int(r["feature_index"]),
                      "feature_name": r["feature_name"], "mi_score": float(r["mi_score"])} for r in csv_rows]
        if converted != rank:
            raise ContractError("MI CSV and JSON scores/ranking disagree")
        mi_progress = read_json(mi_dir / "mi_progress.json")
        validate_checkpoint_state("mi", mi_progress)
        if len(mi_progress["completed_indices"]) != 380:
            raise ContractError("MI completion checkpoint is not at 380 features")
        if {int(k): float(v) for k, v in mi_progress["scores"].items()} != scores:
            raise ContractError("MI checkpoint and final scores disagree")
        run_dir = mi_dir
        required = ["MI_COMPLETE.json", "mi_manifest.json", "mi_ranking.json", "top100_features.json",
                    "mi_scores.csv", "mi_progress.json", "resource_usage.json", "mi_report.md"]
        if stage == "rfe":
            if branch not in BRANCHES:
                raise ContractError("Invalid RFE branch")
            run_dir = root / "stage5_rfe/runs" / config / rid / branch
            result = rfe.verify_run(run_dir)
            if result["status"] != "PASS":
                raise ContractError(str(result))
            value = read_json(run_dir / "rfe_ranking.json")
            raw, support = value["raw_ranking"], value["support"]
            if sorted(raw) != list(range(1, 101)) or support.count(True) != 1:
                raise ContractError("Invalid 100-to-1 RFE ranking or support")
            if any(bool(s) != (r == 1) for s, r in zip(support, raw)):
                raise ContractError("RFE support/rank mismatch")
            if value["ranking"] != rfe.ranking_records(binding["top_features"], raw, support):
                raise ContractError("RFE ordered feature mapping mismatch")
            manifest = read_json(run_dir / "rfe_manifest.json")
            if manifest.get("binding") != binding or manifest.get("completed_steps") != 99:
                raise ContractError("RFE bound MI contents or final step count mismatch")
            if manifest.get("train_rows_observed") != binding["train_footer"]["rows"] or manifest.get("selected_feature_count") != 100:
                raise ContractError("RFE observed input dimensions mismatch")
            expected = rfe._make_run_identity(config=config, branch=branch, run_id=rid,
                       binding=binding, git_commit=manifest["run_identity"]["git_commit"])
            if manifest["run_identity"] != expected:
                raise ContractError("RFE source/MI/config/branch identity mismatch")
            marker = read_json(run_dir / "COMPLETE.json")
            if marker.get("status") != "COMPLETE":
                raise ContractError("RFE completion marker status is not COMPLETE")
            if marker.get("run_identity_digest") != mi.stable_digest(expected):
                raise ContractError("RFE COMPLETE identity digest mismatch")
            if marker.get("ranking_digest") != manifest.get("ranking_digest") or marker.get("completed_steps") != 99:
                raise ContractError("RFE COMPLETE ranking or step count mismatch")
            with (run_dir / "rfe_ranking.csv").open(encoding="utf-8", newline="") as handle:
                exported = list(csv.DictReader(handle))
            expected_csv = [{key: str(field_value) for key, field_value in r.items()} for r in value["ranking"]]
            if exported != expected_csv:
                raise ContractError("RFE CSV and JSON ranking disagree")
            rfe_progress = read_json(run_dir / "rfe_progress.json")
            validate_checkpoint_state("rfe", rfe_progress)
            if rfe_progress["ranking"] != raw or rfe_progress["support"] != support or rfe_progress["completed_steps"] != 99:
                raise ContractError("RFE checkpoint and final ranking disagree")
            required = ["COMPLETE.json", "rfe_manifest.json", "rfe_ranking.json", "rfe_ranking.csv",
                        "rfe_progress.json", "resource_usage.json", "rfe_report.md"]
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise ContractError(f"Missing outputs: {missing}")
        hashes = {name: file_hash(run_dir / name) for name in required}
        return {"status": "PASS", "config": config, "stage": stage, "branch": branch,
                "run_dir": str(run_dir), "hashes": hashes, "official_reproduction": False}
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {"status": "FAIL", "config": config, "stage": stage, "branch": branch, "errors": errors}


def validate_checkpoint_state(stage, value):
    """Conservative structural check before invoking the existing resume API."""
    if stage == "mi":
        indices = value.get("completed_indices", [])
        scores = value.get("scores", {})
        if (len(set(indices)) != len(indices) or any(type(i) is not int or i not in range(380) for i in indices)
                or set(map(str, indices)) != set(scores)
                or not all(math.isfinite(float(s)) and float(s) >= 0 for s in scores.values())):
            raise ContractError("UNSUPPORTED: inconsistent MI checkpoint indices/scores")
    else:
        support, ranking = value.get("support", []), value.get("ranking", [])
        steps = value.get("completed_steps")
        if len(support) != 100 or len(ranking) != 100 or type(steps) is not int or not 0 <= steps <= 99:
            raise ContractError("UNSUPPORTED: RFE checkpoint dimensions/step count")
        if any(type(v) is not bool for v in support) or sum(support) != 100 - steps:
            raise ContractError("UNSUPPORTED: RFE support/step count mismatch")
        if any(type(v) is not int or v < 1 for v in ranking):
            raise ContractError("UNSUPPORTED: invalid RFE rank values")
        if any(r != 1 for s, r in zip(support, ranking) if s):
            raise ContractError("UNSUPPORTED: active RFE features must have rank 1")
        if sorted(r for s, r in zip(support, ranking) if not s) != list(range(2, steps + 2)):
            raise ContractError("UNSUPPORTED: eliminated RFE ranks do not form step=1 history")
        if value.get("active_indices") != [i for i, s in enumerate(support) if s]:
            raise ContractError("UNSUPPORTED: RFE active order mismatch")


def resume_plan(run_dir, stage):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return "START_UNSTARTED_TASK"
    marker = "MI_COMPLETE.json" if stage == "mi" else "COMPLETE.json"
    if (run_dir / marker).is_file():
        return "VERIFY_COMPLETED_TASK"
    if any(p.name.endswith((".tmp", ".writing")) for p in run_dir.iterdir()):
        raise ContractError("UNSUPPORTED: partially written files need review; no automatic cleanup")
    final_files = ("mi_ranking.json", "top100_features.json", "mi_manifest.json", "mi_scores.csv", "mi_report.md") if stage == "mi" else (
        "rfe_ranking.json", "rfe_ranking.csv", "rfe_report.md")
    if any((run_dir / f).exists() for f in final_files):
        raise ContractError("UNSUPPORTED: interrupted final publication; source resume cannot safely reconcile existing final files")
    progress_path = run_dir / ("mi_progress.json" if stage == "mi" else "rfe_progress.json")
    if not (run_dir / "RUN_INCOMPLETE.json").is_file() or not progress_path.is_file():
        raise ContractError("UNSUPPORTED: incomplete run has no usable source checkpoint")
    validate_checkpoint_state(stage, read_json(progress_path))
    return "RESUME_CHECKPOINT"


def assert_execution_binding(payload):
    root = Path(payload["root"])
    locked = root / "EXECUTION_BINDING.json"
    if not locked.is_file():
        raise ContractError("Cell 9 has not locked a successful execution preflight")
    if locked.with_name(locked.name + ".writing").exists():
        raise ContractError("UNSUPPORTED: partially published execution binding needs review")
    tests = payload.get("test_report", {})
    if tests.get("status") != "PASS" or tests.get("source_snapshot") != source_snapshot(payload["source"]):
        raise ContractError("Unit-test evidence is missing or stale")
    dep = dependency_report(payload["source"])
    audit = metadata_audit(payload["source"], payload["dataset"], payload.get("evidence_files"))
    if dep["status"] != "PASS" or audit["status"] != "PASS":
        raise ContractError("Dependencies or dataset metadata changed after preflight")
    identity = binding_identity(payload["source"], payload["dataset"], audit, dep,
                                payload.get("git_commit", "NOT_CAPTURED"))
    if read_json(locked) != identity:
        raise ContractError("Source/library/Train/scientific identity differs from locked experiment")
    if payload.get("confirm_provisional") is not True or payload.get("confirmation") != "RUN PROVISIONAL " + root.name:
        raise ContractError("Explicit provisional confirmation is required")


def execute_task(payload, stage):
    root, source, dataset = (Path(payload[k]) for k in ("root", "source", "dataset"))
    config, branch = payload["config"], payload.get("branch")
    if config not in CONFIGS or (stage == "rfe" and branch not in BRANCHES):
        raise ContractError("Unknown configuration/branch")
    if payload["mode"] not in {"full", "resume"}:
        raise ContractError("Fitting is allowed only in full/resume mode")
    assert_execution_binding(payload)
    mi, rfe = science_modules(source)
    info = read_json(root / "RUN_INFO.json")
    rid = info["run_ids"][config]
    run_dir = root / "stage4_mi/runs" / config / rid if stage == "mi" else root / "stage5_rfe/runs" / config / rid / branch
    record_dir = root / "task_records" / config / ("mi" if stage == "mi" else branch)
    plan = resume_plan(run_dir, stage) if payload["mode"] == "resume" else "FRESH_TASK"
    if payload["mode"] == "full" and run_dir.exists():
        raise ContractError(f"Output exists: {run_dir}; full never overwrites or reuses it")
    intent = record_dir / "START.json"
    if plan == "VERIFY_COMPLETED_TASK":
        result = verify_artifacts(source, dataset, root, config, stage, branch)
        if not intent.is_file() or read_json(intent).get("run_id") != rid:
            raise ContractError("Completed task lacks V2 fresh-run provenance")
        if result["status"] != "PASS":
            return result
        seal = record_dir / "COMPLETE.json"
        if seal.is_file() and read_json(seal).get("hashes") != result["hashes"]:
            raise ContractError("Completed artifacts differ from the saved V2 seal")
        if not seal.exists():
            write_json_once(seal, {**result, "recovered_completed_publication_at": utc()})
        return {**result, "operation": "VERIFIED_COMPLETED_SAME_EXPERIMENT", "fit_called": False}
    if intent.exists():
        previous = read_json(intent)
        if previous["run_id"] != rid or previous["binding_digest"] != digest(read_json(root / "EXECUTION_BINDING.json")):
            raise ContractError("Task intent identity mismatch")
    else:
        if plan == "RESUME_CHECKPOINT":
            raise ContractError("UNSUPPORTED: checkpoint lacks V2 task provenance")
        write_json_once(intent, {"created_at": utc(), "run_id": rid, "config": config, "branch": branch,
                                "binding_digest": digest(read_json(root / "EXECUTION_BINDING.json"))})
    if plan == "RESUME_CHECKPOINT":
        progress = read_json(run_dir / ("mi_progress.json" if stage == "mi" else "rfe_progress.json"))
        if stage == "mi":
            locked = read_json(root / "EXECUTION_BINDING.json")["dataset_identities"][config]
            mi._validate_resume(run_dir, locked, mi._run_parameters())
            if progress.get("source_identity_digest") != locked["identity_digest"]:
                raise ContractError("MI progress identity mismatch")
        else:
            binding = rfe.load_stage4_binding(dataset_root=dataset, config=config,
                        mi_run_dir=root / "stage4_mi/runs" / config / rid)
            identity = rfe._make_run_identity(config=config, branch=branch, run_id=rid,
                       binding=binding, git_commit=payload.get("git_commit", "NOT_CAPTURED"))
            rfe._validate_resume(run_dir, identity)
    guard = resources(source, dataset, root, config)
    key = "mi" if stage == "mi" else "rfe"
    if guard[key]["status"] != "PASS":
        return {"status": "BLOCKED", "config": config, "branch": branch, "resources": guard}
    save_report(root, f"resources_{config}_{key}_{branch or 'mi'}", guard)
    print(f"{stage.upper()} {config}/{branch or 'mi'}: {plan}; output={run_dir}", flush=True)
    if stage == "mi":
        mi.run_mi(dataset_root=dataset, config=config, runs_root=root / "stage4_mi/runs",
                  run_id=rid, resume=plan == "RESUME_CHECKPOINT")
    else:
        upstream = verify_artifacts(source, dataset, root, config, "mi")
        if upstream["status"] != "PASS":
            raise ContractError(f"Stage 4 of this new experiment is not verified: {upstream}")
        mi_seal = root / "task_records" / config / "mi" / "COMPLETE.json"
        if not mi_seal.is_file() or read_json(mi_seal).get("hashes") != upstream["hashes"]:
            raise ContractError("RFE requires the sealed new MI from this experiment")
        rfe.run_rfe(dataset_root=dataset, mi_runs_root=root / "stage4_mi/runs",
                    stage5_runs_root=root / "stage5_rfe/runs", config=config, branch=branch,
                    run_id=rid, git_commit=payload.get("git_commit", "NOT_CAPTURED"),
                    resume=plan == "RESUME_CHECKPOINT")
    verified = verify_artifacts(source, dataset, root, config, stage, branch)
    if verified["status"] != "PASS":
        return verified
    write_json_once(record_dir / "COMPLETE.json", {**verified, "verified_at": utc()})
    return {**verified, "operation": plan, "scientific_status": SCIENTIFIC_STATUS}


def worker_action(action, payload):
    if action == "dependencies":
        return dependency_report(payload["source"])
    if action == "tests":
        return run_test_files(payload["source"])
    if action == "metadata":
        return metadata_audit(payload["source"], payload["dataset"], payload.get("evidence_files"))
    if action == "preflight":
        return preflight(payload)
    if action in {"mi", "rfe"}:
        return execute_task(payload, action)
    if action in {"verify_mi", "verify_rfe"}:
        stage = "mi" if action == "verify_mi" else "rfe"
        result = verify_artifacts(payload["source"], payload["dataset"], payload["root"], payload["config"], stage, payload.get("branch"))
        if result["status"] == "PASS":
            part = "mi" if stage == "mi" else payload["branch"]
            seal = Path(payload["root"]) / "task_records" / payload["config"] / part / "COMPLETE.json"
            if not seal.is_file() or read_json(seal).get("hashes") != result["hashes"]:
                result = {**result, "status": "FAIL", "errors": ["V2 task seal missing or mismatched; resume may recover completed publication"]}
        return result
    raise ContractError(f"Unknown action: {action}")


def redact(text):
    text = re.sub(r"(?i)(Bearer\s+)\S+", r"\1[REDACTED]", text)
    return re.sub(r"(?i)((?:password|token|secret|api[_-]?key)\s*[=:]\s*)[^\s,]+", r"\1[REDACTED]", text)


def _tail_checkpoint(root, info, config, stage, branch):
    if not config:
        return None
    rid = info["run_ids"][config]
    path = Path(root) / "stage4_mi/runs" / config / rid / "mi_progress.json" if stage == "mi" else Path(root) / "stage5_rfe/runs" / config / rid / branch / "rfe_progress.json"
    try:
        value = read_json(path)
        if stage == "mi":
            return f"MI checkpoint: {len(value.get('completed_indices', []))}/380 features"
        return f"RFE checkpoint: {value.get('completed_steps', 0)}/99 eliminations"
    except (OSError, ValueError):
        return None


def run_worker(action, payload):
    """Start one fresh interpreter, tee streams to terminal + durable append logs."""
    root, source = Path(payload["root"]), Path(payload["source"])
    label = "_".join(str(v) for v in (action, payload.get("config"), payload.get("branch")) if v)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    command = [sys.executable, "-u", str(source / "colab/colab_pipeline_v2.py"), "--worker", action]
    env = os.environ.copy()
    env["PYTHONPATH"], env["PYTHONUNBUFFERED"], env["PYTHONDONTWRITEBYTECODE"] = str(source), "1", "1"
    paths = [root / "logs" / (label + "_" + attempt + ".log"),
             source / "logs/ALL_EXECUTION_RESULTS.log"]
    handles = []
    # Logging is a mandatory launch precondition, rather than a success claim
    # after a long experiment whose output could not be saved.
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("a", encoding="utf-8", buffering=1))
    except OSError:
        for h in handles:
            h.close()
        raise ContractError("Logging unavailable; worker was not started: " + str(paths))
    started, start_monotonic = utc(), time.monotonic()
    process, parsed, exit_code, interrupted = None, None, None, False
    result_count = 0
    peak_rss = 0
    info = read_json(root / "RUN_INFO.json")
    config_run_id = info["run_ids"].get(payload.get("config"), info["experiment_id"])

    def emit(channel, line):
        entry = json.dumps({"time": utc(), "experiment": root.name, "task": label,
                            "run_id": config_run_id, "channel": channel, "text": redact(line)}, ensure_ascii=False)
        try:
            for h in handles:
                h.write(entry + "\n")
                h.flush()
        except OSError as exc:
            print(f"LOGGING_FAILED: {exc}; this output may not have been saved", file=sys.stderr, flush=True)
            raise
        if channel == "stdout" and line.startswith(RESULT_MARKER):
            print("Worker result saved to log/report (structured JSON).", flush=True)
        else:
            print(("[stderr] " if channel == "stderr" else "") + redact(line), flush=True)

    try:
        emit("START", json.dumps({"command": command, "start": started, "scientific_status": SCIENTIFIC_STATUS,
            "config": payload.get("config"), "branch": payload.get("branch"), "run_id": config_run_id,
            "request": {k: payload.get(k) for k in ("mode", "source", "dataset", "root", "config", "branch", "git_commit")}}))
    except OSError:
        for handle in handles:
            handle.close()
        raise ContractError("Logging failed before worker launch")
    events = queue.Queue()

    def pump(stream, channel):
        try:
            for line in stream:
                events.put((channel, line.rstrip("\n")))
        finally:
            events.put((channel, None))

    try:
        process = subprocess.Popen(command, cwd=str(source), env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1)
        process.stdin.write(json.dumps(payload))
        process.stdin.close()
        for stream, channel in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            threading.Thread(target=pump, args=(stream, channel), daemon=True).start()
        done, last_progress, last_tick = 0, None, time.monotonic()
        while done < 2 or process.poll() is None:
            try:
                channel, line = events.get(timeout=0.25)
                if line is None:
                    done += 1
                else:
                    emit(channel, line)
                    if channel == "stdout" and line.startswith(RESULT_MARKER):
                        result_count += 1
                        parsed = json.loads(line[len(RESULT_MARKER):])
            except queue.Empty:
                pass
            try:
                import psutil
                parent = psutil.Process(process.pid)
                rss = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True) if p.is_running())
                peak_rss = max(peak_rss, rss)
            except Exception:
                pass
            if action in {"mi", "rfe"} and time.monotonic() - last_tick >= 5:
                progress = _tail_checkpoint(root, info, payload.get("config"), action, payload.get("branch"))
                if progress != last_progress or time.monotonic() - last_tick >= 30:
                    emit("PROGRESS", f"elapsed={time.monotonic()-start_monotonic:.1f}s; {progress or 'first fit/checkpoint pending'}")
                    last_progress, last_tick = progress, time.monotonic()
        exit_code = process.wait()
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        parsed = {"status": "INTERRUPTED" if interrupted else "FAIL", "errors": [str(exc)]}
    finally:
        if process is not None:
            exit_code = process.returncode
        if parsed is None:
            parsed = {"status": "FAIL", "errors": ["Worker returned no structured result; exit zero alone is not PASS"]}
        if not interrupted and result_count != 1:
            parsed = {**parsed, "status": "FAIL", "errors": list(parsed.get("errors", [])) +
                      [f"Expected exactly one worker JSON result; received {result_count}"]}
        if exit_code != 0 and parsed.get("status") == "PASS":
            parsed = {**parsed, "status": "FAIL", "errors": [f"Child exit code {exit_code}"]}
        record = {"action": action, "start": started, "end": utc(), "exit_code": exit_code,
                  "config": payload.get("config"), "branch": payload.get("branch"), "run_id": config_run_id,
                  "runtime_seconds": time.monotonic() - start_monotonic, "program_status": parsed.get("status"),
                  "scientific_status": SCIENTIFIC_STATUS, "peak_process_tree_rss_bytes_sampled": peak_rss or None,
                  "log_paths": [str(p) for p in paths], "result": parsed}
        try:
            emit("END", json.dumps(record, default=str))
        finally:
            for handle in handles:
                handle.close()
        record["report_path"] = save_report(root, label, record)
    if interrupted:
        raise KeyboardInterrupt("Worker interrupted; partial checkpoints preserved. Use resume with this root.")
    print("REPORT:", record["report_path"], flush=True)
    return record


def run_jobs(action, payload):
    """Keep per-config reports even when another configuration fails."""
    ensure_selection(payload["configs"], payload["branches"])
    jobs = []
    for config in payload["configs"]:
        branches = payload["branches"] if action in {"rfe", "verify_rfe"} else (None,)
        for branch in branches:
            try:
                result = run_worker(action, {**payload, "config": config, "branch": branch})
                jobs.append({"config": config, "branch": branch, **result})
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                row = {"config": config, "branch": branch, "program_status": "FAIL", "errors": [str(exc)]}
                save_report(payload["root"], "job_launch_failure", row)
                jobs.append(row)
    return jobs


def summary(root):
    root = Path(root)
    info = read_json(root / "RUN_INFO.json")
    rows = []
    for config in CONFIGS:
        rid = info["run_ids"][config]
        for part, relative, marker in (
            ("mi", Path("stage4_mi/runs") / config / rid, "MI_COMPLETE.json"),
            *((b, Path("stage5_rfe/runs") / config / rid / b, "COMPLETE.json") for b in BRANCHES),
        ):
            folder = root / relative
            seal = root / "task_records" / config / part / "COMPLETE.json"
            state = "NOT_STARTED"
            if (folder / marker).exists():
                state = "COMPLETE_SEAL_PRESENT_NEEDS_VERIFY" if seal.exists() else "COMPLETE_MARKER_NEEDS_VERIFY"
            elif folder.exists():
                state = "INCOMPLETE"
            rows.append({"config": config, "part": part, "run_id": rid, "state": state, "path": str(folder)})
    return rows


def completed_results(root):
    """Reference Cell 12 pattern: read saved result/report JSON only for display."""
    root = Path(root)
    latest = {}
    for path in sorted((root / "reports").glob("*.json")):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if report.get("action") not in {"verify_mi", "verify_rfe"}:
            continue
        config = report.get("config") or report.get("result", {}).get("config")
        branch = report.get("branch") or report.get("result", {}).get("branch") or "mi"
        key = (config, branch)
        if key not in latest or report.get("end", "") >= latest[key][0].get("end", ""):
            latest[key] = (report, path)
    rows = summary(root)
    for row in rows:
        evidence = latest.get((row["config"], row["part"]))
        row["verification_status"] = "NOT_RUN"
        row["scientific_status"] = SCIENTIFIC_STATUS
        if evidence:
            report, path = evidence
            row.update(verification_status=report.get("program_status", "UNKNOWN"),
                       verification_as_of=report.get("end"), verify_report=str(path),
                       verify_exit_code=report.get("exit_code"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, choices=("dependencies", "tests", "metadata", "preflight",
                                                          "mi", "rfe", "verify_mi", "verify_rfe"))
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        result = worker_action(args.worker, payload)
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "UNSUPPORTED" if "UNSUPPORTED:" in str(exc) else "BLOCKED" if isinstance(exc, ContractError) else "FAIL",
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    if args.worker in {"mi", "rfe"} and result.get("status") != "PASS":
        try:
            part = "mi" if args.worker == "mi" else payload["branch"]
            root = Path(payload["root"])
            if read_json(root / "RUN_INFO.json").get("format") == FORMAT:
                attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
                marker = root / "task_records" / payload["config"] / part / (result["status"] + "_" + attempt + ".json")
                write_json_once(marker, {"created_at": utc(), "result": result})
        except Exception as exc:
            print(f"Failure marker could not be saved: {exc}", file=sys.stderr, flush=True)
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
