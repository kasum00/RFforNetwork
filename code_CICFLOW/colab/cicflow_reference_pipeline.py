"""CICFLOW experiments following the ColabPro cell and artifact contract.

Only explicit notebook/CLI calls execute work. Original Stage 0-4 sources,
datasets, completed runs and the ColabPro reference notebook are never edited.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
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
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "cicflow-colabpro-layout-v1"
SCIENTIFIC_STATUS = "PROVISIONAL_RESEARCH"
MARKER = "__CICFLOW_REFERENCE_RESULT__"
CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
EXPECTED = dict(zip(CONFIGS, (
    {"train": 326157, "val": 104147, "test": 103494},
    {"train": 163203, "val": 52145, "test": 51824},
    {"train": 311423, "val": 96408, "test": 95865},
    {"train": 155814, "val": 48258, "test": 47991},
)))
SPLIT_FILES = {
    "train": ("train_windows.parquet", "train.parquet", "train_features.parquet"),
    "val": ("val_windows.parquet", "validation_windows.parquet", "val.parquet", "validation.parquet"),
    "test": ("test_windows.parquet", "test.parquet", "test_features.parquet"),
}
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_split=3,
                 max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
SVM_PARAMS = dict(C=10.0, gamma="scale", kernel="rbf", probability=False,
                  cache_size=10000, shrinking=True, max_iter=20000,
                  class_weight="balanced", random_state=42)
RFE_PARAMS = {
    "rf": dict(n_estimators=100, max_depth=10, min_samples_split=2,
               max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
    "svm": dict(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
}
REQUIRED_ARTIFACTS = (
    "artifact_manifest.json", "events.jsonl", "features.json", "final_scaler.joblib",
    "k_search_results.json", "mi_feature_scores.csv", "mi_top100.json", "model.joblib",
    "preflight_audit.json", "progress.json", "result.json", "rfe_feature_ranking.csv",
    "selection_frozen.json", "hyperparameters.json", "run_identity.json",
    "resource_usage.json", "COMPLETE.json",
)
SOURCE_FILES = (
    "colab/cicflow_reference_pipeline.py", "colab/test_cicflow_reference_pipeline.py",
    "stage4_mi/mi_selection.py", "stage5_rfe/rfe_selection.py",
)


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    config: str
    run_name: str

    @property
    def branch(self):
        if self.experiment not in ("mi_rfe_rf", "mi_rfe_svm"):
            raise ContractError("XGBoost ranking is BLOCKED pending source reconciliation; select RF or SVM")
        return self.experiment.removeprefix("mi_rfe_")

    def validate(self):
        self.branch
        if self.config not in CONFIGS:
            raise ContractError("Unknown FW/FS configuration")
        prefix = f"{self.experiment}_{self.config}_"
        if not self.run_name.startswith(prefix) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name):
            raise ContractError(f"Use a safe run name beginning {prefix}")
        return self


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    """Single writer under RunLock; partial publication is never silently repaired."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Preserve existing artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def mutable_json(path, value):
    """Only progress in our locked incomplete directory may be replaced."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact_info(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def check_files(directory, records):
    for name, expected in records.items():
        if Path(name).name != name or artifact_info(Path(directory) / name) != expected:
            raise ContractError(f"Artifact hash/size mismatch: {name}")


def activate_source(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    for name in ("stage4_mi", "stage5_rfe", "colab"):
        package = sys.modules.get(name)
        if package is None or str(source / name) not in list(getattr(package, "__path__", [])):
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]
            module = types.ModuleType(name)
            module.__path__ = [str(source / name)]
            sys.modules[name] = module
    importlib.invalidate_caches()


def source_snapshot(source):
    records = {name: sha256(Path(source) / name) for name in SOURCE_FILES}
    return {"files": records, "sha256": digest(records)}


def dependencies():
    imports = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
               "pyarrow": "pyarrow", "psutil": "psutil", "joblib": "joblib"}
    versions, errors = {}, []
    for name, module in imports.items():
        try:
            importlib.import_module(module)
            versions[name] = importlib.metadata.version(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return {"status": "PASS" if not errors else "BLOCKED", "python": platform.python_version(),
            "os": os.name, "machine": platform.machine(),
            "versions": versions, "errors": errors, "automatic_install": False}


def contract(branch):
    if branch not in RFE_PARAMS:
        raise ContractError("No reconciled XGBoost ranking contract")
    return {
        "variant": FORMAT, "scientific_status": SCIENTIFIC_STATUS,
        "reference_status": "REFERENCE_ALGORITHM_MATCH_TARGET_NOT_RUNTIME_VERIFIED",
        "reference": "COMPACT_MI_RFE_ColabPro.ipynb -> huyen_code/ciciomt24_training.py",
        "preprocessing": "numeric float64; replace NaN/+Inf/-Inf with 0; cast float32; preserve rows",
        "mi": {"input_features": 380, "calls": 1, "scaler": "StandardScaler.fit_transform(Train)",
               "random_state": 42, "discrete_features": "auto", "n_neighbors": 3,
               "feature_order": "Train Parquet schema", "ranking": "pandas.Series.sort_values(ascending=False)",
               "top_k": 100, "feature_specific_seeds": False},
        "top100_scaler": "new StandardScaler fitted on Train Top100; Validation transform only",
        "rfe": {"estimator": "RandomForestClassifier", "parameters": RFE_PARAMS[branch],
                "step": 1, "n_features_to_select": 1, "importance_transform": "square",
                "tie_semantics": "numpy.argsort default, same as installed sklearn RFE"},
        "classifier": RF_PARAMS if branch == "rf" else SVM_PARAMS,
        "k_grid": list(range(1, 101)), "k_criterion": "Validation Accuracy strict greater; ties smallest k",
        "final_scaler_and_classifier_fit_role": "Train+Validation after frozen selection",
        "test_evaluation_calls": 1, "official_reproduction": False,
        "old_stage4_variant": "per-feature seed42+index; not reused by this experiment",
    }


def make_model(branch):
    if branch == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS)
    if branch == "svm":
        from sklearn.svm import SVC
        return SVC(**SVM_PARAMS)
    raise ContractError("XGBoost is BLOCKED: no reconciled ranking contract")


def resolve_split(folder, role):
    existing = [folder / name for name in SPLIT_FILES[role] if (folder / name).is_file()]
    if len(existing) != 1:
        raise ContractError(f"Expected exactly one {role} file in {folder}; found {existing}")
    return existing[0]


def metadata_audit(source, dataset, config):
    """Footers only for all roles. No Train/Validation/Test values here."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    activate_source(source)
    from stage4_mi.mi_selection import load_feature_schema, validate_feature_names, PROVENANCE_TOKENS
    if config not in CONFIGS or Path(dataset).name != "CICFLOWMETER_FEATURE_CSV":
        raise ContractError("Wrong CICFLOW dataset/configuration")
    folder = Path(dataset).resolve() / config
    names = validate_feature_names(load_feature_schema(folder))
    if any(any(token in name.lower() for token in (*PROVENANCE_TOKENS, "label", "target")) for name in names):
        raise ContractError("Label/target/provenance must not be a feature")
    splits = {}
    for role in SPLIT_FILES:
        path = resolve_split(folder, role)
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if schema.names != names + ["Window_Label"] or parquet.metadata.num_rows != EXPECTED[config][role]:
            raise ContractError(f"{role} schema/order/row count mismatch: {path}")
        if not all(pa.types.is_integer(field.type) or pa.types.is_floating(field.type) for field in schema):
            raise ContractError(f"Nonnumeric feature/label type: {path}")
        stat = path.stat()
        splits[role] = {"path": str(path), "rows": parquet.metadata.num_rows,
                        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "schema_sha256": digest(str(schema)), "row_groups": parquet.metadata.num_row_groups}
    sidecars = {name: sha256(folder / name) for name in
                ("feature_schema.json", "window_feature_schema.json", "integrity_report.json", "stage3_integrity_report.json")
                if (folder / name).is_file()}
    binding = {"config": config, "names": names, "splits": splits, "sidecars": sidecars}
    return {"status": "PASS", "binding": binding, "fingerprint": digest(binding),
            "values_read": False, "fingerprint_scope": "file stat + Parquet footer + small sidecars; NOT full content hash",
            "capture_session_independence": "NOT_VERIFIED"}


def resource_gate(rows, available, free_disk, branch):
    # Conservative admission estimate, not a measured fit benchmark or guarantee.
    required = rows["train"] * 380 * 24 + (rows["train"] + rows["val"]) * 100 * 16 + 1024**3
    if branch == "svm":
        required += SVM_PARAMS["cache_size"] * 1024**2
    reasons = []
    if available < required:
        reasons.append(f"Available RAM {available} < admission estimate {required}")
    if free_disk < 4 * 1024**3:
        reasons.append("Output volume has less than 4 GiB free")
    return {"status": "PASS" if not reasons else "BLOCKED", "required_ram_estimate_bytes": required,
            "available_ram_bytes": available, "disk_free_bytes": free_disk, "reasons": reasons,
            "runtime_estimated": False, "parameter_reduction_allowed": False}


def locations(output, spec):
    spec.validate()
    output = Path(output).resolve()
    return output / "checkpoints" / (spec.run_name + ".incomplete"), output / "experiment_results" / spec.run_name


def next_run_name(output, experiment, config):
    for number in range(1, 100000):
        spec = RunSpec(experiment, config, f"{experiment}_{config}_run{number:02d}")
        checkpoint, final = locations(output, spec)
        if not checkpoint.exists() and not final.exists() and not final.with_name(final.name + ".incomplete").exists():
            return spec.run_name
    raise ContractError("No unused run name")


def run_tests(source):
    activate_source(source)
    name = "colab.test_cicflow_reference_pipeline"
    if name in sys.modules:
        del sys.modules[name]
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun > 0 and result.wasSuccessful() and not result.skipped
    return {"status": "PASS" if passed else "FAIL", "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
            "source_sha256": source_snapshot(source)["sha256"], "versions": dependencies()}


def preflight(request):
    import psutil
    spec = RunSpec(**request["spec"]).validate()
    checkpoint, final = locations(request["output"], spec)
    source = source_snapshot(request["source"])
    deps = dependencies()
    audit = metadata_audit(request["source"], request["dataset"], spec.config)
    output = Path(request["output"])
    if not output.is_dir():
        raise ContractError("Output root must exist")
    source_root, dataset_root = Path(request["source"]).resolve(), Path(request["dataset"]).resolve()
    if (output.resolve() == source_root or source_root in output.resolve().parents
            or dataset_root == output.resolve() or dataset_root in output.resolve().parents):
        raise ContractError("Output must not be inside source or dataset")
    gate = resource_gate(EXPECTED[spec.config], psutil.virtual_memory().available,
                         shutil.disk_usage(output).free, spec.branch)
    evidence = request.get("unit_tests", {})
    completed_resume = (request.get("mode") == "resume" and
                        (final.is_dir() or (checkpoint / "COMPLETE.json").is_file()))
    needs_fit = request.get("mode") != "verify" and not completed_resume
    reasons = (list(gate["reasons"]) if needs_fit else []) + deps["errors"]
    if needs_fit:
        try:
            for parent in (checkpoint.parent, final.parent):
                parent.mkdir(parents=True, exist_ok=True)
                probe = parent / (".cicflow_write_probe_" + uuid.uuid4().hex)
                with probe.open("xb") as handle:
                    handle.write(b"CICFLOW_WRITE_PROBE")
                    handle.flush()
                    os.fsync(handle.fileno())
                probe.unlink()  # Only the unique file created by this preflight.
        except OSError as exc:
            reasons.append(f"Output write probe failed: {exc}")
    if (evidence.get("status") != "PASS" or not evidence.get("tests_run")
            or evidence.get("source_sha256") != source["sha256"] or evidence.get("versions") != deps):
        reasons.append("Unit/regression evidence is missing, stale, skipped or failed")
    mode = request.get("mode")
    if mode not in ("dry_run", "full_train", "resume", "verify"):
        reasons.append("Unsupported mode")
    if mode in ("dry_run", "full_train") and (checkpoint.exists() or final.exists()):
        reasons.append("Run already exists; use a new run name or explicit resume/verify")
    if mode in ("resume", "verify") and not (checkpoint.is_dir() or final.is_dir()):
        reasons.append("No existing run for resume/verify")
    if mode in ("full_train", "resume") and request.get("confirmation") != "RUN PROVISIONAL " + spec.run_name:
        reasons.append("Exact run-specific confirmation is required")
    import inspect
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from stage5_rfe.rfe_selection import branch_contract
    mi_signature = inspect.signature(mutual_info_classif)
    if (mi_signature.parameters["n_neighbors"].default != 3
            or mi_signature.parameters["discrete_features"].default != "auto"):
        reasons.append("Installed MI defaults differ from reference: source reconciliation required")
    if branch_contract(spec.branch)["parameters"] != RFE_PARAMS[spec.branch]:
        reasons.append("Existing RFE estimator source differs from the reconciled reference")
    identity = {"format": FORMAT, "spec": asdict(spec), "source": source,
                "source_root": str(source_root), "dataset_root": str(dataset_root),
                "dataset": audit["binding"], "dataset_fingerprint": audit["fingerprint"],
                "environment": deps, "contract": contract(spec.branch),
                "resolved_estimators": {"classifier": make_model(spec.branch).get_params(),
                                        "rfe": RandomForestClassifier(**RFE_PARAMS[spec.branch]).get_params()},
                "mi_signature": str(mi_signature),
                "git_commit": request.get("git_commit", "NOT_CAPTURED")}
    identity["sha256"] = digest(identity)
    if mode in ("resume", "verify"):
        base = final if final.is_dir() else checkpoint
        if read_json(base / "run_identity.json") != identity:
            reasons.append("Resume identity mismatch: source/dataset/libraries/parameters/git/run")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons, "identity": identity,
            "resource_gate": gate, "mode": mode, "values_read": False,
            "checkpoint_directory": str(checkpoint), "result_directory": str(final),
            "scientific_status": SCIENTIFIC_STATUS, "official_reproduction": False}


def preprocess(raw, labels, *, both_classes=False):
    import numpy as np
    matrix = np.array(raw, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ContractError("Matrix alignment or binary labels invalid")
    if both_classes and set(np.unique(labels)) != {0, 1}:
        raise ContractError("Train must contain both classes")
    counts = {"nan_replaced": int(np.isnan(matrix).sum()), "inf_replaced": int(np.isinf(matrix).sum())}
    matrix[~np.isfinite(matrix)] = 0
    with np.errstate(over="ignore"):
        matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if not np.isfinite(matrix).all():
        raise ContractError("float32 conversion overflow; no row filtering or clipping allowed")
    return matrix, labels.astype(np.int64), counts


def read_values(audit, role, *, selection=False, names=None, reader=None, test_authorized=False):
    if selection and role not in ("train", "val"):
        raise ContractError("Test is forbidden during selection")
    if role == "test" and not test_authorized:
        raise ContractError("Test requires durable authorization after frozen selection/final refit")
    if role not in SPLIT_FILES:
        raise ContractError("Unknown split")
    if reader is None:
        from pyarrow.parquet import read_table
        reader = read_table
    binding = audit["binding"]
    names = list(binding["names"] if names is None else names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(binding["names"]):
        raise ContractError("Invalid requested feature mapping")
    columns = names + ["Window_Label"]
    table = reader(binding["splits"][role]["path"], columns=columns)
    if table.schema.names != columns or table.num_rows != binding["splits"][role]["rows"]:
        raise ContractError("Read schema/order/count mismatch")
    import numpy as np
    raw = np.column_stack([table.column(name).to_numpy(zero_copy_only=False) for name in names])
    labels = table.column("Window_Label").to_numpy(zero_copy_only=False)
    return preprocess(raw, labels, both_classes=role == "train")


def reference_mi(X, y, names):
    """Same call and pandas tie semantics as training.select_mi; never 380 separate calls."""
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ContractError("MI ordered names/matrix mismatch")
    scaled = StandardScaler().fit_transform(X)
    scores = mutual_info_classif(scaled, y, random_state=42)
    series = pd.Series(scores, index=names).sort_values(ascending=False)
    return series


def best_k(rows):
    if not rows or [r["k"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ContractError("K checkpoints must be contiguous from 1")
    import math
    if any(not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1 for row in rows):
        raise ContractError("Validation accuracy must be finite in [0, 1]")
    best = rows[0]
    for row in rows[1:]:
        if row["accuracy"] > best["accuracy"]:
            best = row
    return best["k"], best["accuracy"]


def metrics(y, pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score, recall_score)
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision_attack": float(precision_score(y, pred, zero_division=0)),
            "recall_attack": float(recall_score(y, pred, zero_division=0)),
            "recall_benign": float(recall_score(y, pred, pos_label=0, zero_division=0)),
            "f1_attack": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_labels_0_benign_1_attack": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y, pred, labels=[0, 1],
                target_names=["Benign_proxy", "Attack_proxy"], output_dict=True, zero_division=0),
            "label_origin": "SCENARIO_PROXY_NOT_INDEPENDENTLY_VERIFIED"}


def timed_fit(model, X, y):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    info = {"seconds": time.perf_counter() - started,
            "warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}
    if hasattr(model, "fit_status_"):
        info.update(converged=bool(model.fit_status_ == 0), fit_status=int(model.fit_status_),
                    n_iter=model.n_iter_.tolist(), support_vectors=int(len(model.support_)))
    return model, info


@contextlib.contextmanager
def run_lock(directory):
    """OS lock is released on process death; no stale-lock deletion on resume."""
    handle = (Path(directory) / "RUN.lock").open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class Progress:
    def __init__(self, directory, identity):
        self.directory, self.identity = Path(directory), identity
        self.started, self.peak = time.perf_counter(), 0
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.stage, self.error = "initializing", None

    def event(self, event, **details):
        import psutil
        with self.lock:
            rss = 0
            for proc in [psutil.Process(), *psutil.Process().children(recursive=True)]:
                try:
                    rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, rss)
            if event != "heartbeat":
                self.stage = event
            row = {"utc": utc(), "event": event, "stage": self.stage, "elapsed_seconds": time.perf_counter()-self.started,
                   "rss_bytes": rss, "available_ram_bytes": psutil.virtual_memory().available,
                   "run_name": self.identity["spec"]["run_name"], "details": details}
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
            mutable_json(self.directory / "progress.json", row)
            print(f"[{row['elapsed_seconds']:.1f}s] {event} stage={self.stage} RSS={rss/1024**3:.2f}GiB "
                  + json.dumps({k: v for k, v in details.items() if k != "result"}), flush=True)

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(30):
            try:
                self.event("heartbeat")
            except Exception as exc:
                self.error = str(exc)
                print(f"LOGGING FAILED: {exc}", file=sys.stderr, flush=True)
                return

    def check(self):
        if self.error:
            raise OSError("Resource/event logging failed: " + self.error)

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()


def dump_model(path, value):
    import joblib
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise ContractError(f"Partial model publication is UNSUPPORTED; preserve {path}")
    joblib.dump(value, temporary, compress=3)
    temporary.rename(path)


def csv_once(path, series):
    import io
    output = io.StringIO()
    series.to_csv(output)
    write_once(path, output.getvalue())


def save_chain(directory, stage, number, binding, state, previous):
    record = {"stage": stage, "number": number, "binding": binding, "previous": previous, "state": state}
    record["sha256"] = digest(record)
    write_once(Path(directory) / f"{stage}_{number:03d}.json", record)
    return record["sha256"]


def load_chain(directory, stage, binding):
    records, previous = [], None
    for number, path in enumerate(sorted(Path(directory).glob(stage + "_*.json")), start=1):
        row = read_json(path)
        raw = {key: value for key, value in row.items() if key != "sha256"}
        if (path.name != f"{stage}_{number:03d}.json" or row.get("number") != number
                or row.get("binding") != binding or row.get("previous") != previous
                or digest(raw) != row.get("sha256")):
            raise ContractError(f"Invalid/gapped/checksum-mismatched {stage} checkpoint: {path}")
        records.append(row["state"])
        previous = row["sha256"]
    return records, previous


def validate_rfe_states(states, count=100):
    import numpy as np
    prior_support = np.ones(count, dtype=bool)
    prior_ranking = np.ones(count, dtype=int)
    for step, state in enumerate(states, start=1):
        support = np.asarray(state["support"])
        ranking = np.asarray(state["ranking"])
        if (support.dtype != bool or support.shape != (count,) or ranking.shape != (count,)
                or state["completed_steps"] != step or step >= count):
            raise ContractError("Invalid RFE checkpoint dimensions/step")
        removed = np.flatnonzero(prior_support & ~support)
        if removed.size != 1 or np.any(support & ~prior_support):
            raise ContractError("RFE must remove exactly one active feature per step")
        expected = prior_ranking.copy()
        expected[~support] += 1
        if not np.array_equal(ranking, expected):
            raise ContractError("RFE ranking update differs from sklearn step=1")
        prior_support, prior_ranking = support.copy(), ranking.copy()
    return prior_support, prior_ranking


def saved_mi(directory, identity, X, y, names, progress):
    state_path = directory / "mi_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity["sha256"]:
            raise ContractError("MI checkpoint identity mismatch")
        check_files(directory, state["files"])
        top = read_json(directory / "mi_top100.json")
        if len(top) != 100 or len(set(top)) != 100 or not set(top) <= set(names):
            raise ContractError("MI Top100 mapping mismatch")
        progress.event("mi_checkpoint_loaded", features=100)
        return top, state
    if any((directory / name).exists() for name in ("mi_feature_scores.csv", "mi_top100.json")):
        raise ContractError("UNSUPPORTED: partial MI publication without mi_state.json")
    progress.event("mi_started", features=380, calls=1, random_state=42, scaler="StandardScaler")
    started = time.perf_counter()
    series = reference_mi(X, y, names)
    import numpy as np
    if len(series) != 380 or not np.isfinite(series.to_numpy()).all() or (series < 0).any():
        raise ContractError("MI did not return 380 finite nonnegative scores")
    top = series.head(100).index.tolist()
    csv_once(directory / "mi_feature_scores.csv", series)
    write_once(directory / "mi_top100.json", top)
    state = {"identity": identity["sha256"], "seconds": time.perf_counter()-started,
             "files": {name: artifact_info(directory / name) for name in ("mi_feature_scores.csv", "mi_top100.json")}}
    write_once(state_path, state)
    progress.event("mi_completed", features=100, seconds=state["seconds"])
    return top, state


def saved_rfe(directory, identity, top, mi_state, X, y, progress):
    import pandas as pd
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from stage5_rfe.rfe_selection import manual_rfe_step1
    binding = {"identity": identity["sha256"], "mi_state": digest(mi_state), "top_order": digest(top)}
    state_path = directory / "rfe_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["binding"] != binding:
            raise ContractError("RFE branch/MI/order identity mismatch")
        check_files(directory, state["files"])
        if sorted(state["ranking"]) != list(range(1, 101)):
            raise ContractError("Full RFE ranking is not a permutation 1..100")
        return state["names"], state
    if (directory / "rfe_feature_ranking.csv").exists():
        raise ContractError("UNSUPPORTED: partial RFE publication")
    states, previous = load_chain(directory / "steps", "rfe", binding)
    support, ranking = validate_rfe_states(states)
    estimator = RandomForestClassifier(**RFE_PARAMS[identity["spec"]["experiment"].removeprefix("mi_rfe_")])
    started = time.perf_counter()
    progress.event("rfe_started", completed_eliminations=len(states), target=1, step=1)

    def checkpoint(state):
        nonlocal previous
        progress.check()
        previous = save_chain(directory / "steps", "rfe", state["completed_steps"], binding, state, previous)
        progress.event("rfe_step_completed", elimination=state["completed_steps"], remaining=sum(state["support"]))

    ranking, support, completed = manual_rfe_step1(
        X, y, estimator, initial_support=support, initial_ranking=ranking,
        completed_steps=len(states), checkpoint_callback=checkpoint,
    )
    # sklearn RFE additionally fits the estimator on the final remaining feature.
    clone(estimator).fit(X[:, support], y)
    series = pd.Series(ranking, index=top).sort_values(ascending=True)
    csv_once(directory / "rfe_feature_ranking.csv", series)
    state = {"binding": binding, "names": series.index.tolist(), "ranking": ranking.tolist(),
             "completed_eliminations": completed, "seconds_this_attempt": time.perf_counter()-started,
             "final_estimator_fit_performed": True,
             "files": {"rfe_feature_ranking.csv": artifact_info(directory / "rfe_feature_ranking.csv")}}
    write_once(state_path, state)
    progress.event("rfe_completed", eliminations=completed)
    return state["names"], state


def search_k(directory, identity, rfe_state, X, y, V, vy, top, ranked, branch, progress):
    binding = {"identity": identity["sha256"], "rfe_state": digest(rfe_state)}
    rows, previous = load_chain(directory / "steps", "k", binding)
    if len(rows) > 100:
        raise ContractError("Too many k checkpoints")
    if rows:
        best_k(rows)
    for k in range(len(rows) + 1, 101):
        progress.check()
        indices = [top.index(name) for name in ranked[:k]]
        progress.event("k_fit_started", k=k, total=100)
        model, fit = timed_fit(make_model(branch), X[:, indices], y)
        if fit.get("warnings") or fit.get("converged") is False:
            progress.event("fit_warning", k=k, fit=fit)
        started = time.perf_counter()
        pred = model.predict(V[:, indices])
        row = {"k": k, **metrics(vy, pred), "fit": fit, "predict_seconds": time.perf_counter()-started}
        del model
        rows.append(row)
        previous = save_chain(directory / "steps", "k", k, binding, row, previous)
        winner, accuracy = best_k(rows)
        progress.event("k_completed", k=k, validation_accuracy=row["accuracy"], best_k=winner, best_validation_accuracy=accuracy)
    winner, accuracy = best_k(rows)
    result = {"best_k": winner, "best_validation_accuracy": accuracy, "results": rows,
              "features": ranked[:winner], "nonconverged_k": [r["k"] for r in rows if r["fit"].get("converged") is False]}
    path = directory / "k_search_results.json"
    if path.exists():
        if read_json(path) != result:
            raise ContractError("Saved k-search differs from checkpoint chain")
    else:
        write_once(path, result)
    return result


def frozen_selection(directory, identity, selection):
    value = {"identity": identity["sha256"], "best_k": selection["best_k"], "features": selection["features"],
             "best_validation_accuracy": selection["best_validation_accuracy"],
             "k_search_sha256": sha256(directory / "k_search_results.json"),
             "test_used_for_selection": False, "selection_fit_role": "Train; score on Validation only"}
    for name, payload in (("selection_frozen.json", value), ("features.json", selection["features"])):
        path = directory / name
        if path.exists():
            if read_json(path) != payload:
                raise ContractError("Frozen selection differs; no final refit allowed")
        else:
            write_once(path, payload)
    return value


def final_refit(directory, identity, frozen, train, y, val, vy, names, branch, progress):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    path = directory / "final_state.json"
    binding = {"identity": identity["sha256"], "frozen": digest(frozen)}
    if path.exists():
        state = read_json(path)
        if state["binding"] != binding:
            raise ContractError("Final model differs from frozen selection")
        check_files(directory, state["files"])
        return state
    if any((directory / name).exists() for name in ("model.joblib", "final_scaler.joblib")):
        raise ContractError("UNSUPPORTED: incomplete final model publication")
    progress.event("final_refit_started", rows=len(y)+len(vy), k=len(frozen["features"]), fit_role="Train+Validation")
    indices = [names.index(name) for name in frozen["features"]]
    # The reference allocates a C-order float32 Train+Validation matrix. Keep
    # that layout before scaler reductions as well as preserving row order.
    combined = np.asarray(np.concatenate([train[:, indices], val[:, indices]], axis=0),
                          dtype=np.float32, order="C")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    del combined
    model, fit = timed_fit(make_model(branch), scaled, np.concatenate([y, vy]))
    del scaled
    dump_model(directory / "model.joblib", model)
    dump_model(directory / "final_scaler.joblib", scaler)
    state = {"binding": binding, "fit": fit, "fit_rows": len(y)+len(vy), "fit_role": "train+val",
             "completed_at": utc(), "files": {name: artifact_info(directory / name)
                                               for name in ("model.joblib", "final_scaler.joblib")}}
    write_once(path, state)
    progress.event("final_refit_completed", fit=fit)
    return state


def start_test_once(directory, identity):
    """Durable intent precedes opening Test. An interrupted Test is never repeated."""
    directory = Path(directory)
    frozen = read_json(directory / "selection_frozen.json")
    final = read_json(directory / "final_state.json")
    if frozen["identity"] != identity["sha256"] or final["binding"] != {
            "identity": identity["sha256"], "frozen": digest(frozen)}:
        raise ContractError("Test requires frozen selection and matching completed final refit")
    check_files(directory, final["files"])
    intent = {"identity": identity["sha256"], "final_state": digest(final), "started_at": utc()}
    write_once(directory / "test_evaluation_started.json", intent)
    return intent


def evaluate_once(directory, identity, audit, frozen, final, progress):
    path = directory / "test_result.json"
    if path.exists():
        result = read_json(path)
        intent = read_json(directory / "test_evaluation_started.json")
        if (result["identity"] != identity["sha256"] or result["intent"] != digest(intent)
                or intent["final_state"] != digest(final) or result["test_evaluation_calls"] != 1):
            raise ContractError("Saved Test result binding mismatch")
        return result
    if (directory / "test_evaluation_started.json").exists():
        raise ContractError("BLOCKED: Test was already started; never repeat evaluation in this run")
    import joblib
    intent = start_test_once(directory, identity)
    progress.event("test_evaluation_started")
    T, ty, cleaning = read_values(audit, "test", names=frozen["features"], test_authorized=True)
    model = joblib.load(directory / "model.joblib")
    scaler = joblib.load(directory / "final_scaler.joblib")
    started = time.perf_counter()
    prediction = model.predict(scaler.transform(T))
    result = {"identity": identity["sha256"], "intent": digest(intent), "test_evaluation_calls": 1,
              "metrics": metrics(ty, prediction), "predict_seconds": time.perf_counter()-started,
              "test_rows_evaluated": len(ty), "cleaning": cleaning, "completed_at": utc()}
    write_once(path, result)
    progress.event("test_evaluation_completed", accuracy=result["metrics"]["accuracy"])
    return result


def complete_checkpoint(directory, identity, audit, selection, final, test, progress):
    # Check source/data again before sealing. No row values are read here.
    if metadata_audit(identity["source_root"], identity["dataset_root"], identity["spec"]["config"])["fingerprint"] != identity["dataset_fingerprint"]:
        raise ContractError("Dataset changed during experiment")
    if source_snapshot(identity["source_root"]) != identity["source"]:
        raise ContractError("Source changed during experiment")
    warnings_present = bool(selection["nonconverged_k"] or final["fit"].get("converged") is False
                            or final["fit"].get("warnings")
                            or any(row["fit"].get("warnings") for row in selection["results"]))
    result = {
        "status": "COMPLETED_WITH_CONVERGENCE_WARNING" if warnings_present else "PASS",
        **identity["spec"], "scientific_status": SCIENTIFIC_STATUS, "official_reproduction": False,
        "reference_compatibility": "FIXTURE_REGRESSION_REQUIRED; NOT FULL DATASET EQUIVALENCE PROOF",
        "best_k": selection["best_k"], "best_validation_accuracy": selection["best_validation_accuracy"],
        "selected_features": selection["features"], "metrics": test["metrics"],
        "rows_by_split": {role: item["rows"] for role, item in audit["binding"]["splits"].items()},
        "test_evaluation_calls": test["test_evaluation_calls"], "test_read_after_final_refit": True,
        "test_rows_evaluated": test["test_rows_evaluated"], "final_refit_role": "train+val",
        "fit": final["fit"], "nonconverged_k": selection["nonconverged_k"],
        "elapsed_seconds_this_attempt": time.perf_counter()-progress.started,
        "peak_rss_bytes_sampled_this_attempt": progress.peak,
        "dataset_fingerprint": identity["dataset_fingerprint"], "identity_sha256": identity["sha256"],
        "source_hashes_match": True, "sampled_training": False, "completed_at": utc(),
        "label_origin": "SCENARIO_PROXY_NOT_INDEPENDENTLY_VERIFIED", "capture_session_independence": "NOT_ESTABLISHED",
    }
    write_once(directory / "result.json", result)
    progress.check()
    progress.event("run_validated", status=result["status"], test_evaluation_calls=1)
    return result


def seal_checkpoint(directory, identity, progress):
    progress.check()
    write_once(directory / "resource_usage.json", {
        "attempt_finished_at": utc(), "peak_rss_bytes_sampled_this_attempt": progress.peak,
        "runtime_seconds_this_attempt": time.perf_counter()-progress.started,
        "history": "events.jsonl includes earlier attempts; samples may miss short peaks",
        "parameters_reduced": False,
    })
    names = set(REQUIRED_ARTIFACTS) - {"artifact_manifest.json", "COMPLETE.json"}
    names.update(("mi_state.json", "rfe_state.json", "final_state.json", "test_result.json", "test_evaluation_started.json"))
    records = {name: artifact_info(directory / name) for name in sorted(names)}
    write_once(directory / "artifact_manifest.json", {
        "format": FORMAT, "identity_sha256": identity["sha256"], "files": records,
        "scientific_status": SCIENTIFIC_STATUS, "official_reproduction": False,
        "durability": "fsync + staging rename on mounted Drive; FUSE atomicity is not guaranteed",
    })
    write_once(directory / "COMPLETE.json", {
        "status": "COMPLETE", "identity_sha256": identity["sha256"],
        "manifest_sha256": sha256(directory / "artifact_manifest.json"), "completed_at": utc(),
        "scientific_status": SCIENTIFIC_STATUS, "official_reproduction": False,
    })


def verify_directory(directory):
    """No dataset values, deserialization, fit, or prediction during verification."""
    directory = Path(directory)
    for name in REQUIRED_ARTIFACTS:
        if not (directory / name).is_file():
            raise ContractError(f"Incomplete publication: missing {name}")
    identity = read_json(directory / "run_identity.json")
    if digest({k: v for k, v in identity.items() if k != "sha256"}) != identity["sha256"]:
        raise ContractError("Identity digest mismatch")
    manifest, complete = read_json(directory / "artifact_manifest.json"), read_json(directory / "COMPLETE.json")
    if (complete["status"] != "COMPLETE" or complete["identity_sha256"] != identity["sha256"]
            or manifest["identity_sha256"] != identity["sha256"]
            or complete["manifest_sha256"] != sha256(directory / "artifact_manifest.json")):
        raise ContractError("Completion/manifest identity mismatch")
    check_files(directory, manifest["files"])
    if not (set(REQUIRED_ARTIFACTS) - {"artifact_manifest.json", "COMPLETE.json"}) <= set(manifest["files"]):
        raise ContractError("Manifest omits required artifacts")
    if identity["contract"] != contract(RunSpec(**identity["spec"]).branch):
        raise ContractError("Scientific contract differs from current source")
    names = identity["dataset"]["names"]
    top = read_json(directory / "mi_top100.json")
    with (directory / "mi_feature_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    if (len(rows) != 380 or [r[0] for r in rows[:100]] != top or len(set(top)) != 100
            or set(r[0] for r in rows) != set(names)):
        raise ContractError("MI 380/ranking/Top100 mapping mismatch")
    scores = [float(r[1]) for r in rows]
    import math
    if any(not math.isfinite(v) or v < 0 for v in scores) or scores != sorted(scores, reverse=True):
        raise ContractError("MI score ranking invalid")
    rfe = read_json(directory / "rfe_state.json")
    mi = read_json(directory / "mi_state.json")
    if mi["identity"] != identity["sha256"] or rfe["binding"] != {
            "identity": identity["sha256"], "mi_state": digest(mi), "top_order": digest(top)}:
        raise ContractError("MI/RFE identity or Top100 dependency binding mismatch")
    check_files(directory, mi["files"])
    check_files(directory, rfe["files"])
    if rfe["completed_eliminations"] != 99 or rfe["final_estimator_fit_performed"] is not True:
        raise ContractError("RFE elimination/final-estimator fit incomplete")
    with (directory / "rfe_feature_ranking.csv").open(newline="", encoding="utf-8") as handle:
        ranking_rows = list(csv.reader(handle))[1:]
    ranked = [r[0] for r in ranking_rows]
    if (len(ranked) != 100 or set(ranked) != set(top) or ranked != rfe["names"]
            or [int(r[1]) for r in ranking_rows] != list(range(1, 101))
            or sorted(rfe["ranking"]) != list(range(1, 101))
            or [top.index(name) for name in ranked] != sorted(range(100), key=lambda i: rfe["ranking"][i])):
        raise ContractError("RFE full ranking/order invalid")
    frozen = read_json(directory / "selection_frozen.json")
    selection = read_json(directory / "k_search_results.json")
    if len(selection["results"]) != 100:
        raise ContractError("K grid incomplete")
    winner, accuracy = best_k(selection["results"])
    if (winner != frozen["best_k"] or winner != selection["best_k"]
            or frozen["features"] != ranked[:winner] or read_json(directory / "features.json") != frozen["features"]
            or accuracy != selection["best_validation_accuracy"] or accuracy != frozen["best_validation_accuracy"]
            or selection["features"] != frozen["features"]
            or sha256(directory / "k_search_results.json") != frozen["k_search_sha256"]):
        raise ContractError("Frozen selection/k tie-break invalid")
    final, test = read_json(directory / "final_state.json"), read_json(directory / "test_result.json")
    intent = read_json(directory / "test_evaluation_started.json")
    result = read_json(directory / "result.json")
    if (final["binding"] != {"identity": identity["sha256"], "frozen": digest(frozen)}
            or intent["final_state"] != digest(final) or test["intent"] != digest(intent)
            or test["identity"] != identity["sha256"] or frozen["identity"] != identity["sha256"]
            or result["identity_sha256"] != identity["sha256"]
            or test["test_evaluation_calls"] != 1 or result["test_evaluation_calls"] != 1
            or result["metrics"] != test["metrics"] or result["rows_by_split"] != EXPECTED[identity["spec"]["config"]]
            or result["best_k"] != winner or result["selected_features"] != frozen["features"]
            or result["status"] not in ("PASS", "COMPLETED_WITH_CONVERGENCE_WARNING")
            or final["fit_rows"] != result["rows_by_split"]["train"] + result["rows_by_split"]["val"]
            or result["test_rows_evaluated"] != result["rows_by_split"]["test"]):
        raise ContractError("Final refit/Test/result binding invalid")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    event_names = [e["event"] for e in events]
    if event_names.count("test_evaluation_started") != 1 or event_names.count("test_evaluation_completed") != 1:
        raise ContractError("Test event count is not exactly one")
    if not (event_names.index("selection_frozen") < event_names.index("final_refit_completed")
            < event_names.index("test_evaluation_started") < event_names.index("test_evaluation_completed")
            < event_names.index("run_validated")):
        raise ContractError("Selection/refit/Test order invalid")
    return {"status": "PASS", "result_status": result["status"], "directory": str(directory),
            "files_verified": len(manifest["files"]), "best_k": winner, "metrics": result["metrics"],
            "scientific_status": SCIENTIFIC_STATUS, "official_reproduction": False}


def publish(directory, destination):
    """Copy only sealed artifacts/models, never Parquet or feature matrices."""
    verified = verify_directory(directory)
    if destination.exists():
        existing = verify_directory(destination)
        if sha256(destination / "artifact_manifest.json") != sha256(directory / "artifact_manifest.json"):
            raise ContractError("Existing completed result differs; never overwrite")
        return existing
    staging = destination.with_name(destination.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    manifest = read_json(directory / "artifact_manifest.json")
    allowed = set(manifest["files"]) | {"artifact_manifest.json", "COMPLETE.json"}
    if any(path.name not in allowed and not path.name.endswith(".partial") for path in staging.iterdir()):
        raise ContractError("Unknown files in publication staging directory")
    for name in sorted(allowed):
        source, target = directory / name, staging / name
        if target.exists():
            if artifact_info(source) != artifact_info(target):
                raise ContractError(f"Partial publication differs: {target}")
            continue
        temporary = target.with_name(target.name + ".partial")
        if temporary.exists():
            raise ContractError(f"UNSUPPORTED: partial publication copy preserved: {temporary}")
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if artifact_info(source) != artifact_info(temporary):
            raise OSError(f"Publication copy hash mismatch: {source}")
        temporary.rename(target)
    verify_directory(staging)
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    return {**verified, "directory": str(destination)}


def execute_run(request):
    if request.get("mode") not in ("full_train", "resume"):
        raise ContractError("Training requires explicit full_train/resume")
    plan = preflight(request)
    if plan["status"] != "PASS":
        raise ContractError("Preflight BLOCKED: " + "; ".join(plan["reasons"]))
    identity = plan["identity"]
    spec = RunSpec(**identity["spec"])
    directory, destination = locations(request["output"], spec)
    if destination.exists():
        return verify_directory(destination)
    if request["mode"] == "full_train" or not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        write_once(directory / "run_identity.json", identity)
        write_once(directory / "preflight_audit.json", plan)
        write_once(directory / "hyperparameters.json", identity["contract"])
    with run_lock(directory):
        if (directory / "COMPLETE.json").exists():
            return publish(directory, destination)
        if (directory / "artifact_manifest.json").exists() or (directory / "result.json").exists():
            raise ContractError("UNSUPPORTED: interrupted final sealing; preserve checkpoint for inspection")
        if list(directory.rglob("*.partial")):
            raise ContractError("UNSUPPORTED: partial artifact exists; never delete it to force resume")
        try:
            from sklearn.preprocessing import StandardScaler
            activate_source(request["source"])
            audit = {"binding": identity["dataset"]}
            names = audit["binding"]["names"]
            with Progress(directory, identity) as progress:
                progress.event("attempt_started", mode=request["mode"], scientific_status=SCIENTIFIC_STATUS)
                if (directory / "test_evaluation_started.json").exists() and not (directory / "test_result.json").exists():
                    raise ContractError("BLOCKED: interrupted Test evaluation cannot be repeated")
                progress.event("train_loading_started")
                train, y, cleaning = read_values(audit, "train", selection=True)
                progress.event("train_loaded", rows=len(y), cleaning=cleaning)
                top, mi_state = saved_mi(directory, identity, train, y, names, progress)
                indices = [names.index(name) for name in top]
                scaler = StandardScaler()
                X = scaler.fit_transform(train[:, indices])
                ranked, rfe_state = saved_rfe(directory, identity, top, mi_state, X, y, progress)
                progress.event("validation_loading_started")
                val, vy, cleaning = read_values(audit, "val", selection=True)
                V = scaler.transform(val[:, indices])
                selection = search_k(directory, identity, rfe_state, X, y, V, vy, top, ranked, spec.branch, progress)
                del X, V, scaler
                frozen = frozen_selection(directory, identity, selection)
                progress.event("selection_frozen", k=selection["best_k"], features=selection["features"])
                final = final_refit(directory, identity, frozen, train, y, val, vy, names, spec.branch, progress)
                del train, val, y, vy
                test = evaluate_once(directory, identity, audit, frozen, final, progress)
                complete_checkpoint(directory, identity, audit, selection, final, test, progress)
            # Heartbeat thread is stopped before event/progress hashes are sealed.
            seal_checkpoint(directory, identity, progress)
            return publish(directory, destination)
        except BaseException as exc:
            try:
                write_once(directory / ("FAILED_" + uuid.uuid4().hex + ".json"), {
                    "status": "BLOCKED" if isinstance(exc, ContractError) else "FAILED",
                    "utc": utc(), "error": str(exc), "run_name": spec.run_name,
                    "completed": False, "scientific_status": SCIENTIFIC_STATUS})
            except Exception as logging_error:
                print(f"FAILURE RECORD NOT SAVED: {logging_error}", file=sys.stderr, flush=True)
            raise


def completed_results(output):
    rows = []
    for path in sorted((Path(output) / "experiment_results").glob("mi_rfe_*/result.json")):
        if path.parent.name.endswith(".incomplete"):
            continue
        try:
            result = read_json(path)
            verification = verify_directory(path.parent)
            rows.append({"run_name": result["run_name"], "experiment": result["experiment"], "config": result["config"],
                         "status": result["status"], "verify": verification["status"], "best_k": result["best_k"],
                         "validation_accuracy": result["best_validation_accuracy"], **result["metrics"],
                         "scientific_status": SCIENTIFIC_STATUS, "directory": str(path.parent)})
        except Exception as exc:
            rows.append({"run_name": path.parent.name, "status": "INVALID_OR_FOREIGN_ARTIFACT", "error": str(exc)})
    return rows


def protect(text):
    # Redact common credential assignments/bearers in both displayed and saved output.
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)\S+", r"\1[REDACTED]", str(text))
    return re.sub(r"(?i)((?:password|passwd|secret|access_token|api_key|token)\s*[=:]\s*)[^\s,;]+",
                  r"\1[REDACTED]", text)


def safe_request(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if re.search(r"password|secret|token|api_key", key, re.I)
                else safe_request(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_request(item) for item in value]
    return protect(value) if isinstance(value, str) else value


def launch(action, request):
    """Notebook supervisor: live stdout+stderr, append logs, retain native exit code."""
    source = Path(request["source"]).resolve()
    name = request.get("spec", {}).get("run_name", "setup")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ContractError("Unsafe logging run name")
    log_root = source / "logs" / "colab_reference" / name
    log_root.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    task_log = log_root / f"{action}_{token}.log"
    aggregate = source / "logs" / "ALL_EXECUTION_RESULTS.log"
    handles, logging_errors = [], []
    for path in (task_log, aggregate):
        try:
            handles.append(path.open("a", encoding="utf-8"))
        except OSError as exc:
            logging_errors.append(f"LOG OPEN FAILED {path}: {exc}")
    if logging_errors:
        for handle in handles:
            handle.close()
        raise OSError("; ".join(logging_errors) + "; child not started")

    def log(line):
        for handle in handles:
            try:
                handle.write(protect(line) + "\n")
                handle.flush()
            except OSError as exc:
                error = f"LOG WRITE FAILED: {exc}"
                if error not in logging_errors:
                    print(error, file=sys.stderr, flush=True)
                    logging_errors.append(error)

    command = [sys.executable, "-u", str(source / "colab/cicflow_reference_pipeline.py"), "--worker", action]
    start, start_time = utc(), time.perf_counter()
    process, native_exit, records = None, None, []
    log("=" * 72)
    log(json.dumps({"START": start, "STAGE": "MI-RFE full experiment", "TASK": action,
                    "REQUEST": safe_request(request), "COMMAND_ARGV": command, "SCIENTIFIC_STATUS": SCIENTIFIC_STATUS}))
    try:
        env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", cwd=source, env=env)
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        messages = queue.Queue()

        def pump(stream, label):
            try:
                for line in iter(stream.readline, ""):
                    messages.put((label, line.rstrip("\r\n")))
            finally:
                messages.put((label, None))
                stream.close()

        threads = [threading.Thread(target=pump, args=(stream, label), daemon=True)
                   for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
        for thread in threads:
            thread.start()
        closed = 0
        while closed < 2:
            label, line = messages.get()
            if line is None:
                closed += 1
                continue
            log(f"{utc()} [{label}] {line}")
            if label == "stdout" and line.startswith(MARKER):
                try:
                    records.append(json.loads(line[len(MARKER):]))
                except ValueError:
                    records.append({"status": "FAIL", "error": "Invalid worker protocol JSON"})
            else:
                print(protect(line), file=sys.stderr if label == "stderr" else sys.stdout, flush=True)
        for thread in threads:
            thread.join()
        native_exit = process.wait()
        result = records[0] if len(records) == 1 else {"status": "FAIL", "error": f"Expected one worker result, got {len(records)}"}
        program_status = result.get("status", "UNKNOWN")
        if native_exit != 0 and program_status == "PASS":
            program_status = "FAIL"
        envelope = {"action": action, "exit_code": native_exit, "program_status": program_status,
                    "result": result, "start": start, "end": utc(), "runtime_seconds": time.perf_counter()-start_time,
                    "task_log": str(task_log), "aggregate_log": str(aggregate), "logging_errors": logging_errors}
        log(json.dumps(safe_request(envelope)))
        print(f"{action}: status={program_status}, exit_code={native_exit}, log={task_log}", flush=True)
        if logging_errors:
            envelope["program_status"] = "FAIL"
            envelope["logging_status"] = "FAIL"
        else:
            envelope["logging_status"] = "PASS"
        return envelope
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                native_exit = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                native_exit = process.wait()
        log(json.dumps({"END": utc(), "STATUS": "INTERRUPTED_OR_FAILED", "EXIT_CODE": native_exit,
                        "ERROR": protect(str(exc)), "RUNTIME": time.perf_counter()-start_time,
                        "COMPLETED": False}))
        raise
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError as exc:
                print(f"LOG CLOSE FAILED: {exc}", file=sys.stderr)


def worker(action, request):
    if action == "dependencies":
        return dependencies()
    if action == "tests":
        return run_tests(request["source"])
    if action == "metadata":
        return metadata_audit(request["source"], request["dataset"], request["config"])
    if action == "preflight":
        return preflight(request)
    if action == "run":
        return execute_run(request)
    if action == "verify":
        spec = RunSpec(**request["spec"]).validate()
        checkpoint, destination = locations(request["output"], spec)
        if not destination.is_dir():
            raise ContractError("No published completed result; use resume for a sealed checkpoint")
        return verify_directory(destination)
    if action == "summary":
        return {"status": "PASS", "rows": completed_results(request["output"])}
    raise ContractError("Unknown action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("dependencies", "tests", "metadata", "preflight", "run", "verify", "summary"), required=True)
    parser.add_argument("--request-file", type=Path, help="Optional JSON request; otherwise read stdin")
    args = parser.parse_args()
    try:
        request = read_json(args.request_file) if args.request_file else json.load(sys.stdin)
        result = worker(args.worker, request)
        code = 0 if result.get("status") == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "BLOCKED" if isinstance(exc, (ContractError, FileExistsError)) else "FAIL",
                  "error": f"{type(exc).__name__}: {exc}", "scientific_status": SCIENTIFIC_STATUS}
        code = 2 if result["status"] == "BLOCKED" else 1
    print(MARKER + json.dumps(result, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
