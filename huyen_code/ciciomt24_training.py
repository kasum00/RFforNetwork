"""CTU-13 adaptation: bounded benchmarks or one explicitly approved full run.

The original scripts are inspected through AST, never imported (two have import
side effects). Feature selection functions receive Train/Validation only. Test is
loaded by the official runner only after the feature set and k are frozen.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ciciomt24_sliding_window as sw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS = ROOT / "CICIoMT24" / "WINDOWS_CPLUS"
DEFAULT_ORIGINAL = ROOT / "CODE_ctu13"
FEATURE_NAMES = sw.OUTPUT_FEATURE_COLUMNS
CONFIGS = tuple(f"FW_{fw}_FS_{fs}" for fw, fs in sw.WINDOW_CONFIGS)
SEED = 42
TOP_MI = 100
K_GRID = tuple(range(1, 101))
BASELINE_SVM_PARAMS = dict(C=10.0, gamma="scale", kernel="rbf", probability=True,
                           cache_size=10000, shrinking=True, max_iter=20000,
                           random_state=SEED)
COMPACT_SVM_PARAMS = dict(C=10.0, gamma="scale", kernel="rbf", probability=False,
                          cache_size=10000, shrinking=True, max_iter=20000,
                          class_weight="balanced", random_state=SEED)
# Deliberately separate: source SVM-RFE omits min_samples_split/max_features.
SVM_RFE_PARAMS = dict(n_estimators=100, max_depth=15, class_weight="balanced",
                      n_jobs=-1, random_state=SEED)
RF_RFE_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_split=2,
                     max_features="sqrt", class_weight="balanced",
                     random_state=SEED, n_jobs=-1)
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_split=3,
                 max_features="sqrt", class_weight="balanced",
                 random_state=SEED, n_jobs=-1)
BENCHMARK_TRAIN_SIZES = (4096, 8192)
BENCHMARK_VAL_ROWS = 2048
BENCHMARK_K = (1, 10, 100)
BENCHMARK_RFE_REMOVALS = 3
BENCHMARK_STRIDE = 8


def finite_matrix(X: np.ndarray) -> bool:
    """Bound temporary boolean arrays even when X is a multi-GB memory map."""
    return all(np.isfinite(X[start:start + 8192]).all() for start in range(0, len(X), 8192))


def write_json(path: Path, value: object) -> None:
    """Atomic metadata output; callers allocate a fresh run directory."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def source_hashes(root: Path = DEFAULT_ORIGINAL) -> dict:
    hashes = {name: sha256(root / name) for name in sw.ORIGINAL_CT13_SCRIPT_HASHES}
    if hashes != sw.ORIGINAL_CT13_SCRIPT_HASHES:
        raise ValueError("Five CTU-13 originals differ from the approved snapshot")
    return hashes


def source_audit(root: Path = DEFAULT_ORIGINAL) -> dict:
    """Evaluate only harmless literal parameter assignments, not source modules."""
    result = {}
    for name in ("7_senario1_svm.py", "[[CTU13]8_experiment2_svm.py", "[[CTU13]9_ex2_rf.py"):
        path = root / name
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        values = {"RANDOM_STATE": SEED}
        lines = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in {
                "SVM_PARAMS", "RF_PARAMS", "RFE_ESTIMATOR_PARAMS", "TOP_K_MI", "CONFIGS"
            }:
                continue
            # These audited expressions contain literals, dict(), and RANDOM_STATE.
            values[target.id] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                     {"__builtins__": {}, "dict": dict}, values)
            lines[target.id] = node.lineno
        if name == "[[CTU13]8_experiment2_svm.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RandomForestClassifier":
                    values["RFE_ESTIMATOR_PARAMS"] = {
                        kw.arg: eval(compile(ast.Expression(kw.value), str(path), "eval"),
                                     {"__builtins__": {}}, values) for kw in node.keywords
                    }
                    lines["RFE_ESTIMATOR_PARAMS"] = node.lineno
        result[name] = {"parameters": {k: v for k, v in values.items() if k != "RANDOM_STATE"},
                        "assignment_lines": lines}
    a = result["7_senario1_svm.py"]["parameters"]["SVM_PARAMS"]
    b = result["[[CTU13]8_experiment2_svm.py"]["parameters"]
    c = result["[[CTU13]9_ex2_rf.py"]["parameters"]
    assert a == BASELINE_SVM_PARAMS
    assert {**b["SVM_PARAMS"], "random_state": SEED} == COMPACT_SVM_PARAMS
    assert b["RFE_ESTIMATOR_PARAMS"] == SVM_RFE_PARAMS
    assert c["RFE_ESTIMATOR_PARAMS"] == RF_RFE_PARAMS
    assert c["RF_PARAMS"] == RF_PARAMS
    assert b["TOP_K_MI"] == c["TOP_K_MI"] == TOP_MI
    return result


@dataclass
class PreparedSplit:
    role: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    cleaning: dict | None = None

    def validate(self, *, both_classes: bool = False) -> None:
        if self.role not in ("train", "val", "test"):
            raise ValueError("Unknown split role")
        if self.feature_names != FEATURE_NAMES or self.X.shape != (len(self.y), 380):
            raise ValueError("X must have the exact canonical 380 feature columns")
        if self.X.dtype != np.float32 or not finite_matrix(self.X):
            raise ValueError("X must be finite float32 before fitting")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Label must be binary")
        if both_classes and set(np.unique(self.y)) != {0, 1}:
            raise ValueError("Training benchmark requires both proxy classes")


def prepare_xy(frame: pd.DataFrame, role: str) -> PreparedSplit:
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names")
    missing = set(FEATURE_NAMES) | {"Label"}
    missing -= set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)[:4]}")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURE_NAMES):
        raise ValueError("Behavior features must be numeric")
    raw = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64, copy=True)
    cleaning = {"nan_replaced": int(np.isnan(raw).sum()),
                "inf_replaced": int(np.isinf(raw).sum())}
    # Exactly replace +/-Inf -> NaN -> fillna(0), as in all three originals.
    raw[~np.isfinite(raw)] = 0.0
    with np.errstate(over="ignore"):
        X = np.asarray(raw, dtype=np.float32, order="C")
    label = pd.to_numeric(frame["Label"], errors="raise").to_numpy()
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Label contains a missing/non-binary value")
    result = PreparedSplit(role, X, label.astype(np.int8), cleaning=cleaning)
    result.validate()
    return result


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_labels_0_benign_1_attack": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1],
            target_names=["Benign_proxy", "Attack_proxy"], output_dict=True, zero_division=0),
        "label_origin": "SCENARIO_PROXY",
    }


class PeakRSS:
    """Sample RSS of this process and its descendants, including parallel jobs."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.proc = psutil.Process()
        self._sample()
        self.baseline = self.peak
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        total = 0
        for p in [self.proc, *self.proc.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak = max(self.peak, total)

    def _loop(self):
        while not self.stop.wait(0.05):
            self._sample()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()


class RunProgress:
    """Durable stage/resource log only; unfinished runs are never auto-resumed."""
    def __init__(self, directory: Path, config: str, experiment: str, interval=30):
        self.directory, self.config, self.experiment = directory, config, experiment
        self.interval = interval
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.stage = "initializing"
        self.details = {}
        self.proc = psutil.Process()
        self.proc.cpu_percent()

    def __enter__(self):
        self("run_started", config=self.config, experiment=self.experiment,
             resume_supported=False)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __call__(self, event, **details):
        with self.lock:
            if event != "heartbeat":
                self.stage, self.details = event, details
            rss = 0
            for proc in [self.proc, *self.proc.children(recursive=True)]:
                try:
                    rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            record = {"utc": datetime.now(timezone.utc).isoformat(), "event": event,
                      "stage": self.stage, "elapsed_seconds": time.perf_counter() - self.started,
                      "rss_bytes": rss, "available_ram_bytes": psutil.virtual_memory().available,
                      "cpu_percent": self.proc.cpu_percent(), "config": self.config,
                      "experiment": self.experiment, "details": self.details}
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            write_json(self.directory / "progress.json", record)
            visible = {k: v for k, v in details.items() if k != "result"}
            if event == "k_completed" and "result" in details:
                visible.update(validation_accuracy=details["result"]["accuracy"],
                               fit_seconds=details["result"]["fit"]["seconds"])
            print(f"[{record['elapsed_seconds']:.1f}s] {event} stage={self.stage} "
                  f"rss={rss / 1024**3:.3f}GiB available={record['available_ram_bytes'] / 1024**3:.3f}GiB "
                  f"{json.dumps(visible, ensure_ascii=False) if visible else ''}", flush=True)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self("heartbeat")
            except Exception as exc:
                # Keep the failure observable; it is checked before publishing.
                self.logging_error = str(exc)
                print(f"RESOURCE LOG ERROR: {exc}", flush=True)
                break

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        if exc is not None:
            self("run_failed", error=str(exc), official_result_valid=False)


def emit(progress, event, **details):
    if progress is not None:
        progress(event, **details)


def timed_fit(model, X, y) -> tuple[object, dict]:
    if not finite_matrix(X) or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonfinite input or invalid label before model.fit")
    start = time.perf_counter()
    with PeakRSS() as ram, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    info = {"seconds": time.perf_counter() - start, "peak_rss_bytes": ram.peak,
            "incremental_rss_bytes": ram.peak - ram.baseline,
            "warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}
    if hasattr(model, "fit_status_"):
        info.update(fit_status=int(model.fit_status_),
                    n_iter=np.asarray(model.n_iter_).tolist(),
                    support_vectors=int(len(model.support_)),
                    converged=bool(model.fit_status_ == 0))
    forest = model.estimator_ if isinstance(model, RFE) else model
    if isinstance(forest, RandomForestClassifier):
        states = [tree.tree_.__getstate__() for tree in forest.estimators_]
        info["tree_buffers_bytes"] = int(sum(s["nodes"].nbytes + s["values"].nbytes for s in states))
        info["tree_nodes"] = int(sum(len(s["nodes"]) for s in states))
    return model, info


def require_selection_roles(train: PreparedSplit, val: PreparedSplit) -> None:
    train.validate(both_classes=True)
    val.validate()
    if (train.role, val.role) != ("train", "val"):
        raise ValueError("Selection accepts Train and Validation only; Test is forbidden")


def select_mi(train: PreparedSplit) -> dict:
    train.validate(both_classes=True)
    if train.role != "train":
        raise ValueError("MI may fit on Train only")
    start = time.perf_counter()
    with PeakRSS() as ram:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(train.X)
        scores = mutual_info_classif(scaled, train.y, random_state=SEED)
        series = pd.Series(scores, index=train.feature_names).sort_values(ascending=False)
    names = series.head(TOP_MI).index.tolist()
    return {"names": names, "indices": [FEATURE_NAMES.index(n) for n in names],
            "scores": series, "scaler": scaler,
            "measurement": {"seconds": time.perf_counter() - start,
                            "peak_rss_bytes": ram.peak}}


def scale_top100(train: PreparedSplit, val: PreparedSplit, mi: dict):
    require_selection_roles(train, val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.X[:, mi["indices"]])
    X_val = scaler.transform(val.X[:, mi["indices"]])
    return X_train, X_val, scaler


def make_model(kind: str, *, baseline: bool = False):
    if kind == "svm":
        return SVC(**(BASELINE_SVM_PARAMS if baseline else COMPACT_SVM_PARAMS))
    if kind == "rf" and not baseline:
        return RandomForestClassifier(**RF_PARAMS)
    raise ValueError("Experiment must be svm baseline, MI-RFE-SVM or MI-RFE-RF")


def official_hyperparameters(experiment: str) -> dict:
    """Serializable scientific contract for the selected official experiment."""
    if experiment == "baseline_svm":
        return {
            "experiment": experiment, "feature_count": 380,
            "selection": "none", "classifier": BASELINE_SVM_PARAMS.copy(),
            "final_refit_role": "train+val", "test_evaluations": 1,
        }
    if experiment == "mi_rfe_svm":
        return {
            "experiment": experiment, "mi_top_k": TOP_MI,
            "mi_fit_role": "train", "rfe_step": 1,
            "rfe_n_features_to_select": 1,
            "rfe_estimator": SVM_RFE_PARAMS.copy(),
            "k_grid": list(K_GRID),
            "k_criterion": "validation_accuracy_strict_greater_smallest_k_tie",
            "classifier": COMPACT_SVM_PARAMS.copy(),
            "final_refit_role": "train+val", "test_evaluations": 1,
        }
    if experiment == "mi_rfe_rf":
        return {
            "experiment": experiment, "mi_top_k": TOP_MI,
            "mi_fit_role": "train", "rfe_step": 1,
            "rfe_n_features_to_select": 1,
            "rfe_estimator": RF_RFE_PARAMS.copy(), "k_grid": list(K_GRID),
            "k_criterion": "validation_accuracy_strict_greater_smallest_k_tie",
            "classifier": RF_PARAMS.copy(), "final_refit_role": "train+val",
            "test_evaluations": 1,
        }
    raise ValueError("Unknown official experiment")


def rank_rfe(X_train_top100, y_train, names, kind: str, *, target: int = 1) -> dict:
    if len(names) != TOP_MI or X_train_top100.shape[1] != TOP_MI:
        raise ValueError("RF-RFE requires the 100 MI features")
    if kind not in ("svm", "rf") or not 1 <= target <= TOP_MI:
        raise ValueError("Invalid RFE branch/target")
    params = SVM_RFE_PARAMS if kind == "svm" else RF_RFE_PARAMS
    selector = RFE(RandomForestClassifier(**params), n_features_to_select=target, step=1)
    selector, measurement = timed_fit(selector, X_train_top100, y_train)
    ranks = pd.Series(selector.ranking_, index=names).sort_values(ascending=True)
    return {"names": ranks.index.tolist(), "ranks": ranks,
            "measurement": measurement, "target": target,
            "forest_fits": TOP_MI - target + 1,
            "complete_ranking": target == 1}


def search_k(X_train, y_train, X_val, y_val, top_names, ranked_names,
             kind: str, k_values=K_GRID, *, progress=None) -> dict:
    if tuple(k_values) != tuple(sorted(set(k_values))) or not all(1 <= k <= 100 for k in k_values):
        raise ValueError("k must be unique, ascending and between 1 and 100")
    results, best_k, best_acc = [], None, -1.0
    for k in k_values:
        emit(progress, "k_fit_started", k=k, k_total=len(k_values))
        indices = [top_names.index(n) for n in ranked_names[:k]]
        model, fit = timed_fit(make_model(kind), X_train[:, indices], y_train)
        start = time.perf_counter()
        pred = model.predict(X_val[:, indices])
        row = {"k": k, **metrics(y_val, pred), "fit": fit,
               "predict_seconds": time.perf_counter() - start}
        results.append(row)
        if row["accuracy"] > best_acc:  # Source rule: exact ties keep first/smallest k.
            best_k, best_acc = k, row["accuracy"]
        emit(progress, "k_completed", k=k, k_total=len(k_values),
             best_k=best_k, best_validation_accuracy=best_acc, result=row)
    nonconverged = [r["k"] for r in results if r["fit"].get("converged") is False]
    return {"best_k": best_k, "best_validation_accuracy": best_acc, "results": results,
            "features": ranked_names[:best_k], "nonconverged_k": nonconverged,
            "selection_converged": not nonconverged}


def select_features(train: PreparedSplit, val: PreparedSplit, kind: str,
                    *, progress=None, checkpoint_dir: Path | None = None) -> dict:
    require_selection_roles(train, val)
    emit(progress, "mi_started", train_rows=len(train.y), features=380)
    mi = select_mi(train)
    if checkpoint_dir is not None:
        mi["scores"].to_csv(checkpoint_dir / "mi_feature_scores.csv")
        write_json(checkpoint_dir / "mi_top100.json", mi["names"])
    emit(progress, "mi_completed", **mi["measurement"], top_features=100)
    emit(progress, "top100_scaling_started")
    X_train, X_val, scaler = scale_top100(train, val, mi)
    emit(progress, "rfe_started", target=1, step=1, expected_forest_fits=100, branch=kind)
    ranking = rank_rfe(X_train, train.y, mi["names"], kind)
    if checkpoint_dir is not None:
        ranking["ranks"].to_csv(checkpoint_dir / "rfe_feature_ranking.csv")
    emit(progress, "rfe_completed", **ranking["measurement"], forest_fits=ranking["forest_fits"])
    selection = search_k(X_train, train.y, X_val, val.y, mi["names"], ranking["names"], kind,
                         progress=progress)
    return {"mi": mi, "top100_scaler": scaler, "ranking": ranking, **selection}


def fit_final_classifier(train: PreparedSplit, val: PreparedSplit,
                         kind: str, selected_names=None, *, baseline: bool = False) -> dict:
    require_selection_roles(train, val)
    names = list(FEATURE_NAMES) if baseline else list(selected_names or [])
    if not names or len(names) != len(set(names)) or not set(names) <= set(FEATURE_NAMES):
        raise ValueError("Invalid frozen feature set")
    indices = [FEATURE_NAMES.index(n) for n in names]
    # No MI/RFE here: frozen columns only, exactly source Train+Val refit semantics.
    combined = np.empty((len(train.y) + len(val.y), len(indices)), dtype=np.float32)
    offset = 0
    for data in (train, val):
        for start in range(0, len(data.y), 8192):
            end = min(start + 8192, len(data.y))
            np.take(data.X[start:end], indices, axis=1, out=combined[offset:offset + end - start])
            offset += end - start
    y_combined = np.concatenate([train.y, val.y])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    del combined
    model, fit = timed_fit(make_model(kind, baseline=baseline), combined_scaled, y_combined)
    del combined_scaled
    return {"model": model, "scaler": scaler, "features": names, "fit": fit,
            "fit_rows": len(y_combined), "fit_role": "train+val", "mi_rfe_refit": False}


def evaluate_final(fitted: dict, test: PreparedSplit) -> dict:
    test.validate()
    if test.role != "test":
        raise ValueError("Final evaluation requires a Test split")
    indices = [FEATURE_NAMES.index(n) for n in fitted["features"]]
    test_scaled = fitted["scaler"].transform(test.X[:, indices])
    start = time.perf_counter()
    result_metrics = metrics(test.y, fitted["model"].predict(test_scaled))
    return {**fitted,
            "predict_seconds": time.perf_counter() - start, "metrics": result_metrics,
            "test_rows_evaluated": len(test.y), "test_evaluation_calls": 1}


def refit_final(train: PreparedSplit, val: PreparedSplit, test: PreparedSplit,
                kind: str, selected_names=None, *, baseline: bool = False) -> dict:
    """Compatibility helper; official orchestration defers loading Test until fit."""
    test.validate()
    if test.role != "test":
        raise ValueError("Final evaluation requires a Test split")
    fitted = fit_final_classifier(train, val, kind, selected_names, baseline=baseline)
    return evaluate_final(fitted, test)


@dataclass(frozen=True)
class WindowFile:
    path: Path
    role: str
    scenario: str
    source_id: int
    count: int
    label: int
    segment: sw.Segment | None = None
    fw: int = 0
    fs: int = 0


class WindowStore:
    def __init__(self, root: Path, split_root: Path | None = None):
        self.root = root
        self.summary = json.loads((root / "window_run_summary.json").read_text(encoding="utf-8"))
        s = self.summary
        if s["status"] != "PASS" or s["feature_columns"] != 380 or s["output_columns"] != 394:
            raise ValueError("Window generation has not passed the canonical schema audit")
        if s["readback_validation"]["verified_rows"] != 6_737_006:
            raise ValueError("Unexpected full window count")
        self.segments = sw.load_manifest(split_root or root.parent / "CSV_SPLIT_CPLUS")
        self.files = {}
        for fw, fs in sw.WINDOW_CONFIGS:
            config = f"FW_{fw}_FS_{fs}"
            files = []
            for segment in self.segments:
                count = sw.window_count(segment.flow_count, fw, fs)
                if count:
                    path = sw._expected_window_output_path(root, segment, fw, fs)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    files.append(WindowFile(path, segment.split, segment.scenario,
                        segment.source_csv_id, count, int(segment.proxy_label == "Attack"), segment, fw, fs))
            self.files[config] = sorted(files, key=lambda f: (f.role, f.source_id, str(f.path)))
            for role in ("train", "val", "test"):
                if sum(f.count for f in files if f.role == role) != s["window_counts"][f"FW{fw}_FS{fs}"][role]:
                    raise ValueError("Manifest window count mismatch")
        if {f.path.resolve() for files in self.files.values() for f in files} != {
                p.resolve() for p in root.rglob("*_windows.csv")}:
            raise ValueError("Window file inventory differs from the manifest")

    def snapshot(self):
        return {str(f.path.relative_to(self.root)): {"bytes": f.path.stat().st_size,
                "mtime_ns": f.path.stat().st_mtime_ns}
                for files in self.files.values() for f in files}

    @staticmethod
    def validate_header(path: Path):
        with path.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        if tuple(header) != (*sw.OUTPUT_METADATA_COLUMNS, *FEATURE_NAMES):
            raise ValueError(f"Unexpected 394-column CSV schema: {path}")

    @staticmethod
    def validate_chunk(frame: pd.DataFrame, file: WindowFile, offset: int):
        if file.segment is None:
            raise ValueError("Validated source segment metadata is required")
        expected_index = np.arange(offset, offset + len(frame), dtype=np.int64)
        indices = pd.to_numeric(frame["Window_Index"], errors="raise").to_numpy()
        expected_ids = [sw._window_id(file.segment, file.fw, file.fs, int(i)) for i in expected_index]
        if not np.array_equal(indices, expected_index) or frame["Window_ID"].astype(str).tolist() != expected_ids:
            raise ValueError("Window_Index/Window_ID duplicate, gap, reorder or wrong segment")
        if not ((frame["Split"] == file.role).all()
                and (frame["Scenario"] == file.scenario).all()
                and (frame["Source_CSV_ID"] == f"sid{file.source_id:03d}").all()
                and (frame["Source_CSV"].astype(str).str.casefold() == file.segment.source_csv.casefold()).all()
                and (frame["Label_Origin"] == "SCENARIO_PROXY").all()
                and (frame["Proxy_Label"] == ("Attack" if file.label else "Benign")).all()
                and (frame["Label"] == file.label).all()
                and (frame["Num_Flows"] == file.fw).all()):
            raise ValueError("Window split/source/scenario/proxy provenance mismatch")
        for col in ("Original_Row_Index_Start", "Original_Row_Index_End"):
            values = frame[col].astype(str).str.extract(r"^row:(\d+)$", expand=False)
            if values.isna().any():
                raise ValueError("Malformed original-row provenance")
            raw = values.to_numpy(dtype=np.int64)
            if (raw < file.segment.min_original_row_index).any() or (raw > file.segment.max_original_row_index).any():
                raise ValueError("Original row is outside the manifest segment")
        if frame[["Window_Start_Timestamp", "Window_End_Timestamp"]].isna().any().any():
            raise ValueError("Missing timestamp provenance")

    def load_sample(self, config: str, role: str, rows: int, *, stride: int = BENCHMARK_STRIDE):
        if role not in ("train", "val"):
            raise ValueError("Stage5.1 benchmark must not read Test CSV data")
        if not 1 <= rows <= (8192 if role == "train" else 2048):
            raise ValueError("Technical benchmark sample cap exceeded")
        files = [f for f in self.files[config] if f.role == role]
        quotas = proportional_quotas([f.count for f in files], rows)
        matrices, labels, lineage = [], [], []
        parsed_rows, cleaning = 0, {"nan_replaced": 0, "inf_replaced": 0}
        clean_seconds = 0.0
        start = time.perf_counter()
        with PeakRSS() as ram:
            for file, quota in zip(files, quotas):
                if not quota:
                    continue
                self.validate_header(file.path)
                end = min(file.count, quota * stride)
                positions = np.linspace(0, end - 1, quota, dtype=np.int64)
                offset = 0
                for frame in pd.read_csv(file.path, nrows=int(positions[-1]) + 1,
                                         chunksize=4096, low_memory=False):
                    self.validate_chunk(frame, file, offset)
                    lo, hi = np.searchsorted(positions, [offset, offset + len(frame)])
                    chosen = frame.iloc[positions[lo:hi] - offset]
                    parsed_rows += len(frame)
                    if len(chosen):
                        if not ((chosen["Split"] == role).all()
                                and (chosen["Label_Origin"] == "SCENARIO_PROXY").all()
                                and (chosen["Source_CSV_ID"] == f"sid{file.source_id:03d}").all()
                                and (chosen["Label"] == file.label).all()):
                            raise ValueError("Sample metadata/label disagrees with manifest")
                        clean_start = time.perf_counter()
                        data = prepare_xy(chosen, role)
                        clean_seconds += time.perf_counter() - clean_start
                        matrices.append(data.X)
                        labels.append(data.y)
                        for key in cleaning:
                            cleaning[key] += data.cleaning[key]
                        lineage.extend({"Config": config, "Split": role, "Scenario": file.scenario,
                                        "Source_CSV_ID": file.source_id, "Window_ID": str(row.Window_ID),
                                        "Window_Index": int(row.Window_Index), "Label": int(row.Label)}
                                       for row in chosen.itertuples())
                    offset += len(frame)
            result = PreparedSplit(role, np.concatenate(matrices), np.concatenate(labels), cleaning=cleaning)
            result.validate(both_classes=True)
            if len(result.y) != rows or len({r["Window_ID"] for r in lineage}) != rows:
                raise ValueError("Sample count or Window_ID uniqueness failed")
        info = {"seconds": time.perf_counter() - start, "peak_rss_bytes": ram.peak,
                "rows_retained": rows, "rows_parsed": parsed_rows,
                "benign": int((result.y == 0).sum()), "attack": int((result.y == 1).sum()),
                "cleaning": cleaning, "sample_method": "deterministic per-source proportional quotas; sparse prefix stride=8",
                "clean_prepare_seconds": clean_seconds,
                "scientific_result": False}
        return result, info, lineage

    def load_full(self, config: str, role: str, cache: Path, *, progress=None) -> PreparedSplit:
        """Future official path: every row, streaming float32 memmap; no sampling."""
        files = [f for f in self.files[config] if f.role == role]
        n = sum(f.count for f in files)
        emit(progress, "split_load_started", split=role, expected_rows=n, files=len(files))
        X = np.lib.format.open_memmap(cache / f"{role}_X.npy", mode="w+", dtype=np.float32, shape=(n, 380))
        y = np.lib.format.open_memmap(cache / f"{role}_y.npy", mode="w+", dtype=np.int8, shape=(n,))
        offset = 0
        for file in files:
            self.validate_header(file.path)
            count = 0
            for frame in pd.read_csv(file.path, chunksize=8192, low_memory=False):
                self.validate_chunk(frame, file, count)
                data = prepare_xy(frame, role)
                if not (data.y == file.label).all():
                    raise ValueError("Full-loader proxy labels differ from manifest")
                end = offset + len(data.y)
                if end > n:
                    raise ValueError("Full-loader row overflow")
                X[offset:end], y[offset:end] = data.X, data.y
                offset = end
                count += len(data.y)
            if count != file.count:
                raise ValueError("Full-loader rows differ from manifest")
            emit(progress, "source_file_loaded", split=role, source_id=file.source_id,
                 source_rows=count, loaded_rows=offset, expected_rows=n)
        if offset != n:
            raise ValueError("Full-loader did not preserve every window")
        X.flush()
        y.flush()
        result = PreparedSplit(role, X, y)
        result.validate()
        emit(progress, "split_load_completed", split=role, rows=n, features=380,
             benign=int((y == 0).sum()), attack=int((y == 1).sum()), finite=True)
        return result


def proportional_quotas(weights, n):
    weights = np.asarray(weights, dtype=np.int64)
    if n > int(weights.sum()) or n < 0:
        raise ValueError("Sample quota exceeds available rows")
    if n == 0:
        return [0] * len(weights)
    exact = weights / weights.sum() * n
    quotas = np.floor(exact).astype(int)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[:n - int(quotas.sum())]:
        quotas[i] += 1
    return quotas.tolist()


def smaller_sample(data: PreparedSplit, rows: int, lineage=None, *, return_indices=False):
    if rows > len(data.y) or rows < 2:
        raise ValueError("Invalid nested sample size")
    groups = np.asarray([r["Source_CSV_ID"] for r in lineage]) if lineage is not None else data.y
    if len(groups) != len(data.y):
        raise ValueError("Sample lineage length mismatch")
    group_ids = np.unique(groups)
    available = [np.flatnonzero(groups == group) for group in group_ids]
    # Keep every represented source when possible, then allocate proportionally.
    minimum = int(rows >= len(available))
    capacities = [len(idx) - minimum for idx in available]
    extra = proportional_quotas(capacities, rows - minimum * len(available))
    quotas = [q + minimum for q in extra]
    indices = np.sort(np.concatenate([idx[np.linspace(0, len(idx) - 1, quota, dtype=np.int64)]
                                      for idx, quota in zip(available, quotas) if quota]))
    result = PreparedSplit(data.role, data.X[indices], data.y[indices])
    result.validate(both_classes=True)
    return (result, indices) if return_indices else result


def _benchmark_worker_impl(root: Path, config: str, output: Path) -> dict:
    store = WindowStore(root)
    train, train_load, train_ids = store.load_sample(config, "train", 8192)
    val, val_load, val_ids = store.load_sample(config, "val", BENCHMARK_VAL_ROWS)
    result = {"status": "RUNNING", "config": config, "technical_benchmark_only": True,
              "test_csv_rows_read": 0, "load": {"train": train_load, "val": val_load}, "samples": []}
    write_json(output, result)
    for size in BENCHMARK_TRAIN_SIZES:
        sample, selected_indices = smaller_sample(train, size, train_ids, return_indices=True)
        for i in selected_indices:
            train_ids[int(i)][f"Included_Train_n{size}"] = True
        pd.DataFrame(train_ids + val_ids).fillna(False).to_csv(output.with_suffix(".sample_ids.csv"), index=False)
        row = {"train_rows": size, "val_rows": len(val.y),
               "benign_train": int((sample.y == 0).sum()), "attack_train": int((sample.y == 1).sum()),
               "source_count": len({train_ids[int(i)]["Source_CSV_ID"] for i in selected_indices}),
               "scenario_count": len({train_ids[int(i)]["Scenario"] for i in selected_indices}), "branches": {}}
        result["samples"].append(row)
        # Technical baseline fits Train and predicts Validation. Official baseline
        # fits Train+Val once, as tested in refit_final; no choice is made here.
        scaler = StandardScaler()
        X_train, X_val = scaler.fit_transform(sample.X), scaler.transform(val.X)
        model, fit = timed_fit(make_model("svm", baseline=True), X_train, sample.y)
        start = time.perf_counter()
        row["baseline_svm"] = {"fit": fit, "metrics": metrics(val.y, model.predict(X_val)),
                               "predict_seconds": time.perf_counter() - start}
        write_json(output, result)
        print(f"{config} n={size} baseline SVM {fit['seconds']:.3f}s", flush=True)
        mi = select_mi(sample)
        row["mi"] = mi["measurement"]
        write_json(output, result)
        print(f"{config} n={size} MI-380 {mi['measurement']['seconds']:.3f}s", flush=True)
        X_top, X_val_top, _ = scale_top100(sample, val, mi)
        for kind in ("svm", "rf"):
            rank = rank_rfe(X_top, sample.y, mi["names"], kind,
                            target=100 - BENCHMARK_RFE_REMOVALS)
            search = search_k(X_top, sample.y, X_val_top, val.y,
                              mi["names"], rank["names"], kind, BENCHMARK_K)
            row["branches"][kind] = {"rfe": {"measurement": rank["measurement"],
                "target": rank["target"], "forest_fits": rank["forest_fits"],
                "complete_ranking": False}, "k_results": search["results"],
                "official_best_k": None,
                "nonconverged_k": search["nonconverged_k"],
                "note": "Partial RFE leaves 97 tied ranks; these k fits are technical timing only"}
            write_json(output, result)
            print(f"{config} n={size} {kind} partial-RFE {rank['measurement']['seconds']:.3f}s; k={BENCHMARK_K}", flush=True)
        write_json(output, result)
    result["status"] = "PASS_TECHNICAL_ONLY"
    write_json(output, result)
    return result


def benchmark_worker(root: Path, config: str, output: Path) -> dict:
    start = time.perf_counter()
    with PeakRSS() as ram:
        result = _benchmark_worker_impl(root, config, output)
    result["worker_peak_rss_bytes_sampled"] = ram.peak
    result["worker_total_seconds"] = time.perf_counter() - start
    write_json(output, result)
    return result


def environment() -> dict:
    mem = psutil.virtual_memory()
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_total_bytes": mem.total,
            "ram_available_bytes": mem.available, "platform": platform.platform()}


def resource_plan(store: WindowStore) -> dict:
    plans = {}
    for config in CONFIGS:
        n = {role: sum(f.count for f in store.files[config] if f.role == role)
             for role in ("train", "val", "test")}
        nfit = n["train"] + n["val"]
        raw = sum(n.values()) * 380 * 4
        # libsvm validates float64 C-contiguous data; cache_size is unchanged.
        plans[config] = {"windows": n, "raw_float32_all_splits_bytes": raw,
            "baseline_final_float32_bytes": nfit * 380 * 4,
            "baseline_libsvm_float64_fit_bytes": nfit * 380 * 8,
            "svm_kernel_cache_cap_bytes": 10000 * 1024**2,
            "baseline_ram_scenario_bytes": raw + 2 * nfit * 380 * 4 + nfit * 380 * 8 + 10000 * 1024**2,
            "mi_train_float64_copy_bytes": n["train"] * 380 * 8,
            "mi_selection_array_scenario_bytes": (4 * n["train"] + n["val"]) * 380 * 4,
            "top100_train_val_float32_bytes": nfit * 100 * 4,
            "rfe_forests_per_branch": 100, "rfe_trees_per_branch": 10000,
            "classifier_validation_fits_per_branch": 100, "rf_classifier_trees_search_plus_final": 30300,
            "kernel_cache_is_cap_not_upfront_allocation": True}
    return plans


def official_memory_preflight(store: WindowStore, config: str, experiment: str) -> dict:
    """Conservative admission rule, not a claim of the actual allocated peak.

    The source's 10,000-MB cache cap is retained. Exceeding this budget requires
    reviewing hardware/resources before a full run, never automatic sampling.
    """
    plan = resource_plan(store)[config]
    n = plan["windows"]
    nfit = n["train"] + n["val"]
    available = psutil.virtual_memory().available
    overhead = 512 * 1024**2
    if experiment == "baseline_svm":
        scenario = plan["raw_float32_all_splits_bytes"] + nfit * 380 * (4 + 8 + 8) + 10000 * 1024**2 + overhead
    elif experiment == "mi_rfe_svm":
        scenario = max(plan["mi_selection_array_scenario_bytes"],
                       plan["raw_float32_all_splits_bytes"] + nfit * 100 * (4 + 8 + 8) + 10000 * 1024**2) + overhead
    else:
        # Forest buffers depend on data; use array footprint plus headroom as an
        # admission floor. Benchmark tree estimates must still be reviewed.
        scenario = plan["mi_selection_array_scenario_bytes"] + 1024**3
    budget = int(available * .85)
    result = {"estimated_scenario_bytes": scenario, "available_ram_bytes": available,
              "admission_budget_bytes": budget, "cache_is_cap_not_fixed_allocation": True,
              "unknown_forest_and_support_vector_memory": True}
    if scenario > budget:
        raise MemoryError(f"Full-run RAM preflight requires resource review: scenario={scenario:,}, "
                          f"85%-available budget={budget:,}. No samples or parameters were changed.")
    return result


def projections(store: WindowStore, workers: list[dict]) -> dict:
    """Conditional scaling scenarios, not promises of full runtime."""
    result = {}
    for worker in workers:
        config = worker["config"]
        if worker.get("status") != "PASS_TECHNICAL_ONLY":
            result[config] = {"status": "UNRESOLVED_BENCHMARK_TIMEOUT"}
            continue
        s = worker["samples"][-1]
        n = {role: sum(f.count for f in store.files[config] if f.role == role)
             for role in ("train", "val", "test")}
        sample_n = s["train_rows"]
        r = n["train"] / sample_n
        rf = (n["train"] + n["val"]) / sample_n
        log_r = r * math.log2(n["train"]) / math.log2(sample_n)
        mi = s["mi"]["seconds"]
        baseline = s["baseline_svm"]["fit"]["seconds"]
        load_scenario = 0.0
        for role in ("train", "val", "test"):
            measured = worker["load"]["val" if role == "test" else role]
            clean = measured.get("clean_prepare_seconds", 0.0)
            load_scenario += ((measured["seconds"] - clean) * n[role] / measured["rows_parsed"]
                              + clean * n[role] / measured["rows_retained"])
        p = {"status": "CONDITIONAL_EXTRAPOLATION_NOT_VALIDATED_FULL",
             "mi_seconds_linear_to_nlogn": [mi * r, mi * log_r],
             "baseline_svm_final_seconds_n2_to_n3": ([baseline * rf**2, baseline * rf**3]
                 if s["baseline_svm"]["fit"].get("converged", True) else None),
             "csv_parse_validate_clean_seconds_linear_scenario": load_scenario,
             "csv_projection_excludes_full_memmap_write_time": True,
             "svm_time_not_reliable_if_small_sample_convergence_or_support_vector_fraction_changes": True,
             "partial_rfe_100_fit_projection_uncertain": True}
        for kind in ("svm", "rf"):
            branch = s["branches"][kind]
            partial = branch["rfe"]["measurement"]["seconds"]
            rows = branch["k_results"]
            mean = sum(row["fit"]["seconds"] for row in rows) / len(rows)
            p[f"{kind}_rfe_seconds_scenario"] = [partial / 4 * 100 * r * .4,
                                                    partial / 4 * 100 * log_r * 1.5]
            p[f"{kind}_rfe_single_forest_buffers_bytes_linear_proxy"] = branch["rfe"]["measurement"].get("tree_buffers_bytes", 0) * r
            if kind == "svm":
                converged = not any(row["fit"].get("converged") is False for row in rows)
                p["compact_svm_k_search_seconds_n2_to_n3"] = [100 * mean * r**2, 100 * mean * r**3] if converged else None
                final = rows[-1]["fit"]["seconds"]
                p["compact_svm_final_k100_seconds_n2_to_n3"] = [final * rf**2, final * rf**3] if converged else None
                p["svm_projection_suppressed_due_to_nonconvergence"] = not converged
            else:
                p["rf_k_search_seconds_linear_to_nlogn"] = [100 * mean * r, 100 * mean * log_r]
                p["rf_final_k100_seconds_linear_to_nlogn"] = [rows[-1]["fit"]["seconds"] * rf,
                    rows[-1]["fit"]["seconds"] * rf * math.log2(n["train"] + n["val"]) / math.log2(sample_n)]
                p["rf_final_k100_tree_buffers_bytes_linear_proxy"] = rows[-1]["fit"].get("tree_buffers_bytes", 0) * rf
        result[config] = p
    return result


def run_benchmark(root: Path, original: Path, out: Path, *, timeout: int = 240) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    if not 30 <= timeout <= 600:
        raise ValueError("Benchmark worker timeout must be between 30 and 600 seconds")
    store = WindowStore(root)
    before, scripts = store.snapshot(), source_hashes(original)
    out.mkdir(parents=True)
    audit = {"environment": environment(), "original_hashes": scripts,
             "source_parameters": source_audit(original), "resource_plan": resource_plan(store),
             "benchmark_limits": {"train_rows": list(BENCHMARK_TRAIN_SIZES), "validation_rows": BENCHMARK_VAL_ROWS,
                 "test_data_rows_read": 0, "rfe_removals": BENCHMARK_RFE_REMOVALS,
                 "k_values": list(BENCHMARK_K), "max_seconds_per_config": timeout},
             "window_snapshot_before": before}
    write_json(out / "audit.json", audit)
    workers = []
    for config in CONFIGS:
        target = out / f"{config}.json"
        print(f"BENCHMARK {config}: Train <=8192, Val=2048; Test=0", flush=True)
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_benchmark-worker",
                                 "--windows", str(root), "--config", config, "--output", str(target)])
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            proc.wait()
            code = -1
        worker = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"config": config}
        if code:
            worker["status"] = "TIMEOUT_OR_WORKER_FAILURE"
            worker["returncode"] = code
            write_json(target, worker)
        workers.append(worker)
    if store.snapshot() != before or source_hashes(original) != scripts:
        raise ValueError("Window stat snapshot or original SHA256 changed during benchmark")
    result = {"status": "PASS" if all(w["status"] == "PASS_TECHNICAL_ONLY" for w in workers) else "PARTIAL_BENCHMARK",
              "technical_benchmark_only": True, "models_saved": False, "full_training_executed": False,
              "original_sha256_before_after_match": True, "window_size_mtime_before_after_match": True,
              "workers": workers, "projections": projections(store, workers), "resource_plan": resource_plan(store)}
    write_json(out / "benchmark_summary.json", result)
    print(f"BENCHMARK RESULT: {result['status']}; {out}", flush=True)
    return result


def run_official(root: Path, original: Path, out: Path, config: str, experiment: str,
                 *, confirm_full_training: bool = False,
                 official_authorization=None) -> dict:
    if not confirm_full_training:
        raise PermissionError("Full training requires explicit approval and --confirm-full-training")
    if config not in CONFIGS or experiment not in ("baseline_svm", "mi_rfe_svm", "mi_rfe_rf"):
        raise ValueError("Unknown official configuration/experiment")
    authorization_record = None
    if experiment in ("baseline_svm", "mi_rfe_svm"):
        # Enforced below every launcher/CLI so --confirm-full-training alone can
        # never bypass Stage 7.3B approval.
        from colab.svm_run_authorization import validate_context_for_training
        authorized = validate_context_for_training(
            official_authorization, config, experiment, out)
        authorization_record = authorized.public_record()
    if out.exists() or out.with_name(out.name + ".incomplete").exists():
        raise FileExistsError("Refusing to overwrite an official result or incomplete run")
    store = WindowStore(root)
    before, scripts = store.snapshot(), source_hashes(original)
    source_audit(original)
    memory_admission = official_memory_preflight(store, config, experiment)
    plan = resource_plan(store)[config]
    for file in store.files[config]:
        if file.role in ("train", "val"):
            store.validate_header(file.path)
    anchor = out.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if psutil.disk_usage(str(anchor)).free < math.ceil(plan["raw_float32_all_splits_bytes"] * 1.5):
        raise OSError("Insufficient free space for complete matrices, model and reserve")
    staged = out.with_name(out.name + ".incomplete")
    staged.mkdir(parents=True)
    cache = staged / "matrices"
    cache.mkdir()
    scientific_contract = official_hyperparameters(experiment)
    feature_schema = {
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "excluded_from_X": list(sw.OUTPUT_METADATA_COLUMNS),
        "nonfinite_policy": "replace NaN and +/-Inf with 0 before float32 fit",
        "label_column": "Label", "label_origin": "SCENARIO_PROXY",
    }
    write_json(staged / "hyperparameters.json", scientific_contract)
    write_json(staged / "feature_schema.json", feature_schema)
    if authorization_record is not None:
        write_json(staged / "authorization.json", authorization_record)
    write_json(staged / "preflight_audit.json", {
        "config": config, "experiment": experiment, "environment": environment(),
        "memory_admission": memory_admission, "disk_free_bytes": psutil.disk_usage(str(anchor)).free,
        "expected_windows": plan["windows"], "feature_names": list(FEATURE_NAMES),
        "original_sha256": scripts, "window_stat_snapshot_before": before,
        "implementation_sha256": sha256(Path(__file__)), "single_experiment_single_config": True,
        "test_csv_first_read_after_final_fit": True, "resume_supported": False,
        "incomplete_checkpoints_are_not_official_results": True,
        "stage73b_authorization": authorization_record,
        "scientific_contract": scientific_contract,
        "feature_schema_sha256": sha256(staged / "feature_schema.json")})
    start = time.perf_counter()
    try:
        with PeakRSS() as total_ram, RunProgress(staged, config, experiment) as progress:
            train = store.load_full(config, "train", cache, progress=progress)
            val = store.load_full(config, "val", cache, progress=progress)
            baseline = experiment == "baseline_svm"
            kind = "rf" if experiment == "mi_rfe_rf" else "svm"
            if baseline:
                selected = list(FEATURE_NAMES)
                nonconverged_k = []
                selection_measurements = {}
            else:
                selection = select_features(train, val, kind, progress=progress, checkpoint_dir=staged)
                selected = selection["features"]
                nonconverged_k = selection["nonconverged_k"]
                selection_measurements = {"mi": selection["mi"]["measurement"],
                                          "rfe": selection["ranking"]["measurement"],
                                          "best_k": selection["best_k"],
                                          "best_validation_accuracy": selection["best_validation_accuracy"]}
                write_json(staged / "k_search_results.json", selection["results"])
            write_json(staged / "selection_frozen.json", {"best_k": len(selected) if baseline else selection["best_k"],
                       "features": selected, "nonconverged_k": nonconverged_k,
                       "criterion": ("fixed_380_baseline" if baseline else
                                     "validation_accuracy_strict_greater_smallest_k_tie")})
            progress("selection_frozen", features=len(selected), **selection_measurements)
            progress("final_refit_started", train_val_rows=len(train.y) + len(val.y), features=len(selected))
            fitted = fit_final_classifier(train, val, kind, selected, baseline=baseline)
            progress("final_refit_completed", **fitted["fit"])
            # No Test CSV data is read until selection AND the Train+Val fit finish.
            test = store.load_full(config, "test", cache, progress=progress)
            progress("test_evaluation_started", rows=len(test.y), evaluation_number=1)
            final = evaluate_final(fitted, test)
            progress("test_evaluation_completed", **final["metrics"])
            joblib.dump(final["model"], staged / "model.joblib")
            joblib.dump(final["scaler"], staged / "final_scaler.joblib")
            write_json(staged / "features.json", selected)
            result = {k: v for k, v in final.items() if k not in ("model", "scaler")}
            result.update(config=config, experiment=experiment,
                          status=("COMPLETED_WITH_CONVERGENCE_WARNING" if nonconverged_k or final["fit"].get("converged") is False else "PASS"),
                          nonconverged_selection_k=nonconverged_k, memory_admission=memory_admission,
                          selection_measurements=selection_measurements,
                          rows_by_split={"train": len(train.y), "val": len(val.y), "test": len(test.y)},
                          label_counts_by_split={d.role: {"benign": int((d.y == 0).sum()),
                                                         "attack": int((d.y == 1).sum())}
                                                 for d in (train, val, test)},
                          source_hashes_match=source_hashes(original) == scripts,
                          window_stat_snapshot_match=store.snapshot() == before,
                          label_origin="SCENARIO_PROXY", capture_session_independence="NOT_ESTABLISHED",
                          test_read_after_final_refit=True, sampled_training=False,
                          authorization=authorization_record,
                          hyperparameters=scientific_contract,
                          feature_schema_sha256=sha256(staged / "feature_schema.json"),
                          stage_runtimes_seconds={
                              "mi": selection_measurements.get("mi", {}).get("seconds"),
                              "rfe": selection_measurements.get("rfe", {}).get("seconds"),
                              "final_refit": final["fit"].get("seconds"),
                              "test_predict": final.get("predict_seconds"),
                          },
                          support_vectors=final["fit"].get("support_vectors"),
                          fit_warnings=final["fit"].get("warnings", []))
            if not result["source_hashes_match"] or not result["window_stat_snapshot_match"]:
                raise ValueError("Input/original integrity check failed")
            if result["rows_by_split"] != plan["windows"]:
                raise ValueError("Official split row counts differ from the manifest")
            if getattr(progress, "logging_error", None):
                raise OSError(f"Resource logging failed: {progress.logging_error}")
            total_ram._sample()
            result.update(peak_rss_bytes_sampled=total_ram.peak,
                          elapsed_seconds=time.perf_counter() - start)
            write_json(staged / "result.json", result)
            progress("run_validated", status=result["status"], **result["rows_by_split"])
        # Close memory maps before publishing on Windows; preserve matrix cache.
        for data in (train, val, test):
            for array in (data.X, data.y):
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        os.replace(staged, out)
        return result
    except BaseException as exc:
        write_json(staged / "FAILED.json", {"status": "INCOMPLETE_DO_NOT_USE", "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    bench.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--timeout-per-config", type=int, default=240)
    worker = sub.add_parser("_benchmark-worker")
    worker.add_argument("--windows", type=Path, required=True)
    worker.add_argument("--config", choices=CONFIGS, required=True)
    worker.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("train", help="One approved experiment/configuration only; no implicit pipeline loop")
    run.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    run.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", choices=CONFIGS, required=True)
    run.add_argument("--experiment", choices=("baseline_svm", "mi_rfe_svm", "mi_rfe_rf"), required=True)
    run.add_argument("--confirm-full-training", action="store_true")
    args = p.parse_args()
    if args.command == "benchmark":
        run_benchmark(args.windows, args.original, args.output, timeout=args.timeout_per_config)
    elif args.command == "_benchmark-worker":
        benchmark_worker(args.windows, args.config, args.output)
    else:
        if args.experiment in ("baseline_svm", "mi_rfe_svm"):
            raise PermissionError(
                "Official SVM cannot run from the direct training CLI; use the "
                "Stage 7.3B supervisor with a run-specific approval JSON")
        run_official(args.windows, args.original, args.output, args.config, args.experiment,
                     confirm_full_training=args.confirm_full_training)


if __name__ == "__main__":
    main()
