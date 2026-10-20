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

ROOT = Path(r"D:\COMPACT-MI-RFE")
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
                 *, confirm_full_training: bool = False) -> dict:
    if not confirm_full_training:
        raise PermissionError("Full training requires explicit approval and --confirm-full-training")
    if config not in CONFIGS or experiment not in ("baseline_svm", "mi_rfe_svm", "mi_rfe_rf"):
        raise ValueError("Unknown official configuration/experiment")
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
    write_json(staged / "preflight_audit.json", {
        "config": config, "experiment": experiment, "environment": environment(),
        "memory_admission": memory_admission, "disk_free_bytes": psutil.disk_usage(str(anchor)).free,
        "expected_windows": plan["windows"], "feature_names": list(FEATURE_NAMES),
        "original_sha256": scripts, "window_stat_snapshot_before": before,
        "implementation_sha256": sha256(Path(__file__)), "single_experiment_single_config": True,
        "test_csv_first_read_after_final_fit": True, "resume_supported": False,
        "incomplete_checkpoints_are_not_official_results": True})
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
                          test_read_after_final_refit=True, sampled_training=False)
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
        run_official(args.windows, args.original, args.output, args.config, args.experiment,
                     confirm_full_training=args.confirm_full_training)


# This module is an isolated technical laboratory. The original main() above is
# retained only to make the diff auditable; it is never exposed by this CLI.
import ciciomt24_training as reference

MEMORYLAB_ROOT = ROOT / "huyen_code" / "runs" / "stage55_memorylab"
MEMORYLAB_CONFIG = "FW_512_FS_16"
MEMORYLAB_TRAIN_ROWS = 8192
MEMORYLAB_VAL_ROWS = 2048
MEMORYLAB_MODES = ("baseline", "E1", "E2", "E3", "E4", "E1_E2_E3_E4")


def run_official(*args, **kwargs):
    """An experimental module must never publish an official model result."""
    raise PermissionError("Memorylab cannot run official training")


def ml_array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def ml_scaler_stats(scaler: StandardScaler) -> dict:
    return {key: np.asarray(getattr(scaler, key)).tolist()
            for key in ("mean_", "var_", "scale_")}


def ml_np_random_state_sha() -> str:
    state = np.random.get_state()
    h = hashlib.sha256()
    h.update(state[0].encode("ascii"))
    h.update(state[1].tobytes())
    h.update(str(state[2:]).encode("ascii"))
    return h.hexdigest().upper()


class MLResourceTrace:
    """50 ms sampled process RSS/private commit and page-fault counters."""
    def __enter__(self):
        self.proc = psutil.Process()
        self.stop = threading.Event()
        self.peak_rss_bytes = 0
        self.peak_private_bytes = None
        self.first_faults = None
        self.last_faults = None
        self.available_before_bytes = psutil.virtual_memory().available
        self.started = time.perf_counter()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        rss, private = 0, 0
        faults = None
        for proc in [self.proc, *self.proc.children(recursive=True)]:
            try:
                info = proc.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss += info.rss
            if hasattr(info, "private"):
                private += info.private
            if proc.pid == self.proc.pid:
                faults = getattr(info, "num_page_faults", None)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        if private:
            self.peak_private_bytes = max(self.peak_private_bytes or 0, private)
        if faults is not None:
            if self.first_faults is None:
                self.first_faults = faults
            self.last_faults = faults

    def _loop(self):
        while not self.stop.wait(0.05):
            self._sample()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()
        self.seconds = time.perf_counter() - self.started
        self.available_after_bytes = psutil.virtual_memory().available

    def to_dict(self):
        return {"seconds": self.seconds, "peak_rss_bytes": self.peak_rss_bytes,
                "peak_private_commit_bytes": self.peak_private_bytes,
                "available_before_bytes": self.available_before_bytes,
                "available_after_bytes": self.available_after_bytes,
                "process_page_faults_delta": (None if self.first_faults is None
                    else self.last_faults - self.first_faults),
                "page_faults_include_soft_faults": True,
                "sampling_interval_seconds": 0.05}


def ml_load_sample(path: Path, role: str) -> PreparedSplit:
    if role not in ("train", "val") or path.name != f"{role}_sample.npz":
        raise PermissionError("Memorylab may load Train/Validation sample only")
    with np.load(path, allow_pickle=False) as archive:
        data = PreparedSplit(role, archive["X"], archive["y"])
    data.validate(both_classes=True)
    return data


def ml_scale_top100(train: PreparedSplit, val: PreparedSplit, mi: dict, *,
                    inplace_copy: bool):
    if not inplace_copy:
        X_train, X_val, scaler = reference.scale_top100(train, val, mi)
        return X_train, X_val, scaler, {"enabled": False}
    reference.require_selection_roles(train, val)
    train_copy = train.X[:, mi["indices"]]
    val_copy = val.X[:, mi["indices"]]
    checks = {
        "train_copy_writable": bool(train_copy.flags.writeable),
        "val_copy_writable": bool(val_copy.flags.writeable),
        "train_shares_raw": bool(np.shares_memory(train_copy, train.X)),
        "val_shares_raw": bool(np.shares_memory(val_copy, val.X)),
        "train_dtype": str(train_copy.dtype), "val_dtype": str(val_copy.dtype),
        "train_c_contiguous": bool(train_copy.flags.c_contiguous),
        "val_c_contiguous": bool(val_copy.flags.c_contiguous)}
    if (not checks["train_copy_writable"] or not checks["val_copy_writable"]
            or checks["train_shares_raw"] or checks["val_shares_raw"]
            or train_copy.dtype != np.float32 or val_copy.dtype != np.float32):
        raise ValueError("E3 isolation or dtype guard failed")
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(train_copy)
    X_val = scaler.transform(val_copy)
    checks.update(enabled=True,
        train_transform_reused_copy=bool(np.shares_memory(X_train, train_copy)),
        val_transform_reused_copy=bool(np.shares_memory(X_val, val_copy)))
    if not checks["train_transform_reused_copy"] or not checks["val_transform_reused_copy"]:
        raise ValueError("E3 did not scale the isolated arrays in place")
    return X_train, X_val, scaler, checks


def ml_search_k(X_train, y_train, X_val, y_val, top_names, ranked_names,
                *, release_old_model: bool, ranked_workspace: bool,
                verified_workspace: dict | None = None):
    """Source k-grid and estimator; independent E1 and E4 implementation flags."""
    if len(ranked_names) != TOP_MI or len(top_names) != TOP_MI:
        raise ValueError("Complete MI/RFE ranking is required")
    workspace_checks = verified_workspace or {"enabled": False}
    if ranked_workspace:
        if not workspace_checks.get("all_100_prefixes_equal"):
            raise ValueError("E4 requires prevalidated ranked workspaces")

    rows, best_k, best_acc = [], None, -1.0
    for k in K_GRID:
        if ranked_workspace:
            train_k, val_k = X_train[:, :k], X_val[:, :k]
        else:
            indices = [top_names.index(n) for n in ranked_names[:k]]
            train_k, val_k = X_train[:, indices], X_val[:, indices]
        model, fit = reference.timed_fit(reference.make_model("rf"), train_k, y_train)
        predicted = model.predict(val_k)
        row = {"k": k, **reference.metrics(y_val, predicted), "fit": fit}
        rows.append(row)
        if row["accuracy"] > best_acc:
            best_k, best_acc = k, row["accuracy"]
        if release_old_model:
            del model
        del train_k, val_k, predicted
    return {
        "best_k": best_k, "best_validation_accuracy": best_acc,
        "features": ranked_names[:best_k], "results": rows,
        "workspace_checks": workspace_checks}


def ml_mode_flags(mode: str) -> dict:
    if mode not in MEMORYLAB_MODES:
        raise ValueError("Unknown technical benchmark mode")
    return {flag: mode == flag or mode == "E1_E2_E3_E4"
            for flag in ("E1", "E2", "E3", "E4")}


def ml_worker(output: Path, mode: str) -> dict:
    if output != MEMORYLAB_ROOT or mode not in MEMORYLAB_MODES:
        raise PermissionError("Memorylab worker has one fixed technical output/config")
    flags = ml_mode_flags(mode)
    traces = {}
    start = time.perf_counter()
    with MLResourceTrace() as whole:
        with MLResourceTrace() as phase:
            train = ml_load_sample(output / "train_sample.npz", "train")
        traces["load_train"] = phase.to_dict()
        raw_train_before = {"X": ml_array_sha(train.X), "y": ml_array_sha(train.y)}
        val = None
        if not flags["E2"]:
            with MLResourceTrace() as phase:
                val = ml_load_sample(output / "val_sample.npz", "val")
            traces["load_val"] = phase.to_dict()
        rng_before = ml_np_random_state_sha()
        with MLResourceTrace() as phase:
            mi = reference.select_mi(train)
        traces["mi"] = phase.to_dict()
        rng_after = ml_np_random_state_sha()
        if val is None:
            with MLResourceTrace() as phase:
                val = ml_load_sample(output / "val_sample.npz", "val")
            traces["load_val"] = phase.to_dict()
        raw_val_before = {"X": ml_array_sha(val.X), "y": ml_array_sha(val.y)}
        with MLResourceTrace() as phase:
            X_top, X_val_top, top_scaler, e3_checks = ml_scale_top100(
                train, val, mi, inplace_copy=flags["E3"])
        traces["top100_scaling"] = phase.to_dict()
        top100_hashes = {"train": ml_array_sha(X_top), "val": ml_array_sha(X_val_top)}
        with MLResourceTrace() as phase:
            ranking = reference.rank_rfe(X_top, train.y, mi["names"], "rf")
        traces["rfe"] = phase.to_dict()
        with MLResourceTrace() as phase:
            if flags["E1"] or flags["E4"]:
                workspace_checks = None
                if flags["E4"]:
                    order = [mi["names"].index(n) for n in ranking["names"]]
                    train_ranked = X_top[:, order]
                    val_ranked = X_val_top[:, order]
                    prefix_matches = []
                    for k in K_GRID:
                        indices = [mi["names"].index(n) for n in ranking["names"][:k]]
                        prefix_matches.append(bool(
                            np.array_equal(X_top[:, indices], train_ranked[:, :k])
                            and np.array_equal(X_val_top[:, indices], val_ranked[:, :k])
                            and X_top.dtype == train_ranked.dtype
                            and X_val_top.dtype == val_ranked.dtype))
                    workspace_checks = {"enabled": True,
                        "all_100_prefixes_equal": all(prefix_matches),
                        "first_mismatched_k": next((i + 1 for i, ok in
                            enumerate(prefix_matches) if not ok), None),
                        "sklearn_validation_shares_input": {}}
                    if not all(prefix_matches):
                        raise ValueError("E4 changed a ranked k input")
                    from sklearn.utils.validation import validate_data
                    for k in (1, 50, 100):
                        view = train_ranked[:, :k]
                        forest = RandomForestClassifier(**RF_PARAMS)
                        checked = validate_data(forest, view, dtype=np.float32,
                            accept_sparse="csc", ensure_all_finite=False)
                        workspace_checks["sklearn_validation_shares_input"][str(k)] = bool(
                            np.shares_memory(checked, view))
                    del X_top, X_val_top
                    X_top, X_val_top = train_ranked, val_ranked
                    del train_ranked, val_ranked
                selection = ml_search_k(X_top, train.y, X_val_top, val.y,
                    mi["names"], ranking["names"], release_old_model=flags["E1"],
                    ranked_workspace=flags["E4"], verified_workspace=workspace_checks)
            else:
                selection = reference.search_k(X_top, train.y, X_val_top, val.y,
                    mi["names"], ranking["names"], "rf")
                selection["workspace_checks"] = {"enabled": False}
        traces["k_search"] = phase.to_dict()
        del X_top, X_val_top
        with MLResourceTrace() as phase:
            fitted = reference.fit_final_classifier(train, val, "rf",
                selection["features"])
        traces["refit_train_val_only"] = phase.to_dict()
        refit_stats = ml_scaler_stats(fitted["scaler"])
        del fitted
        raw_train_after = {"X": ml_array_sha(train.X), "y": ml_array_sha(train.y)}
        raw_val_after = {"X": ml_array_sha(val.X), "y": ml_array_sha(val.y)}
        if raw_train_before != raw_train_after or raw_val_before != raw_val_after:
            raise ValueError("Raw Train/Validation sample changed during benchmark")
    rows = [{
        "k": r["k"], "accuracy": r["accuracy"],
        "confusion_matrix": r["confusion_matrix_labels_0_benign_1_attack"],
        "fit_seconds": r["fit"]["seconds"],
        "fit_peak_rss_bytes": r["fit"]["peak_rss_bytes"],
        "forest_tree_buffers_bytes": r["fit"].get("tree_buffers_bytes"),
        "fit_warnings": r["fit"]["warnings"]}
        for r in selection["results"]]
    outcome = {
        "status": "TECHNICAL_BENCHMARK_NOT_OFFICIAL", "mode": mode,
        "flags": flags, "config": MEMORYLAB_CONFIG,
        "sample_rows": {"train": len(train.y), "val": len(val.y), "test": 0},
        "sample_labels": {"train_benign": int((train.y == 0).sum()),
            "val_benign": int((val.y == 0).sum())},
        "raw_sha256_before": {"train": raw_train_before, "val": raw_val_before},
        "raw_sha256_after": {"train": raw_train_after, "val": raw_val_after},
        "mi_global_rng_state_before": rng_before,
        "mi_global_rng_state_after": rng_after,
        "mi_scores_canonical": mi["scores"].reindex(FEATURE_NAMES).tolist(),
        "mi_top100_names": mi["names"], "mi_scaler": ml_scaler_stats(mi["scaler"]),
        "top100_scaler": ml_scaler_stats(top_scaler),
        "top100_scaled_sha256": top100_hashes,
        "E3_isolation": e3_checks,
        "rfe_ranked_names": ranking["names"],
        "rfe_ranks_by_name": {n: int(v) for n, v in ranking["ranks"].items()},
        "rfe_target": ranking["target"], "rfe_forest_fits": ranking["forest_fits"],
        "k_records": rows, "best_k": selection["best_k"],
        "best_validation_accuracy": selection["best_validation_accuracy"],
        "frozen_features": selection["features"],
        "E4_workspace": selection["workspace_checks"],
        "final_refit_scaler": refit_stats,
        "phase_resources": traces,
        "total_resources": whole.to_dict(),
        "elapsed_seconds": time.perf_counter() - start,
        "source_params": {"mi_top": TOP_MI, "mi_seed": SEED,
            "rf_rfe": RF_RFE_PARAMS, "rf_classifier": RF_PARAMS,
            "rfe_step": 1, "rfe_target": 1, "k_grid": [1, 100]},
        "test_csv_data_rows_read": 0,
        "test_model_evaluations": 0}
    filename = "baseline_profile.json" if mode == "baseline" else f"{mode}.json"
    write_json(output / filename, outcome)
    print(f"MEMORYLAB {mode}: k={outcome['best_k']} "
          f"peak={whole.peak_rss_bytes / 1024**3:.3f} GiB "
          f"elapsed={outcome['elapsed_seconds']:.1f}s", flush=True)
    return outcome


def ml_compare(baseline: dict, candidate: dict) -> dict:
    def same_floats(left, right):
        return np.array_equal(np.asarray(left, dtype=np.float64).view(np.uint64),
                              np.asarray(right, dtype=np.float64).view(np.uint64))
    checks = {
        "same_X_y_bytes": baseline["raw_sha256_before"] == candidate["raw_sha256_before"]
            == candidate["raw_sha256_after"],
        "mi_scores_bitwise_380": same_floats(baseline["mi_scores_canonical"],
                                             candidate["mi_scores_canonical"]),
        "mi_top100_order": baseline["mi_top100_names"] == candidate["mi_top100_names"],
        "mi_scaler_bitwise": all(same_floats(baseline["mi_scaler"][key],
            candidate["mi_scaler"][key]) for key in ("mean_", "var_", "scale_")),
        "top100_scaler_bitwise": all(same_floats(baseline["top100_scaler"][key],
            candidate["top100_scaler"][key]) for key in ("mean_", "var_", "scale_")),
        "top100_scaled_bytes": baseline["top100_scaled_sha256"]
            == candidate["top100_scaled_sha256"],
        "rfe_ranks_100": baseline["rfe_ranks_by_name"] == candidate["rfe_ranks_by_name"],
        "rfe_order_100": baseline["rfe_ranked_names"] == candidate["rfe_ranked_names"],
        "rfe_forest_fits_100": baseline["rfe_forest_fits"]
            == candidate["rfe_forest_fits"] == 100,
        "all_100_selected_prefixes": all(
            baseline["rfe_ranked_names"][:k] == candidate["rfe_ranked_names"][:k]
            for k in K_GRID),
        "k_grid_1_to_100": [r["k"] for r in candidate["k_records"]] == list(K_GRID),
        "best_k": baseline["best_k"] == candidate["best_k"],
        "frozen_features": baseline["frozen_features"] == candidate["frozen_features"],
        "refit_scaler_bitwise": all(same_floats(baseline["final_refit_scaler"][key],
            candidate["final_refit_scaler"][key]) for key in ("mean_", "var_", "scale_")),
        "raw_unchanged": candidate["raw_sha256_before"] == candidate["raw_sha256_after"],
        "mi_global_rng_stream_unchanged": (
            baseline["mi_global_rng_state_before"]
            == baseline["mi_global_rng_state_after"]
            and candidate["mi_global_rng_state_before"]
            == candidate["mi_global_rng_state_after"]),
        "test_not_read_or_evaluated": candidate["test_csv_data_rows_read"] == 0
            and candidate["test_model_evaluations"] == 0}
    differences = []
    if len(candidate["k_records"]) != 100 or len(baseline["k_records"]) != 100:
        checks["all_100_validation_accuracy_and_confusion"] = False
    else:
        for left, right in zip(baseline["k_records"], candidate["k_records"]):
            if (left["k"] != right["k"]
                    or not same_floats([left["accuracy"]], [right["accuracy"]])
                    or left["confusion_matrix"] != right["confusion_matrix"]):
                differences.append({"k": right["k"], "baseline_accuracy": left["accuracy"],
                    "candidate_accuracy": right["accuracy"],
                    "baseline_confusion": left["confusion_matrix"],
                    "candidate_confusion": right["confusion_matrix"]})
        checks["all_100_validation_accuracy_and_confusion"] = not differences
    if candidate["flags"]["E3"]:
        checks["E3_isolated_writable_copies"] = (
            candidate["E3_isolation"]["train_copy_writable"]
            and candidate["E3_isolation"]["val_copy_writable"]
            and not candidate["E3_isolation"]["train_shares_raw"]
            and not candidate["E3_isolation"]["val_shares_raw"])
    if candidate["flags"]["E4"]:
        checks["E4_all_100_workspace_prefixes"] = candidate["E4_workspace"][
            "all_100_prefixes_equal"]
    return {
        "status": "EQUIVALENCE_PASS_TECHNICAL_SAMPLE" if all(checks.values())
            else "EQUIVALENCE_FAIL",
        "checks": checks, "first_validation_differences": differences[:10],
        "validation_difference_count": len(differences),
        "mi_max_absolute_difference": float(np.max(np.abs(
            np.asarray(baseline["mi_scores_canonical"])
            - np.asarray(candidate["mi_scores_canonical"])))),
        "mi_max_relative_difference": float(np.max(
            np.abs(np.asarray(baseline["mi_scores_canonical"])
            - np.asarray(candidate["mi_scores_canonical"]))
            / np.maximum(np.abs(np.asarray(baseline["mi_scores_canonical"])), 1e-300))),
        "mi_boundary_tie_count": int(sum(
            s == baseline["mi_scores_canonical"][FEATURE_NAMES.index(
                baseline["mi_top100_names"][-1])]
            for s in baseline["mi_scores_canonical"]))}


def ml_prepare_sample(output: Path) -> dict:
    store = reference.WindowStore(DEFAULT_WINDOWS)
    before = store.snapshot()
    source = reference.source_hashes(DEFAULT_ORIGINAL)
    reference.source_audit(DEFAULT_ORIGINAL)
    reports = {}
    for role, count in (("train", MEMORYLAB_TRAIN_ROWS), ("val", MEMORYLAB_VAL_ROWS)):
        data, load, lineage = store.load_sample(MEMORYLAB_CONFIG, role, count)
        path = output / f"{role}_sample.npz"
        np.savez(path, X=data.X, y=data.y)
        reports[role] = {"rows": len(data.y), "load": load,
            "X_sha256": ml_array_sha(data.X), "y_sha256": ml_array_sha(data.y),
            "window_id_sha256": hashlib.sha256("\n".join(
                item["Window_ID"] for item in lineage).encode("utf-8")).hexdigest().upper(),
            "sample_file_sha256": sha256(path), "sample_file_bytes": path.stat().st_size}
    if store.snapshot() != before or reference.source_hashes(DEFAULT_ORIGINAL) != source:
        raise ValueError("Window metadata or original scripts changed during sample preparation")
    payload = {"status": "TECHNICAL_BENCHMARK_NOT_OFFICIAL",
        "config": MEMORYLAB_CONFIG, "samples": reports,
        "official_runner_sha256": sha256(ROOT / "huyen_code" / "ciciomt24_training.py"),
        "five_original_sha256": source,
        "windows_snapshot_before": before,
        "test_csv_data_rows_read": 0, "test_model_evaluations": 0,
        "sample_method": "WindowStore.load_sample deterministic per-source quota and stride=8",
        "source_window_files_modified": False}
    write_json(output / "sample_manifest.json", payload)
    return payload


def ml_benchmark() -> dict:
    output = MEMORYLAB_ROOT
    if output.exists() and (
            not output.is_dir()
            or {p.name for p in output.iterdir()} != {"optimization_diff.patch"}):
        raise FileExistsError(f"Refusing to overwrite memorylab: {output}")
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise MemoryError("Technical sample admission requires at least 2 GiB available")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "INCOMPLETE", "official_training": False,
        "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        sample = ml_prepare_sample(output)
        for mode in MEMORYLAB_MODES:
            print(f"MEMORYLAB starting {mode} (technical sample, no Test)", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker",
                "--mode", mode], check=True, timeout=1800)
        baseline = json.loads((output / "baseline_profile.json").read_text(encoding="utf-8"))
        runs = {"baseline": baseline}
        runs.update({mode: json.loads((output / f"{mode}.json").read_text(encoding="utf-8"))
            for mode in MEMORYLAB_MODES[1:]})
        comparisons = {mode: ml_compare(baseline, runs[mode])
            for mode in MEMORYLAB_MODES[1:]}
        write_json(output / "equivalence_report.json", {
            "status": "PASS" if all(c["status"] == "EQUIVALENCE_PASS_TECHNICAL_SAMPLE"
                for c in comparisons.values()) else "EQUIVALENCE_FAIL",
            "baseline": "original runner functions, identical fixed sample",
            "sample_rows": baseline["sample_rows"], "per_mode": comparisons,
            "test_rows_read_or_evaluated": 0})
        summary = {"status": "TECHNICAL_BENCHMARK_NOT_OFFICIAL",
            "sample_rows": baseline["sample_rows"], "sample_manifest": sample["samples"],
            "modes": {mode: {"peak_rss_bytes": r["total_resources"]["peak_rss_bytes"],
                "peak_private_commit_bytes": r["total_resources"]["peak_private_commit_bytes"],
                "elapsed_seconds": r["elapsed_seconds"],
                "phase_resources": r["phase_resources"],
                "best_k": r["best_k"],
                "equivalence_status": ("REFERENCE" if mode == "baseline"
                    else comparisons[mode]["status"])}
                for mode, r in runs.items()},
            "all_100_k_each_mode": all(len(r["k_records"]) == 100 for r in runs.values()),
            "test_csv_data_rows_read": 0, "official_full_training": False}
        write_json(output / "optimization_benchmarks.json", summary)
        recommendation = {
            "original_runner_FW512_FS8_guard_required_available_bytes": 25762976100,
            "original_guard_changed": False,
            "optimized_runner_admission": "INSUFFICIENT_EVIDENCE",
            "reason": "A fixed 8192/2048 technical sample cannot bound full FW512/FS8 MI allocations, RSS or paging.",
            "full_FW512_FS8_training_launched": False,
            "test_csv_data_rows_read": 0}
        write_json(output / "resource_admission_recommendation.json", recommendation)
        current_hash = sha256(ROOT / "huyen_code" / "ciciomt24_training.py")
        if current_hash != sample["official_runner_sha256"]:
            raise ValueError("Official runner SHA256 changed during memorylab")
        write_json(output / "status.json", {"status": "COMPLETE_TECHNICAL_ONLY",
            "official_training": False, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "equivalence_status": json.loads((output / "equivalence_report.json").read_text(
                encoding="utf-8"))["status"]})
        return summary
    except BaseException as exc:
        write_json(output / "status.json", {"status": "INCOMPLETE",
            "official_training": False, "error": str(exc),
            "stopped_utc": datetime.now(timezone.utc).isoformat()})
        raise


def memorylab_main():
    p = argparse.ArgumentParser(description="Stage5.5 technical RAM lab; no official training")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("benchmark")
    worker = sub.add_parser("_worker")
    worker.add_argument("--mode", choices=MEMORYLAB_MODES, required=True)
    args = p.parse_args()
    if args.command == "benchmark":
        ml_benchmark()
    else:
        ml_worker(MEMORYLAB_ROOT, args.mode)


if __name__ == "__main__":
    memorylab_main()
