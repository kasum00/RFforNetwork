"""Stage 7.2 opt-in Train-only SVM scaling probe, never official training.

No Stage 7.1 defaults, scientific classifier parameters, Validation/Test rows,
or full-SVM resource gate are changed by this module. The review command reads
JSON only and never opens window CSVs or fits an estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler

import ciciomt24_svm_probe as stage71
import ciciomt24_training as training


SCHEMA = "stage7.2.train_only.svm_scaling.v1"
SIZES = (16_000, 32_000, 64_000)
HARD_CAP = 64_000
GIB = 1024**3
K_VALUES = (1, 10, 100)


@dataclass(frozen=True)
class ScalingLimits:
    sizes: tuple[int, ...] = SIZES
    seed: int = 42
    timeout_per_size_sec: int = 1200
    max_rss_gib: float = 12.0

    def validate(self) -> None:
        if (not self.sizes or tuple(self.sizes) != tuple(sorted(set(self.sizes)))
                or any(type(n) is not int or n not in SIZES for n in self.sizes)
                or max(self.sizes) > HARD_CAP):
            raise ValueError("Scaling sizes must be increasing, unique and chosen from 16000/32000/64000")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if type(self.timeout_per_size_sec) is not int or not 60 <= self.timeout_per_size_sec <= 3600:
            raise ValueError("Scaling timeout must be 60..3600 seconds per worker")
        if not math.isfinite(self.max_rss_gib) or not 2 <= self.max_rss_gib <= 32:
            raise ValueError("Scaling RSS cap must be 2..32 GiB")

    @property
    def max_rss_bytes(self) -> int:
        return int(self.max_rss_gib * GIB)


def headroom_bytes(available: int) -> int:
    return max(2 * GIB, math.ceil(available * .2))


def admission_reason(available: int, limits: ScalingLimits) -> str | None:
    return ("INSUFFICIENT_AVAILABLE_RAM" if available < limits.max_rss_bytes + headroom_bytes(available)
            else None)


def guard_reason(elapsed: float, tree_rss: int, available: int,
                 limits: ScalingLimits) -> str | None:
    if elapsed > limits.timeout_per_size_sec:
        return "TIMEOUT"
    if tree_rss > limits.max_rss_bytes:
        return "RSS_CAP_EXCEEDED"
    if available < max(2 * GIB, math.ceil(psutil.virtual_memory().total * .05)):
        return "HOST_AVAILABLE_RAM_LOW"
    return None


def class_quotas(total_by_label: dict[int, int], size: int) -> dict[int, int]:
    """Natural Train ratio, with an explicit two-Benign technical floor if needed."""
    total = sum(total_by_label.values())
    if size > total or any(total_by_label.get(label, 0) < 2 for label in (0, 1)):
        raise ValueError("Train cannot supply requested size and both proxy classes")
    benign = max(2, round(size * total_by_label[0] / total))
    benign = min(benign, total_by_label[0])
    attack = size - benign
    if benign < 2 or attack < 2 or attack > total_by_label[1]:
        raise ValueError("Train class counts cannot supply fit and internal holdout")
    return {0: benign, 1: attack}


def selected_positions(files, size: int, seed: int):
    """Nested uniform-without-replacement indices, separately per proxy class.

    Full-class permutations cost about 11 MB for FW256/FS8; no full feature
    matrix or Validation/Test rows are loaded. Prefixes are nested across sizes.
    """
    totals = {label: sum(f.count for f in files if f.label == label) for label in (0, 1)}
    quotas = class_quotas(totals, size)
    picked = {i: np.empty(0, dtype=np.int64) for i in range(len(files))}
    for label in (0, 1):
        offsets = np.cumsum([f.count for f in files if f.label == label], dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        global_rows = np.sort(rng.permutation(totals[label])[:quotas[label]])
        matching = [i for i, f in enumerate(files) if f.label == label]
        assignments = np.searchsorted(offsets, global_rows, side="right")
        for local_file, file_index in enumerate(matching):
            positions = global_rows[assignments == local_file]
            start = 0 if local_file == 0 else offsets[local_file - 1]
            picked[file_index] = positions - start
    if sum(len(part) for part in picked.values()) != size:
        raise AssertionError("Requested scaling sample size was not selected")
    return picked, totals, quotas


def load_scaling_sample(windows: Path, manifest_root: Path, config: str,
                        size: int, seed: int):
    """Stream only Train CSVs; validate every parsed chunk before retaining rows."""
    if config not in training.CONFIGS or size not in SIZES or size > HARD_CAP:
        raise ValueError("Unknown FW/FS or non-scaling sample size")
    if len(training.FEATURE_NAMES) != 380:
        raise ValueError("Approved behavior feature schema must contain 380 columns")
    store = training.WindowStore(windows, split_root=manifest_root)
    files = [f for f in store.files[config] if f.role == "train"]
    files.sort(key=lambda f: (f.source_id, str(f.path)))
    picked, totals, quotas = selected_positions(files, size, seed)
    arrays, labels, lineage = [], [], []
    parsed_rows = 0
    began = time.perf_counter()
    with training.PeakRSS() as peak:
        for file_index, file in enumerate(files):
            positions = picked[file_index]
            if not len(positions):
                continue
            store.validate_header(file.path)
            offset = 0
            retained = 0
            for frame in pd.read_csv(file.path, chunksize=4096, low_memory=False):
                store.validate_chunk(frame, file, offset)
                hi = offset + len(frame)
                lo_i, hi_i = np.searchsorted(positions, [offset, hi])
                chosen = frame.iloc[positions[lo_i:hi_i] - offset]
                parsed_rows += len(frame)
                if len(chosen):
                    prepared = training.prepare_xy(chosen, "train")
                    if not (prepared.y == file.label).all():
                        raise ValueError("Selected proxy labels disagree with manifest")
                    arrays.append(prepared.X)
                    labels.append(prepared.y)
                    lineage.extend({"Window_ID": str(row.Window_ID),
                                    "Source_CSV_ID": int(file.source_id),
                                    "Scenario": file.scenario,
                                    "Label": int(row.Label)} for row in chosen.itertuples())
                    retained += len(chosen)
                offset = hi
                if offset > int(positions[-1]):
                    break
            if retained != len(positions):
                raise ValueError(f"Requested {len(positions)} rows but read {retained}: {file.path}")
    if not arrays:
        raise ValueError("No Train windows selected")
    sample = training.PreparedSplit("train", np.concatenate(arrays), np.concatenate(labels))
    sample.validate(both_classes=True)
    ids = [row["Window_ID"] for row in lineage]
    if len(sample.y) != size or len(ids) != size or len(set(ids)) != size:
        raise ValueError("Scaling sample missing/repeated Window_ID or wrong row count")
    actual = {"benign": int((sample.y == 0).sum()), "attack": int((sample.y == 1).sum())}
    if actual != {"benign": quotas[0], "attack": quotas[1]}:
        raise ValueError("Sample class quota mismatch")
    info = {"requested_rows": size, "actual_rows": len(sample.y),
            "population_by_proxy_label": {"benign": totals[0], "attack": totals[1]},
            "sample_by_proxy_label": actual,
            "minority_floor_applied": quotas[0] != round(size * totals[0] / sum(totals.values())),
            "sample_window_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "source_count": len({row["Source_CSV_ID"] for row in lineage}),
            "scenario_count": len({row["Scenario"] for row in lineage}),
            "rows_parsed": parsed_rows, "load_seconds": time.perf_counter() - began,
            "load_peak_rss_bytes": peak.peak,
            "sample_method": "global per-class seeded permutation of full Train manifest indices; no replacement; natural ratio with disclosed two-Benign floor",
            "population_ratio_is_not_probe_ratio": True}
    return sample, info


def run_worker(windows: Path, manifest_root: Path, config: str, experiment: str,
               size: int, seed: int, expected_hashes: dict[str, str]) -> dict:
    if experiment not in ("baseline_svm", "mi_rfe_svm") or size not in SIZES:
        raise ValueError("Worker accepts one approved SVM branch and one scaling size")
    actual_hashes = {"training": training.sha256(Path(training.__file__)),
                     "scaling": training.sha256(Path(__file__)),
                     "summary": training.sha256(windows / "window_run_summary.json"),
                     "manifest": training.sha256(manifest_root / "split_manifest.csv")}
    if actual_hashes != expected_hashes:
        raise ValueError("Source/manifest hash differs from supervisor snapshot")
    with training.PeakRSS() as peak:
        sample, sampling = load_scaling_sample(windows, manifest_root, config, size, seed)
        fit_i, hold_i, selected_i = stage71.stratified_train_indices(sample.y, size, seed)
        if len(selected_i) != size or len(fit_i) + len(hold_i) != size:
            raise AssertionError("Internal Train/holdout count mismatch")
        fit = training.PreparedSplit("train", sample.X[fit_i], sample.y[fit_i])
        hold = training.PreparedSplit("val", sample.X[hold_i], sample.y[hold_i])
        fit.validate(both_classes=True)
        hold.validate(both_classes=True)
        result = {"schema": SCHEMA, "status": "RUNNING", "experiment": experiment,
                  "config": config, "seed": seed, "sample_rows_requested": size,
                  "sample_rows_actual": len(sample.y), "fit_rows": len(fit.y),
                  "internal_train_holdout_rows": len(hold.y),
                  "sample": sampling,
                  "fit_by_proxy_label": {"benign": int((fit.y == 0).sum()),
                                          "attack": int((fit.y == 1).sum())},
                  "holdout_by_proxy_label": {"benign": int((hold.y == 0).sum()),
                                              "attack": int((hold.y == 1).sum())},
                  "feature_count": 380, "source_hashes": actual_hashes,
                  "environment": training.environment(),
                  "label_origin": "SCENARIO_PROXY", "sampled_training": True,
                  "scientific_result": False, "full_training_authorized": False,
                  "official_best_k": None, "validation_csv_rows_read": 0,
                  "test_csv_rows_read": 0}
        if experiment == "baseline_svm":
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(fit.X)
            X_hold = scaler.transform(hold.X)
            model, measurement = training.timed_fit(
                training.make_model("svm", baseline=True), X_fit, fit.y)
            result["fits"] = [{"features": 380, "fit": measurement,
                               "predict": stage71._predict_only(model, X_hold)}]
        else:
            mi = training.select_mi(fit)
            X_fit, X_hold, _ = training.scale_top100(fit, hold, mi)
            ranking = training.rank_rfe(X_fit, fit.y, mi["names"], "svm", target=97)
            result["mi"] = mi["measurement"]
            result["partial_rfe"] = {"measurement": ranking["measurement"],
                                     "forest_fits": ranking["forest_fits"],
                                     "complete_ranking": False}
            result["fits"] = []
            for k in K_VALUES:
                indices = [mi["names"].index(name) for name in ranking["names"][:k]]
                model, measurement = training.timed_fit(
                    training.make_model("svm"), X_fit[:, indices], fit.y)
                result["fits"].append({"features": k, "fit": measurement,
                                       "predict": stage71._predict_only(model, X_hold[:, indices])})
    result["peak_process_rss_bytes"] = peak.peak
    result["status"] = "PASS_TECHNICAL_ONLY" if evidence_complete(result) else "REVIEW_REQUIRED_INCOMPLETE"
    return result


def evidence_complete(row: dict) -> bool:
    """Fail closed; no PASS from a worker missing required measurements."""
    if not isinstance(row, dict) or row.get("schema") != SCHEMA:
        return False
    size, fit_rows, hold_rows = (row.get(k) for k in
                                 ("sample_rows_requested", "fit_rows", "internal_train_holdout_rows"))
    if (type(size) is not int or size not in SIZES or row.get("sample_rows_actual") != size
            or type(fit_rows) is not int or type(hold_rows) is not int
            or fit_rows + hold_rows != size or min(fit_rows, hold_rows) < 2
            or row.get("feature_count") != 380 or row.get("label_origin") != "SCENARIO_PROXY"
            or row.get("validation_csv_rows_read") != 0 or row.get("test_csv_rows_read") != 0
            or row.get("scientific_result") is not False
            or row.get("sampled_training") is not True
            or row.get("full_training_authorized") is not False
            or row.get("official_best_k") is not None
            or not isinstance(row.get("peak_process_rss_bytes"), int)
            or row["peak_process_rss_bytes"] <= 0):
        return False
    sample = row.get("sample")
    if not isinstance(sample, dict):
        return False
    if (sample.get("actual_rows") != size or sample.get("requested_rows") != size
            or not isinstance(sample.get("sample_window_ids_sha256"), str)
            or len(sample["sample_window_ids_sha256"]) != 64
            or not isinstance(sample.get("source_count"), int) or sample["source_count"] < 1
            or not isinstance(sample.get("scenario_count"), int) or sample["scenario_count"] < 1):
        return False
    for key, expected in (("sample_by_proxy_label", size),
                          ("population_by_proxy_label", None)):
        counts = sample.get(key, {})
        if (not isinstance(counts, dict) or set(counts) != {"benign", "attack"}
                or any(type(v) is not int or v < 2 for v in counts.values())
                or (expected is not None and sum(counts.values()) != expected)):
            return False
    for key, expected in (("fit_by_proxy_label", fit_rows),
                          ("holdout_by_proxy_label", hold_rows)):
        counts = row.get(key, {})
        if (not isinstance(counts, dict) or set(counts) != {"benign", "attack"}
                or any(type(v) is not int or v < 1 for v in counts.values())
                or sum(counts.values()) != expected):
            return False
    fits = row.get("fits", [])
    wanted = [380] if row.get("experiment") == "baseline_svm" else list(K_VALUES)
    if (not isinstance(fits, list) or any(not isinstance(f, dict) for f in fits)
            or [f.get("features") for f in fits] != wanted):
        return False
    for item in fits:
        fit, predict = item.get("fit"), item.get("predict")
        if not isinstance(fit, dict) or not isinstance(predict, dict):
            return False
        if (fit.get("converged") is not True or fit.get("fit_status") != 0
                or not isinstance(fit.get("warnings"), list)
                or not isinstance(fit.get("n_iter"), list) or not fit["n_iter"]
                or not isinstance(fit.get("seconds"), (int, float)) or fit["seconds"] <= 0
                or not isinstance(fit.get("peak_rss_bytes"), int) or fit["peak_rss_bytes"] <= 0
                or not isinstance(fit.get("support_vectors"), int)
                or not 1 <= fit["support_vectors"] <= fit_rows
                or predict.get("rows") != hold_rows
                or not isinstance(predict.get("seconds"), (int, float)) or predict["seconds"] < 0):
            return False
        item["support_vector_fraction"] = fit["support_vectors"] / fit_rows
    if row.get("experiment") == "mi_rfe_svm":
        mi, rfe = row.get("mi"), row.get("partial_rfe")
        if not isinstance(mi, dict) or not isinstance(rfe, dict):
            return False
        for measured in (mi, rfe.get("measurement")):
            if not isinstance(measured, dict):
                return False
            if (not isinstance(measured.get("seconds"), (int, float)) or measured["seconds"] <= 0
                    or not isinstance(measured.get("peak_rss_bytes"), int)
                    or measured["peak_rss_bytes"] <= 0):
                return False
        if rfe.get("complete_ranking") is not False or rfe.get("forest_fits") != 4:
            return False
    return True


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _update_owned(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_scaling_probe(repo: Path, windows: Path, manifest_root: Path, config: str,
                      experiment: str, output: Path, limits: ScalingLimits = ScalingLimits(),
                      *, scaling_opt_in: bool = False) -> dict:
    if not scaling_opt_in:
        raise PermissionError("Stage 7.2 scaling requires explicit --enable-scaling-probe")
    limits.validate()
    if config not in training.CONFIGS or experiment not in ("baseline_svm", "mi_rfe_svm"):
        raise ValueError("One approved FW/FS and one SVM experiment are required")
    if output.suffix.lower() != ".json" or output.exists():
        raise FileExistsError("Choose a fresh .json output; existing artifacts are never overwritten")
    workers = [output.with_name(f"{output.stem}.n{n}.json") for n in limits.sizes]
    if any(p.exists() for p in workers):
        raise FileExistsError("One or more worker JSON artifacts already exist")
    lock = Path(tempfile.gettempdir()) / ("stage72_svm_" + hashlib.sha256(
        str(windows.resolve()).encode()).hexdigest()[:16] + ".lock")
    token = uuid.uuid4().hex
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()} {token}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"Another scaling probe or stale lock needs inspection: {lock}") from exc
    try:
        hashes = {"training": training.sha256(repo / "huyen_code/ciciomt24_training.py"),
                  "scaling": training.sha256(Path(__file__)),
                  "summary": training.sha256(windows / "window_run_summary.json"),
                  "manifest": training.sha256(manifest_root / "split_manifest.csv")}
        originals = training.source_hashes(repo / "CODE_ctu13")
        available = psutil.virtual_memory().available
        result = {"schema": SCHEMA, "status": "INCOMPLETE_DO_NOT_USE",
                  "experiment": experiment, "config": config, "sizes": list(limits.sizes),
                  "seed": limits.seed, "timeout_per_size_sec": limits.timeout_per_size_sec,
                  "max_tree_rss_bytes": limits.max_rss_bytes,
                  "available_ram_before_bytes": available,
                  "source_hashes": hashes, "original_script_hashes": originals,
                  "environment": training.environment(),
                  "label_origin": "SCENARIO_PROXY", "technical_only": True,
                  "sampled_training": True, "scientific_result": False,
                  "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                  "full_training_authorized": False, "official_best_k": None,
                  "resource_gate": "REVIEW_REQUIRED", "runs": []}
        _write_new(output, result)
        started = time.monotonic()
        try:
            for size, worker_output in zip(limits.sizes, workers):
                available = psutil.virtual_memory().available
                reason = admission_reason(available, limits)
                if reason:
                    result.update(status="FAIL_RESOURCE_ADMISSION", stop_reason=reason,
                                  available_ram_at_stop_bytes=available)
                    break
                if hashes != {"training": training.sha256(repo / "huyen_code/ciciomt24_training.py"),
                              "scaling": training.sha256(Path(__file__)),
                              "summary": training.sha256(windows / "window_run_summary.json"),
                              "manifest": training.sha256(manifest_root / "split_manifest.csv")}:
                    result.update(status="REVIEW_REQUIRED_OR_FAILED", stop_reason="SOURCE_HASH_CHANGED")
                    break
                print(f"[Stage 7.2] {experiment}/{config}: starting {size} Train windows", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                           "--windows", str(windows), "--manifest-root", str(manifest_root),
                           "--config", config, "--experiment", experiment,
                           "--size", str(size), "--seed", str(limits.seed),
                           "--expected-hashes", json.dumps(hashes),
                           "--output", str(worker_output)]
                proc = subprocess.Popen(command, cwd=str(repo))
                began, peak_tree, stop = time.monotonic(), 0, None
                while proc.poll() is None:
                    try:
                        peak_tree = max(peak_tree, stage71._tree_rss(proc))
                        stop = guard_reason(time.monotonic() - began, peak_tree,
                                            psutil.virtual_memory().available, limits)
                    except psutil.AccessDenied:
                        stop = "RSS_MONITOR_UNAVAILABLE"
                    if stop:
                        stage71._stop_own_worker(proc)
                        break
                    time.sleep(.25)
                code = proc.poll()
                row = json.loads(worker_output.read_text(encoding="utf-8")) if worker_output.is_file() else {}
                row.update(worker_returncode=code, peak_process_tree_rss_bytes_sampled=peak_tree,
                           supervisor_stop_reason=stop, timeout_limit_sec=limits.timeout_per_size_sec,
                           rss_cap_bytes=limits.max_rss_bytes,
                           available_ram_before_bytes=available)
                result["runs"].append(row)
                _update_owned(output, result)
                print(f"[Stage 7.2] {size}: {row.get('status')} stop={stop} RSS={peak_tree}", flush=True)
                if (stop or code != 0 or row.get("status") != "PASS_TECHNICAL_ONLY"
                        or peak_tree <= 0 or row.get("source_hashes") != hashes
                        or row.get("sample_rows_actual") != size or not evidence_complete(row)):
                    result.update(status="REVIEW_REQUIRED_OR_FAILED",
                                  stop_reason=stop or (f"WORKER_{row.get('status')}" if code != 0 else
                                                       "MISSING_OR_INCONSISTENT_EVIDENCE"))
                    break
            else:
                result["status"] = "PASS_TECHNICAL_ONLY"
        except KeyboardInterrupt:
            if "proc" in locals() and proc.poll() is None:
                stage71._stop_own_worker(proc)
            result.update(status="INTERRUPTED_DO_NOT_USE", stop_reason="USER_INTERRUPT")
            raise
        except BaseException as exc:
            if "proc" in locals() and proc.poll() is None:
                stage71._stop_own_worker(proc)
            result.update(status="REVIEW_REQUIRED_OR_FAILED",
                          stop_reason=f"SUPERVISOR_ERROR: {type(exc).__name__}: {exc}")
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            _update_owned(output, result)
        return result
    finally:
        if lock.exists() and lock.read_text(encoding="utf-8") == f"{os.getpid()} {token}\n":
            lock.unlink()


def render_resource_review(baseline_path: Path, compact_path: Path) -> str:
    """Read only two existing JSON artifacts. No WindowStore, CSV or fit call."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    if (baseline.get("schema") != SCHEMA or compact.get("schema") != SCHEMA
            or baseline.get("experiment") != "baseline_svm"
            or compact.get("experiment") != "mi_rfe_svm"
            or baseline.get("config") != compact.get("config")):
        raise ValueError("Require matching Stage 7.2 Baseline and MI-RFE-SVM JSON")
    def factor(current, previous):
        if (not isinstance(current, (int, float)) or not isinstance(previous, (int, float))
                or previous <= 0):
            return "n/a"
        return f"{current / previous:.2f}x"

    lines = ["# Stage 7.2 resource review — REVIEW_REQUIRED", "",
             "Train-only technical samples; NOT Test accuracy or full-SVM authorization.",
             "No full-data feasibility is inferred from these bounded points.", "",
             "| Experiment / k | Sample / fit / holdout | SVC fit s | Predict s | Fit growth | Peak tree RSS GiB | RSS growth | SV / fit | SV growth | SV fraction | Converged | Warnings |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    issues = []
    for item in (baseline, compact):
        if (item.get("status") != "PASS_TECHNICAL_ONLY"
                or item.get("resource_gate") != "REVIEW_REQUIRED"
                or item.get("full_training_authorized") is not False
                or item.get("validation_csv_rows_read") != 0
                or item.get("test_csv_rows_read") != 0):
            issues.append(f"{item['experiment']}: overall status/gate/row-read evidence is incomplete")
        if item.get("sizes") != [16_000, 32_000, 64_000]:
            issues.append(f"{item['experiment']}: expected 16k/32k/64k sequence is incomplete")
        if len(item.get("runs", [])) != 3:
            issues.append(f"{item['experiment']}: not all three worker artifacts are present")
        if [r.get("sample_rows_requested") for r in item.get("runs", [])] != [16_000, 32_000, 64_000]:
            issues.append(f"{item['experiment']}: worker sizes/order do not match the approved sequence")
        previous_by_k = {}
        for run in item.get("runs", []):
            sample = run.get("sample") if isinstance(run.get("sample"), dict) else {}
            if (run.get("status") != "PASS_TECHNICAL_ONLY"
                    or not evidence_complete(run)
                    or run.get("source_hashes") != item.get("source_hashes")
                    or run.get("supervisor_stop_reason") is not None
                    or run.get("worker_returncode") != 0):
                issues.append(f"{item['experiment']} size={run.get('sample_rows_requested')}: incomplete/failed worker")
            size = run.get("sample_rows_requested")
            rss = run.get("peak_process_tree_rss_bytes_sampled")
            for fit in (run.get("fits") or []):
                measurement = fit.get("fit", {})
                k = fit.get("features")
                prior = previous_by_k.get(k)
                seconds = measurement.get("seconds")
                support = measurement.get("support_vectors")
                warning_text = str(measurement.get("warnings", []))
                lines.append(
                    f"| {item['experiment']} / {k} | "
                    f"{size}/{run.get('fit_rows')}/{run.get('internal_train_holdout_rows')} | "
                    f"{seconds} | {fit.get('predict', {}).get('seconds')} | "
                    f"{factor(seconds, prior[0]) if prior else 'n/a'} | "
                    f"{round(rss / GIB, 3) if isinstance(rss, (int, float)) else 'n/a'} | "
                    f"{factor(rss, prior[1]) if prior else 'n/a'} | "
                    f"{support}/{run.get('fit_rows')} | "
                    f"{factor(support, prior[2]) if prior else 'n/a'} | "
                    f"{fit.get('support_vector_fraction')} | "
                    f"{measurement.get('converged')} | {warning_text} |")
                previous_by_k[k] = (seconds, rss, support)
            if item["experiment"] == "mi_rfe_svm":
                mi = run.get("mi") or {}
                rfe = run.get("partial_rfe") or {}
                rfe_measurement = rfe.get("measurement") or {}
                lines.append(f"| MI + partial RF-RFE at {size} | "
                             f"Benign/Attack {sample.get('sample_by_proxy_label')} | "
                             f"MI {mi.get('seconds')} + RFE {rfe_measurement.get('seconds')} | "
                             f"n/a | n/a | n/a | n/a | 4 forest fits only | n/a | n/a | n/a | not full ranking |")
    lines.extend(["", "Evidence verdict: " + ("REVIEW_REQUIRED: inspect failures" if issues else
                                          "TECHNICAL_ONLY_PASS: manual resource review still required"),
                  *[f"- {issue}" for issue in issues],
                  "", "Environment Baseline / MI-RFE-SVM: " + str(baseline.get("environment")) +
                  " / " + str(compact.get("environment")),
                  "Proxy-label population/sample counts: " +
                  str([(r.get("sample") or {}).get("population_by_proxy_label") for r in baseline.get("runs", [])]) +
                  " / " +
                  str([(r.get("sample") or {}).get("sample_by_proxy_label") for r in baseline.get("runs", [])]),
                  "", "Sampling is class-stratified and may floor Benign to two; partial RF-RFE is not a complete ranking.",
                  "RBF/libsvm scaling is nonlinear; 16k/32k/64k cannot guarantee RAM/time at ~674k or ~1.35M Train windows.",
                  "RSS polling is not a hard OS memory limit. Capture/session independence remains unproved.",
                  "Validation/Test CSV rows were not read for this review. Full SVM remains blocked; explicit later approval required."])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--enable-scaling-probe", action="store_true")
    probe.add_argument("--repo", type=Path, required=True)
    probe.add_argument("--windows", type=Path, required=True)
    probe.add_argument("--manifest-root", type=Path, required=True)
    probe.add_argument("--config", choices=training.CONFIGS, required=True)
    probe.add_argument("--experiment", choices=("baseline_svm", "mi_rfe_svm"), required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument("--timeout-per-size", type=int, default=1200)
    probe.add_argument("--max-rss-gib", type=float, default=12.0)
    worker = sub.add_parser("_worker")
    worker.add_argument("--windows", type=Path, required=True)
    worker.add_argument("--manifest-root", type=Path, required=True)
    worker.add_argument("--config", choices=training.CONFIGS, required=True)
    worker.add_argument("--experiment", choices=("baseline_svm", "mi_rfe_svm"), required=True)
    worker.add_argument("--size", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--expected-hashes", required=True)
    worker.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("review")
    review.add_argument("--baseline", type=Path, required=True)
    review.add_argument("--compact", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        run_scaling_probe(args.repo, args.windows, args.manifest_root, args.config,
                          args.experiment, args.output,
                          ScalingLimits(tuple(args.sizes), args.seed,
                                        args.timeout_per_size, args.max_rss_gib),
                          scaling_opt_in=args.enable_scaling_probe)
    elif args.command == "_worker":
        if args.output.exists():
            raise FileExistsError(args.output)
        try:
            value = run_worker(args.windows, args.manifest_root, args.config,
                               args.experiment, args.size, args.seed,
                               json.loads(args.expected_hashes))
            _write_new(args.output, value)
        except BaseException as exc:
            _write_new(args.output, {"schema": SCHEMA, "status": "WORKER_FAILED",
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "scientific_result": False, "sampled_training": True,
                                     "full_training_authorized": False,
                                     "validation_csv_rows_read": 0,
                                     "test_csv_rows_read": 0})
            raise
    else:
        print(render_resource_review(args.baseline, args.compact))


if __name__ == "__main__":
    main()
