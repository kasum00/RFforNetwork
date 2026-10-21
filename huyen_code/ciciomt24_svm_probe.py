"""Stage 7.1: bounded, Train-only SVM timing probe. Never an official result.

This reuses the Stage 5 window loader, preprocessing, estimators, MI and partial
RF-RFE. It does not open Validation/Test CSVs or select an official best_k.
The supervisor owns only its own worker process tree and enforces soft RSS/time
limits; it cannot prove full-run feasibility or prevent every transient OOM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler

import ciciomt24_training as training


SCHEMA = "stage7.1.train_only.svm_probe.v1"
MAX_POOL_ROWS = 8192  # Existing Stage 5 loader's audited Train cap.
HEADROOM_BYTES = 2 * 1024**3
LOW_AVAILABLE_BYTES = 1024**3
TECHNICAL_K = (1, 10, 100)


@dataclass(frozen=True)
class ProbeLimits:
    sizes: tuple[int, ...] = (2000, 5000, 8192)
    seed: int = 42
    timeout_per_size_sec: int = 600
    max_rss_gib: float = 8.0

    def validate(self) -> None:
        if (not self.sizes or tuple(self.sizes) != tuple(sorted(set(self.sizes)))
                or any(not isinstance(n, int) or n < 4 or n > MAX_POOL_ROWS for n in self.sizes)):
            raise ValueError("Probe sizes must be unique, increasing and <=8192")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if not 30 <= self.timeout_per_size_sec <= 1200:
            raise ValueError("Per-size timeout must be 30..1200 seconds")
        if not math.isfinite(self.max_rss_gib) or not 2 <= self.max_rss_gib <= 16:
            raise ValueError("Probe RSS cap must be 2..16 GiB")

    @property
    def max_rss_bytes(self) -> int:
        return int(self.max_rss_gib * 1024**3)


def admission_reason(available_bytes: int, limits: ProbeLimits) -> str | None:
    """A startup guard, not an upper bound on actual memory requirements."""
    return ("INSUFFICIENT_AVAILABLE_RAM" if available_bytes < limits.max_rss_bytes + HEADROOM_BYTES
            else None)


def guard_reason(elapsed_sec: float, tree_rss_bytes: int,
                 available_bytes: int, limits: ProbeLimits) -> str | None:
    if elapsed_sec > limits.timeout_per_size_sec:
        return "TIMEOUT"
    if tree_rss_bytes > limits.max_rss_bytes:
        return "RSS_CAP_EXCEEDED"
    if available_bytes < LOW_AVAILABLE_BYTES:
        return "HOST_AVAILABLE_RAM_LOW"
    return None


def stratified_train_indices(y: np.ndarray, size: int, seed: int,
                             holdout_fraction: float = .2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a seeded nested sample and internal holdout from Train only."""
    if len(y) < size or size < 4 or not 0 < holdout_fraction < .5:
        raise ValueError("Invalid Train-only sample or internal holdout")
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(group) < 2 for group in groups) or sum(map(len, groups)) != len(y):
        raise ValueError("Train pool must contain at least two rows of each proxy class")
    quotas = training.proportional_quotas([len(group) for group in groups], size)
    for label in (0, 1):
        if quotas[label] < 2:
            other = 1 - label
            quotas[other] -= 2 - quotas[label]
            quotas[label] = 2
    if any(q < 2 or q > len(group) for q, group in zip(quotas, groups)):
        raise ValueError("Train pool cannot supply both classes to fit and holdout")
    rng = np.random.default_rng(seed)
    fit_parts, hold_parts, selected_parts = [], [], []
    for group, quota in zip(groups, quotas):
        picked = rng.permutation(group)[:quota]
        hold_rows = max(1, min(quota - 1, round(quota * holdout_fraction)))
        hold_parts.append(picked[:hold_rows])
        fit_parts.append(picked[hold_rows:])
        selected_parts.append(picked)
    fit = np.sort(np.concatenate(fit_parts))
    hold = np.sort(np.concatenate(hold_parts))
    selected = np.sort(np.concatenate(selected_parts))
    if len(selected) != size or np.intersect1d(fit, hold).size:
        raise AssertionError("Internal Train sample overlap/count error")
    return fit, hold, selected


def _predict_only(model, X: np.ndarray) -> dict:
    start = time.perf_counter()
    with training.PeakRSS() as ram:
        predicted = model.predict(X)
    return {"seconds": time.perf_counter() - start,
            "peak_rss_bytes": ram.peak, "rows": len(predicted),
            "note": "No accuracy is reported; internal Train holdout is not Validation/Test"}


def run_worker(windows: Path, manifest_root: Path, config: str,
               experiment: str, size: int, pool_rows: int, seed: int) -> dict:
    """Only WindowStore.load_sample(config, 'train', ...) may read CSV rows."""
    store = training.WindowStore(windows, split_root=manifest_root)
    pool, load, lineage = store.load_sample(config, "train", pool_rows)
    fit_i, hold_i, selected_i = stratified_train_indices(pool.y, size, seed)
    fit = training.PreparedSplit("train", pool.X[fit_i], pool.y[fit_i])
    # The role 'val' satisfies the existing scale_top100 API. Its rows still
    # originate exclusively from Train, not from the real Validation split.
    internal_holdout = training.PreparedSplit("val", pool.X[hold_i], pool.y[hold_i])
    fit.validate(both_classes=True)
    internal_holdout.validate(both_classes=True)
    ids = [lineage[int(i)]["Window_ID"] for i in selected_i]
    if len(ids) != len(set(ids)):
        raise ValueError("Probe sample repeats a Window_ID")
    result = {"status": "RUNNING", "schema": SCHEMA, "config": config,
              "experiment": experiment, "seed": seed, "pool_rows": pool_rows,
              "sample_rows": size, "fit_rows": len(fit.y),
              "internal_train_holdout_rows": len(internal_holdout.y),
              "sample_window_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
              "sample_method": load["sample_method"] + "; seeded stratified internal Train subset",
              "sample_by_proxy_label": {"benign": int((pool.y[selected_i] == 0).sum()),
                                        "attack": int((pool.y[selected_i] == 1).sum())},
              "fit_by_proxy_label": {"benign": int((fit.y == 0).sum()),
                                     "attack": int((fit.y == 1).sum())},
              "holdout_by_proxy_label": {"benign": int((internal_holdout.y == 0).sum()),
                                         "attack": int((internal_holdout.y == 1).sum())},
              "source_count": len({lineage[int(i)]["Source_CSV_ID"] for i in selected_i}),
              "load": load, "label_origin": "SCENARIO_PROXY",
              "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
              "official_best_k": None, "scientific_result": False}
    if experiment == "baseline_svm":
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(fit.X)
        X_hold = scaler.transform(internal_holdout.X)
        model, measured = training.timed_fit(training.make_model("svm", baseline=True), X_fit, fit.y)
        result["fits"] = [{"features": 380, "fit": measured,
                           "predict": _predict_only(model, X_hold)}]
    elif experiment == "mi_rfe_svm":
        mi = training.select_mi(fit)
        X_fit, X_hold, _ = training.scale_top100(fit, internal_holdout, mi)
        ranking = training.rank_rfe(X_fit, fit.y, mi["names"], "svm", target=97)
        result["mi"] = mi["measurement"]
        result["partial_rfe"] = {"measurement": ranking["measurement"],
                                 "forest_fits": ranking["forest_fits"],
                                 "complete_ranking": False}
        result["fits"] = []
        for k in TECHNICAL_K:
            indices = [mi["names"].index(name) for name in ranking["names"][:k]]
            model, measured = training.timed_fit(
                training.make_model("svm"), X_fit[:, indices], fit.y)
            result["fits"].append({"features": k, "fit": measured,
                                   "predict": _predict_only(model, X_hold[:, indices])})
    else:
        raise ValueError("Only the two source SVM branches are probeable")
    result["status"] = ("PASS_TECHNICAL_ONLY" if all(
        item["fit"].get("converged") is True for item in result["fits"])
        else "REVIEW_REQUIRED_NONCONVERGENCE")
    return result


def _tree_rss(proc: subprocess.Popen) -> int:
    try:
        parent = psutil.Process(proc.pid)
        owned = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for child in owned:
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total  # AccessDenied must fail closed in the supervisor.


def _stop_own_worker(proc: subprocess.Popen) -> None:
    """Never target an unrelated user process or a process by name."""
    try:
        parent = psutil.Process(proc.pid)
        owned = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        owned = []
    for child in reversed(owned):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_probe(repo: Path, windows: Path, manifest_root: Path, config: str,
              experiment: str, output: Path, limits: ProbeLimits = ProbeLimits()) -> dict:
    limits.validate()
    if config not in training.CONFIGS or experiment not in ("baseline_svm", "mi_rfe_svm"):
        raise ValueError("Probe requires one approved FW/FS and one source SVM experiment")
    workers = [output.with_name(f"{output.stem}.n{n}.json") for n in limits.sizes]
    if output.exists() or any(path.exists() for path in workers):
        raise FileExistsError("Probe refuses to overwrite an existing output or worker artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    available = psutil.virtual_memory().available
    result = {"schema": SCHEMA, "status": "INCOMPLETE_DO_NOT_USE", "config": config,
              "experiment": experiment, "sizes": list(limits.sizes), "seed": limits.seed,
              "timeout_per_size_sec": limits.timeout_per_size_sec,
              "max_tree_rss_bytes": limits.max_rss_bytes,
              "available_ram_before_bytes": available,
              "training_code_sha256": training.sha256(repo / "huyen_code" / "ciciomt24_training.py"),
              "window_summary_sha256": training.sha256(windows / "window_run_summary.json"),
              "original_script_hashes": training.source_hashes(repo / "CODE_ctu13"),
              "environment": training.environment(),
              "label_origin": "SCENARIO_PROXY", "technical_only": True,
              "sampled_training": True, "validation_csv_rows_read": 0,
              "test_csv_rows_read": 0, "full_training_authorized": False,
              "resource_gate": "REVIEW_REQUIRED", "runs": []}
    reason = admission_reason(available, limits)
    if reason:
        result.update(status="FAIL_RESOURCE_ADMISSION", stop_reason=reason)
        training.write_json(output, result)
        return result
    training.write_json(output, result)
    started = time.monotonic()
    try:
        for size, worker_output in zip(limits.sizes, workers):
            command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                       "--windows", str(windows), "--manifest-root", str(manifest_root),
                       "--config", config, "--experiment", experiment,
                       "--size", str(size), "--pool-rows", str(max(limits.sizes)),
                       "--seed", str(limits.seed), "--output", str(worker_output)]
            proc = subprocess.Popen(command, cwd=str(repo))
            began, peak, stop = time.monotonic(), 0, None
            while proc.poll() is None:
                try:
                    peak = max(peak, _tree_rss(proc))
                    stop = guard_reason(time.monotonic() - began, peak,
                                        psutil.virtual_memory().available, limits)
                except psutil.AccessDenied:
                    stop = "RSS_MONITOR_UNAVAILABLE"
                if stop:
                    _stop_own_worker(proc)
                    break
                time.sleep(.25)
            code = proc.poll()
            row = json.loads(worker_output.read_text(encoding="utf-8")) if worker_output.exists() else {}
            row.update(sample_rows=size, worker_returncode=code,
                       peak_process_tree_rss_bytes_sampled=peak,
                       supervisor_stop_reason=stop)
            result["runs"].append(row)
            training.write_json(output, result)
            if stop or code != 0 or row.get("status") != "PASS_TECHNICAL_ONLY":
                result.update(status="REVIEW_REQUIRED_OR_FAILED", stop_reason=stop or row.get("status") or "WORKER_FAILED")
                break
        else:
            result["status"] = "PASS_TECHNICAL_ONLY"
    except KeyboardInterrupt:
        if "proc" in locals() and proc.poll() is None:
            _stop_own_worker(proc)
        result.update(status="INTERRUPTED_DO_NOT_USE", stop_reason="USER_INTERRUPT")
        raise
    except BaseException as exc:
        if "proc" in locals() and proc.poll() is None:
            _stop_own_worker(proc)
        result.update(status="REVIEW_REQUIRED_OR_FAILED",
                      stop_reason=f"SUPERVISOR_ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        training.write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--windows", type=Path, required=True)
        p.add_argument("--manifest-root", type=Path, required=True)
        p.add_argument("--config", choices=training.CONFIGS, required=True)
        p.add_argument("--experiment", choices=("baseline_svm", "mi_rfe_svm"), required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "probe":
            p.add_argument("--repo", type=Path, required=True)
            p.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 8192])
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--timeout-per-size", type=int, default=600)
            p.add_argument("--max-rss-gib", type=float, default=8)
        else:
            p.add_argument("--size", type=int, required=True)
            p.add_argument("--pool-rows", type=int, required=True)
            p.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        run_probe(args.repo, args.windows, args.manifest_root, args.config,
                  args.experiment, args.output,
                  ProbeLimits(tuple(args.sizes), args.seed, args.timeout_per_size, args.max_rss_gib))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        try:
            value = run_worker(args.windows, args.manifest_root, args.config,
                               args.experiment, args.size, args.pool_rows, args.seed)
            training.write_json(args.output, value)
        except BaseException as exc:
            training.write_json(args.output, {"status": "WORKER_FAILED", "error": str(exc),
                                              "schema": SCHEMA, "validation_csv_rows_read": 0,
                                              "test_csv_rows_read": 0})
            raise


if __name__ == "__main__":
    main()
