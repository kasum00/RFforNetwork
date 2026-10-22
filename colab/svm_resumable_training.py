"""Completed-stage resume of the canonical MI -> RF-RFE -> RBF SVM pipeline.

The numerical functions come from ciciomt24_training. No partial SVC/RFE model
is resumed. Test is claimed once, after a verified completed final-model bundle.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import psutil

from .dataset_validator import manifest_root_for
from .experiment_launcher import _training
from .svm_resumable_authorization import validate_authorization
from .svm_resumable_store import (
    CheckpointStore, atomic_json, digest, file_sha, read_json, utc_now, verified_copy,
)


class SessionDeadline:
    def __init__(self, seconds: int, margin: int, *, clock=time.monotonic,
                 deadline_epoch: float | None = None, wall_clock=time.time):
        self.clock, self.started = clock, clock()
        self.seconds, self.margin = seconds, margin
        self.ends = self.started + (seconds if deadline_epoch is None
                                   else max(0.0, deadline_epoch - wall_clock()))

    def may_start(self) -> bool:
        return self.ends - self.clock() > self.margin


class DurableProgress:
    """Low-frequency Drive heartbeat plus immediate durable stage/fit events."""
    def __init__(self, root: Path, session_id: str, interval: int = 60):
        self.root, self.session_id, self.interval = root, session_id, interval
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.started = self.stage_started = time.monotonic()
        self.stage, self.current_k, self.completed_k = "initializing", None, 0
        self.last_checkpoint = self.last_error = self.logging_error = None
        self.proc = psutil.Process()
        self.proc.cpu_percent()
        self.peak_rss = 0
        self.prior_observed_seconds = 0.0
        previous_path = self.root / "progress.json"
        if previous_path.is_file():
            previous = read_json(previous_path)
            if previous.get("session_id") != session_id:
                self.prior_observed_seconds = float(previous.get("total_elapsed_seconds", 0.0))

    def __enter__(self):
        self("worker_started", pid=os.getpid())
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __call__(self, event: str, **details):
        with self.lock:
            if self.logging_error:
                raise OSError(f"Durable progress unavailable: {self.logging_error}")
            if event != "heartbeat":
                self.stage, self.stage_started = event, time.monotonic()
            if "k" in details:
                self.current_k = details["k"]
            if "completed_k" in details:
                self.completed_k = details["completed_k"]
            if "checkpoint" in details:
                self.last_checkpoint = details["checkpoint"]
            if "error" in details:
                self.last_error = details["error"]
            tree = [self.proc, *self.proc.children(recursive=True)]
            rss = 0
            for proc in tree:
                try:
                    rss += proc.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            try:
                process_status = self.proc.status()
            except psutil.NoSuchProcess:
                process_status = "exited"
            record = {
                "utc": utc_now(), "event": event, "stage": self.stage,
                "session_id": self.session_id, "worker_pid": os.getpid(),
                "process_status": process_status, "current_k": self.current_k,
                "completed_k": self.completed_k, "elapsed_seconds": time.monotonic() - self.started,
                "total_elapsed_seconds": self.prior_observed_seconds + time.monotonic() - self.started,
                "total_elapsed_is_observed_worker_time_not_idle_between_sessions": True,
                "stage_elapsed_seconds": time.monotonic() - self.stage_started,
                "tree_rss_bytes": rss, "peak_tree_rss_bytes": self.peak_rss,
                "available_ram_bytes": psutil.virtual_memory().available,
                "cpu_percent": self.proc.cpu_percent(),
                "last_completed_checkpoint": self.last_checkpoint,
                "last_error_or_stop_reason": self.last_error, "details": details,
            }
            destination = self.root / ("heartbeat.jsonl" if event == "heartbeat" else "events.jsonl")
            import json
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_json(self.root / "progress.json", record)
            print(f"[{record['utc']}] {self.stage} k={self.current_k} "
                  f"completed={self.completed_k}/100 elapsed={record['elapsed_seconds']:.0f}s "
                  f"stage_elapsed={record['stage_elapsed_seconds']:.0f}s "
                  f"pid={os.getpid()} rss={rss/1024**3:.2f}GiB "
                  f"cpu={record['cpu_percent']:.1f}% last={self.last_checkpoint} "
                  f"error={self.last_error}", flush=True)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self("heartbeat")
            except Exception as exc:
                self.logging_error = str(exc)
                print(f"HEARTBEAT_WRITE_FAILED: {exc}", flush=True)
                return

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def _array_metadata(directory: Path) -> dict:
    return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)}
            for p in directory.iterdir() if p.suffix == ".npy"}


def _reuse_local(directory: Path, signature: str):
    if not directory.exists():
        return None
    manifest = read_json(directory / "cache.json")
    if manifest.get("signature") != signature:
        raise ValueError("Local prepared matrix identity differs; use a fresh LOCAL_RUNS path")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Invalid local cache manifest")
    if {p.name for p in directory.iterdir()} != set(files) | {"cache.json"}:
        raise ValueError("Prepared memmap cache has missing/extra/unverified files")
    for name, meta in files.items():
        if Path(name).name != name:
            raise ValueError("Unsafe local cache filename")
        p = directory / name
        if (not p.is_file() or p.stat().st_size != meta["bytes"]
                or file_sha(p) != meta["sha256"]):
            raise ValueError("Prepared memmap checksum mismatch")
    return manifest


def prepared_split(training, store, config: str, role: str, cache_root: Path,
                   binding_sha: str, progress):
    if role not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    snapshot = {str(f.path.relative_to(store.root)): {
        "bytes": f.path.stat().st_size, "mtime_ns": f.path.stat().st_mtime_ns}
        for f in store.files[config] if f.role == role}
    signature = digest({"binding": binding_sha, "split": role, "files": snapshot})
    directory = cache_root / role
    cached = _reuse_local(directory, signature)
    if cached is None:
        temporary = directory.with_name(directory.name + ".incomplete-" + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        data = store.load_full(config, role, temporary, progress=progress)
        for array in (data.X, data.y):
            if hasattr(array, "flush"):
                array.flush()
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        cached = {"signature": signature, "files": _array_metadata(temporary)}
        atomic_json(temporary / "cache.json", cached, exclusive=True)
        os.rename(temporary, directory)
    else:
        progress("prepared_split_reused", split=role)
    X = np.load(directory / f"{role}_X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / f"{role}_y.npy", mmap_mode="r", allow_pickle=False)
    data = training.PreparedSplit(role, X, y)
    data.validate(both_classes=role == "train")
    expected = sum(f.count for f in store.files[config] if f.role == role)
    if len(data.y) != expected:
        raise ValueError("Cached prepared split has wrong full row count")
    return data


def _scaled_top100(training, train, val, mi, scaler, cache: Path,
                   mi_reference: str, progress):
    signature = digest({"mi": mi_reference, "train": len(train.y), "val": len(val.y)})
    directory = cache / "top100"
    if _reuse_local(directory, signature) is None:
        temporary = directory.with_name(directory.name + ".incomplete-" + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        # Same scaler and float32 transform as scale_top100; never fit on resume.
        for role, data in (("train", train), ("val", val)):
            transformed = scaler.transform(data.X[:, mi["indices"]])
            if transformed.dtype != np.float32 or not training.finite_matrix(transformed):
                raise ValueError("Top100 matrix dtype/nonfinite mismatch")
            np.save(temporary / f"{role}.npy", transformed, allow_pickle=False)
            del transformed
        atomic_json(temporary / "cache.json", {
            "signature": signature, "files": _array_metadata(temporary)}, exclusive=True)
        os.rename(temporary, directory)
    progress("top100_matrices_ready")
    return (np.load(directory / "train.npy", mmap_mode="r", allow_pickle=False),
            np.load(directory / "val.npy", mmap_mode="r", allow_pickle=False))


def choose_best(rows: list[dict]) -> dict:
    if [row.get("k") for row in rows] != list(range(1, 101)):
        raise ValueError("All k=1..100 must complete before choosing best_k")
    best = rows[0]
    for row in rows[1:]:
        if row["metrics"]["accuracy"] > best["metrics"]["accuracy"]:
            best = row
    return best


def _pause(progress, stage: str) -> dict:
    progress("session_paused_at_checkpoint", next_stage=stage,
             error="APPROACHING_APPROVED_SESSION_DEADLINE")
    return {"status": "PAUSED_AT_SAFE_CHECKPOINT", "next_stage": stage,
            "test_evaluation_calls_this_session": 0}


def verified_selection(checkpoints: CheckpointStore) -> dict:
    """Verify the complete preprocessing -> 100 k -> frozen selection chain."""
    data_ref = checkpoints.reference("data_prepared")
    checkpoints.load_required("mi", parents={"data": data_ref})
    mi_ref = checkpoints.reference("mi")
    mi = checkpoints.data("mi")
    rfe = checkpoints.data("rfe", parents={"mi": mi_ref})
    ranked = rfe.get("features")
    if (not isinstance(ranked, list) or len(ranked) != 100 or len(set(ranked)) != 100
            or set(ranked) != set(mi.get("features", []))
            or rfe.get("complete_ranking") is not True or rfe.get("forest_fits") != 100):
        raise ValueError("Incomplete or inconsistent saved RFE ranking")
    parents = {"mi": mi_ref, "rfe": checkpoints.reference("rfe")}
    rows = checkpoints.completed_k(parents, ranked)
    best = choose_best(rows)
    frozen = checkpoints.data("selection", parents={**parents, "k_history": digest(rows)})
    if frozen.get("best_k") != best["k"] or frozen.get("features") != best["features"]:
        raise ValueError("Frozen selection differs from complete Validation history")
    final = checkpoints.data("final", parents={"selection": checkpoints.reference("selection")})
    if final.get("features") != frozen["features"]:
        raise ValueError("Final model has another feature prefix")
    return frozen


def verify_publication(destination: Path, sources: dict, binding_sha256: str) -> dict:
    manifest = read_json(destination / "artifact_manifest.json")
    files = manifest.get("files")
    if (manifest.get("binding_sha256") != binding_sha256
            or not isinstance(files, dict) or set(files) != set(sources)):
        raise ValueError("Published manifest is incomplete or has another binding")
    if {p.name for p in destination.iterdir()} != set(sources) | {"artifact_manifest.json"}:
        raise ValueError("Published directory has missing/extra artifacts")
    for name, source in sources.items():
        meta = files[name]
        path = destination / name
        if (not isinstance(meta, dict) or path.is_symlink() or not path.is_file()
                or path.stat().st_size != meta.get("bytes")
                or source.stat().st_size != meta.get("bytes")
                or file_sha(path) != meta.get("sha256")
                or file_sha(source) != meta.get("sha256")):
            raise ValueError(f"Published artifact differs from verified checkpoint: {name}")
    return manifest


def publish_results(checkpoints: CheckpointStore) -> Path:
    """Publish completed bundles only. No training or Test access here."""
    verified_selection(checkpoints)
    final_ref = checkpoints.reference("final")
    checkpoints.load_required("final_test", parents={"final": final_ref})
    claim = read_json(checkpoints.root / "results" / "test_evaluation_claim.json")
    if (claim.get("binding_sha256") != checkpoints.binding_sha256
            or claim.get("final_checkpoint_sha256") != final_ref
            or claim.get("test_evaluation_number") != 1):
        raise ValueError("Durable Test claim does not match final result")
    sources = {
        "result.json": checkpoints.directory("final_test") / "result.json",
        "final_test_metrics.json": checkpoints.directory("final_test") / "data.json",
        "final_model.joblib": checkpoints.directory("final") / "final_model.joblib",
        "final_scaler.joblib": checkpoints.directory("final") / "final_scaler.joblib",
        "best_k.json": checkpoints.directory("selection") / "data.json",
        "run_manifest.json": checkpoints.root / "run_manifest.json",
    }
    result = read_json(sources["result.json"])
    if (result.get("binding_sha256") != checkpoints.binding_sha256
            or result.get("test_evaluation_calls") != 1
            or result.get("status") not in ("PASS", "COMPLETED_WITH_CONVERGENCE_WARNING")
            or result.get("rows_by_split") != checkpoints.binding["rows_by_split"]):
        raise ValueError("Final result identity/counts/status differ")
    destination = checkpoints.root / "results" / "completed"
    if destination.exists():
        verify_publication(destination, sources, checkpoints.binding_sha256)
        return destination
    temporary = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    for name, source in sources.items():
        verified_copy(source, temporary / name)
    manifest = {"binding_sha256": checkpoints.binding_sha256,
                "files": {name: {"bytes": (temporary / name).stat().st_size,
                                 "sha256": file_sha(temporary / name)} for name in sources},
                "matrix_cache_saved": False, "no_test_reevaluation": True}
    atomic_json(temporary / "artifact_manifest.json", manifest, exclusive=True)
    os.rename(temporary, destination)
    return publish_results(checkpoints)


def run_resumable(repo: Path, windows: Path, local_runs: Path, authorization) -> dict:
    context = validate_authorization(authorization)
    spec = context.base.identity
    training = _training(repo)
    checkpoints = CheckpointStore(context.root, context.binding, resume=context.resume)
    config, session_id = spec["config"], context.policy["session_id"]
    deadline = SessionDeadline(context.policy["session_budget_seconds"],
                               context.policy["stop_starting_new_work_margin_seconds"],
                               deadline_epoch=getattr(context, "deadline_epoch", None))
    session_started = time.monotonic()
    with DurableProgress(context.root, session_id,
                         context.policy["heartbeat_interval_seconds"]) as progress:
        try:
            if checkpoints.load("final_test") is not None:
                published = publish_results(checkpoints)
                progress("completed_result_reused", checkpoint="final_test")
                return {"status": "COMPLETED_VERIFIED", "result_directory": str(published),
                        "test_evaluation_calls_this_session": 0}
            if (context.root / "results" / "test_evaluation_claim.json").exists():
                raise PermissionError("Test was already claimed without a verified result; manual review required")
            store = training.WindowStore(windows, split_root=manifest_root_for(windows, repo))
            training.source_hashes(repo / "CODE_ctu13")
            training.source_audit(repo / "CODE_ctu13")
            before = store.snapshot()
            cache = local_runs / "resumable_cache" / checkpoints.binding_sha256
            cache.mkdir(parents=True, exist_ok=True)
            final_record = checkpoints.load("final")
            if final_record is None:
                if not deadline.may_start():
                    return _pause(progress, "data_preparation")
                progress("data_preparation_started")
                train = prepared_split(training, store, config, "train", cache,
                                       checkpoints.binding_sha256, progress)
                val = prepared_split(training, store, config, "val", cache,
                                     checkpoints.binding_sha256, progress)
                training.require_selection_roles(train, val)
                expected = context.binding["rows_by_split"]
                if len(train.y) != expected["train"] or len(val.y) != expected["val"]:
                    raise ValueError("Full Train/Validation count mismatch")
                if checkpoints.load("data_prepared") is None:
                    checkpoints.save("data_prepared", {
                        "train_rows": len(train.y), "val_rows": len(val.y),
                        "feature_names": list(training.FEATURE_NAMES),
                        "test_feature_rows_read": 0,
                        "label_counts": {d.role: {"benign": int((d.y == 0).sum()),
                                                   "attack": int((d.y == 1).sum())}
                                         for d in (train, val)},
                    })
                data_ref = checkpoints.reference("data_prepared")
                mi_parents = {"data": data_ref}
                mi_record = checkpoints.load("mi", parents=mi_parents)
                if mi_record is None:
                    if not deadline.may_start():
                        return _pause(progress, "mi")
                    progress("mi_fit_started", fit_started_utc=utc_now())
                    mi = training.select_mi(train)
                    # Fit Top100 scaler using the exact canonical function once.
                    X_train, X_val, top_scaler = training.scale_top100(train, val, mi)
                    del X_train, X_val
                    checkpoints.save("mi", {
                        "features": mi["names"], "indices": mi["indices"],
                        "measurement": mi["measurement"], "fit_role": "train",
                    }, parents=mi_parents, writers={
                        "mi.joblib": lambda p: joblib.dump(mi, p),
                        "top100_scaler.joblib": lambda p: joblib.dump(top_scaler, p),
                        "mi_feature_scores.csv": lambda p: mi["scores"].to_csv(p),
                    })
                    del mi, top_scaler
                    progress("mi_completed", checkpoint="mi", fit_finished_utc=utc_now())
                mi_ref = checkpoints.reference("mi")
                mi = joblib.load(checkpoints.directory("mi") / "mi.joblib")
                top_scaler = joblib.load(checkpoints.directory("mi") / "top100_scaler.joblib")
                mi_meta = checkpoints.data("mi", parents=mi_parents)
                if (mi["names"] != mi_meta["features"] or len(mi["names"]) != 100
                        or len(set(mi["names"])) != 100
                        or mi["indices"] != [training.FEATURE_NAMES.index(n) for n in mi["names"]]
                        or int(top_scaler.n_features_in_) != 100):
                    raise ValueError("MI/preprocessing order mismatch")
                X_train, X_val = _scaled_top100(training, train, val, mi, top_scaler,
                                               cache, mi_ref, progress)
                rfe_parents = {"mi": mi_ref}
                if checkpoints.load("rfe", parents=rfe_parents) is None:
                    if not deadline.may_start():
                        return _pause(progress, "rfe")
                    progress("rfe_fit_started", target=1, step=1, fit_started_utc=utc_now())
                    ranking = training.rank_rfe(X_train, train.y, mi["names"], "svm", target=1)
                    checkpoints.save("rfe", {
                        "features": ranking["names"],
                        "measurement": ranking["measurement"],
                        "complete_ranking": ranking["complete_ranking"],
                        "forest_fits": ranking["forest_fits"],
                    }, parents=rfe_parents, writers={
                        "rfe_feature_ranking.csv": lambda p: ranking["ranks"].to_csv(p),
                    })
                    progress("rfe_completed", checkpoint="rfe", fit_finished_utc=utc_now())
                rfe = checkpoints.data("rfe", parents=rfe_parents)
                ranked = rfe["features"]
                if (len(ranked) != 100 or set(ranked) != set(mi["names"])
                        or rfe.get("complete_ranking") is not True or rfe.get("forest_fits") != 100):
                    raise ValueError("Full sklearn RFE ranking is required")
                k_parents = {"mi": mi_ref, "rfe": checkpoints.reference("rfe")}
                rows = checkpoints.completed_k(k_parents, ranked)
                progress("k_search_resumed", completed_k=len(rows),
                         checkpoint=f"k_search/k_{len(rows):03d}" if rows else "rfe")
                for k in range(len(rows) + 1, 101):
                    if not deadline.may_start():
                        return _pause(progress, f"k={k}")
                    started_utc = utc_now()
                    progress("k_fit_starting", k=k, completed_k=len(rows),
                             fit_started_utc=started_utc)
                    selection = training.search_k(
                        X_train, train.y, X_val, val.y, mi["names"], ranked,
                        "svm", k_values=(k,), progress=progress)
                    row = selection["results"][0]
                    finished_utc = utc_now()
                    saved = {
                        "k": k, "features": ranked[:k],
                        "metrics": {name: value for name, value in row.items()
                                    if name not in ("k", "fit", "predict_seconds")},
                        "fit": row["fit"], "predict_seconds": row["predict_seconds"],
                        "fit_started_utc": started_utc, "fit_finished_utc": finished_utc,
                        "train_rows": len(train.y), "validation_rows": len(val.y),
                    }
                    stage = f"k_search/k_{k:03d}"
                    checkpoints.save(stage, saved, parents=k_parents)
                    rows.append(saved)
                    progress("k_checkpoint_committed", k=k, completed_k=len(rows),
                             checkpoint=stage, validation_accuracy=saved["metrics"]["accuracy"],
                             fit_started_utc=started_utc, fit_finished_utc=finished_utc)
                best = choose_best(rows)
                selection_parents = {**k_parents, "k_history": digest(rows)}
                if checkpoints.load("selection", parents=selection_parents) is None:
                    checkpoints.save("selection", {
                        "best_k": best["k"], "features": ranked[:best["k"]],
                        "best_validation_accuracy": best["metrics"]["accuracy"],
                        "k_results": rows,
                        "criterion": "validation_accuracy_strict_greater_smallest_k_tie",
                    }, parents=selection_parents)
                frozen = checkpoints.data("selection", parents=selection_parents)
                progress("selection_frozen", checkpoint="selection", best_k=frozen["best_k"])
                del X_train, X_val, mi, top_scaler
                if not deadline.may_start():
                    return _pause(progress, "final_fit")
                progress("final_refit_started", fit_started_utc=utc_now())
                fitted = training.fit_final_classifier(train, val, "svm", frozen["features"])
                if fitted["fit_rows"] != expected["train"] + expected["val"]:
                    raise ValueError("Final model did not fit all Train+Validation rows")
                meta = {name: value for name, value in fitted.items() if name not in ("model", "scaler")}
                checkpoints.save("final", meta,
                                 parents={"selection": checkpoints.reference("selection")}, writers={
                    "final_model.joblib": lambda p: joblib.dump(fitted["model"], p),
                    "final_scaler.joblib": lambda p: joblib.dump(fitted["scaler"], p),
                })
                progress("final_refit_completed", checkpoint="final", fit_finished_utc=utc_now())
                del fitted, train, val
            # Validate the whole completed selection chain even on final-model-only resume.
            verified_selection(checkpoints)
            mi_ref = checkpoints.reference("mi")
            rfe = checkpoints.data("rfe", parents={"mi": mi_ref})
            k_parents = {"mi": mi_ref, "rfe": checkpoints.reference("rfe")}
            rows = checkpoints.completed_k(k_parents, rfe["features"])
            best = choose_best(rows)
            frozen = checkpoints.data("selection", parents={**k_parents, "k_history": digest(rows)})
            if frozen["best_k"] != best["k"] or frozen["features"] != best["features"]:
                raise ValueError("Frozen selection differs from completed Validation history")
            final_meta = checkpoints.data("final", parents={"selection": checkpoints.reference("selection")})
            if final_meta["features"] != frozen["features"]:
                raise ValueError("Final model uses another feature prefix")
            progress("final_model_verified", completed_k=100, checkpoint="final")
            expected_rows = context.binding["rows_by_split"]
            if (final_meta.get("fit_role") != "train+val"
                    or final_meta.get("fit_rows") != expected_rows["train"] + expected_rows["val"]
                    or final_meta.get("mi_rfe_refit") is not False):
                raise ValueError("Final checkpoint does not satisfy the full Train+Validation refit contract")
            if not deadline.may_start():
                return _pause(progress, "final_test")
            fitted = dict(final_meta)
            fitted["model"] = joblib.load(checkpoints.directory("final") / "final_model.joblib")
            fitted["scaler"] = joblib.load(checkpoints.directory("final") / "final_scaler.joblib")
            params = fitted["model"].get_params()
            if any(params.get(key) != value for key, value in training.COMPACT_SVM_PARAMS.items()):
                raise ValueError("Saved final SVC hyperparameters differ")
            if int(fitted["scaler"].n_features_in_) != len(frozen["features"]):
                raise ValueError("Final scaler feature count differs")
            if int(fitted["model"].n_features_in_) != len(frozen["features"]):
                raise ValueError("Final SVC feature count differs")
            final_ref = checkpoints.reference("final")
            if not deadline.may_start():
                return _pause(progress, "final_test")
            if store.snapshot() != before:
                raise ValueError("Window size/mtime changed before Test; evaluation blocked")
            training.source_hashes(repo / "CODE_ctu13")
            checkpoints.claim_test_once(final_ref, session_id)
            progress("test_evaluation_started", evaluation_number=1)
            # The durable claim precedes the FIRST Test loader call. Failure cannot auto-retry.
            test = prepared_split(training, store, config, "test", cache,
                                  checkpoints.binding_sha256, progress)
            final = training.evaluate_final(fitted, test)
            if len(test.y) != context.binding["rows_by_split"]["test"]:
                raise ValueError("Full Test count mismatch")
            if store.snapshot() != before:
                raise ValueError("Source window size/mtime changed during this session")
            training.source_hashes(repo / "CODE_ctu13")
            result = {name: value for name, value in final.items() if name not in ("model", "scaler")}
            nonconverged = [row["k"] for row in rows if row["fit"].get("converged") is False]
            result.update(
                experiment="mi_rfe_svm", config=config, run_name=spec["run_name"],
                status="COMPLETED_WITH_CONVERGENCE_WARNING" if nonconverged
                or final["fit"].get("converged") is False else "PASS",
                rows_by_split=context.binding["rows_by_split"], best_k=frozen["best_k"],
                validation_accuracy=frozen["best_validation_accuracy"],
                nonconverged_selection_k=nonconverged, sampled_training=False,
                test_read_after_final_refit=True, label_origin="SCENARIO_PROXY",
                capture_session_independence="NOT_ESTABLISHED",
                scientific_contract=context.binding["scientific_contract"],
                binding_sha256=checkpoints.binding_sha256,
                label_counts_by_split={
                    **checkpoints.data("data_prepared")["label_counts"],
                    "test": {"benign": int((test.y == 0).sum()), "attack": int((test.y == 1).sum())},
                },
                session_elapsed_seconds=time.monotonic() - session_started,
                peak_tree_rss_bytes_sampled=progress.peak_rss,
                stage_runtimes_seconds={
                    "mi": checkpoints.data("mi")["measurement"]["seconds"],
                    "rfe": rfe["measurement"]["seconds"],
                    "k_search_fit_sum": sum(row["fit"]["seconds"] for row in rows),
                    "final_refit": final["fit"]["seconds"],
                    "test_predict": final["predict_seconds"],
                },
            )
            checkpoints.save("final_test", {"metrics": final["metrics"], "test_evaluation_calls": 1},
                             parents={"final": final_ref}, writers={
                "result.json": lambda p: atomic_json(p, result, exclusive=True),
            })
            progress("test_evaluation_completed", checkpoint="final_test")
            published = publish_results(checkpoints)
            progress("run_validated_and_artifacts_verified", checkpoint="final_test")
            return {"status": "COMPLETED_VERIFIED", "result_directory": str(published),
                    "best_k": frozen["best_k"], "test_evaluation_calls_this_session": 1}
        except BaseException as exc:
            try:
                progress("worker_failed", error=f"{type(exc).__name__}: {exc}")
            except OSError:
                pass
            raise
