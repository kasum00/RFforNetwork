"""Selection and finalization in explicit user-launched CPU workers."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from .common import (Blocked, FORMAT, SCIENTIFIC, DRIVE_CHECKPOINT_INTERVAL_SECONDS, atomic, check_files, core, digest,
                     drive_json_once, info, paths, read, redact, sha, utc)
from .artifact_reuse import import_selection, validate_selection
from .dataset import cache_arrays, inspect, raw_split, svc_prefix
from .drive_checkpoint import DriveCheckpoint, latest, restore, state_files
from .local_progress import KProgress, choose
from .local_progress import LocalProgress
from .preflight import review

MARKER = "__SVM5H_RESULT__"


class Monitor:
    """Local-only resource sampling; guard checked at safe boundaries."""
    def __init__(self, budget):
        import psutil
        self.psutil, self.proc, self.budget = psutil, psutil.Process(), budget
        self.done, self.peak, self.low, self.reason = threading.Event(), 0, None, None

    def sample(self):
        rss = self.proc.memory_info().rss
        available = self.psutil.virtual_memory().available
        self.peak = max(self.peak, rss)
        self.fit_peak = max(getattr(self, "fit_peak", 0), rss)
        self.low = available if self.low is None else min(self.low, available)
        if rss > self.budget["max_process_rss_bytes"] or available < self.budget["min_available_ram_bytes"]:
            self.reason = "Observed RAM/RSS budget breached; no next fit permitted"
        return rss

    def loop(self):
        while not self.done.wait(0.5):
            try:
                self.sample()
            except Exception as exc:
                self.reason = "Resource monitor failed: " + str(exc)
                return

    def check(self):
        self.sample()
        if self.reason:
            raise Blocked(self.reason)
        if shutil.disk_usage(self.local).free < 1024**3:
            raise Blocked("Local disk below 1 GiB safe-boundary headroom")

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        value = redact(value)
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def scaled_top(request, state, local, identity, train, y, val, vy, top):
    import joblib
    from sklearn.preprocessing import StandardScaler
    module = core(request["source"])
    indices = [identity["dataset"]["names"].index(n) for n in top]
    binding = {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top}
    scaler_path, marker = state / "top100_scaler.joblib", state / "top100_scaler_state.json"
    saved = read(marker) if marker.exists() else None
    if saved:
        if saved["binding"] != binding:
            raise Blocked("Top100 scaler identity differs")
        check_files(state, saved["files"])
    elif scaler_path.exists():
        raise Blocked("Uncommitted Top100 scaler; preserve for review")

    def build():
        if saved:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(train[:, indices])
        else:
            scaler = StandardScaler()
            # Preserve reference advanced-indexing layout and float32 arithmetic.
            X = scaler.fit_transform(train[:, indices])
        V = scaler.transform(val[:, indices])
        if not saved:
            module.dump_model(scaler_path, scaler)
            atomic(marker, {"binding": binding, "files": {scaler_path.name: info(scaler_path)},
                            "fit_role": "train", "rows": len(y)}, exclusive=True)
        return {"train": X, "val": V}, {"float32_scaler_output": True}

    arrays, record = cache_arrays(local / "cache/scaled_top100", binding, build)
    if not marker.exists():
        raise Blocked("Cached transforms without committed scaler")
    return arrays["train"], arrays["val"]


def ranked_arrays(local, identity, X, V, top, ranked):
    import numpy as np
    order = [top.index(n) for n in ranked]
    def build():
        return {"train": np.ascontiguousarray(X[:, order], dtype=np.float32),
                "val": np.ascontiguousarray(V[:, order], dtype=np.float32)}, {"order": ranked}
    arrays, _ = cache_arrays(local / "cache/ranked_top100",
        {"identity": identity["sha256"], "top": top, "ranked": ranked}, build)
    return arrays["train"], arrays["val"]


def search(state, identity, ranked, X, y, V, vy, progress, *, model_factory, fit_fn,
           metrics_fn, boundary, stop_requested=lambda: False, rss=lambda: 0):
    """Sequential 1..100. Injected factories let tests use fake SVC, never real fits."""
    history = KProgress(state, identity, ranked, sha(Path(state) / "rfe_state.json"))
    for k in range(len(history.rows)+1, 101):
        if stop_requested():
            boundary(force=True, reason="graceful_stop_before_next_k")
            return None
        progress.check()
        progress.event("k_started", k=k, completed_k=len(history.rows))
        started = utc()
        train_k = svc_prefix(X, k)
        val_k = svc_prefix(V, k)
        rss_before = rss()
        model = model_factory()
        model, fit = fit_fn(model, train_k, y)
        prediction_started = time.perf_counter()
        predicted = model.predict(val_k)
        prediction_seconds = time.perf_counter()-prediction_started
        row = {"k": k, "features": ranked[:k], **metrics_fn(vy, predicted), "fit": fit,
               "predict_seconds": prediction_seconds, "started": started, "finished": utc(),
               "rss_before_fit_bytes": rss_before, "rss_after_predict_bytes": rss()}
        del model, train_k, val_k, predicted
        gc.collect()
        history.append(row)
        best = choose(history.rows, ranked)
        progress.event("k_completed", k=k, completed_k=k, fit_seconds=fit["seconds"],
                       validation_accuracy=row["accuracy"], best_k=best["k"],
                       warnings=fit.get("warnings", []), converged=fit.get("converged"))
        boundary(force=stop_requested(), reason="graceful_stop" if stop_requested() else "interval_after_k")
        if stop_requested():
            return None
    return history.rows


def freeze(state, identity, rows, ranked, module):
    winner = choose(rows, ranked, complete=True)
    value = {"best_k": winner["k"], "best_validation_accuracy": winner["accuracy"],
             "features": ranked[:winner["k"]], "results": rows,
             "nonconverged_k": [r["k"] for r in rows if r["fit"].get("converged") is False]}
    path = state / "k_search_results.json"
    if path.exists():
        if read(path) != value:
            raise Blocked("Saved selection differs")
    else:
        atomic(path, value, exclusive=True)
    return module.frozen_selection(state, identity, value)


def validate_final(state, identity):
    from .verification import validate_state
    validate_state(state, identity, final_required=True)
    return read(Path(state) / "selection_frozen.json"), read(Path(state) / "final_state.json")


def test_once(request, state, local, remote, identity, progress, checkpoint):
    """Durable claim is outside the 5h schedule: loss can never trigger a new Test."""
    import joblib
    module = core(request["source"])
    frozen, final = validate_final(state, identity)
    claim_path = remote / "TEST_CLAIM.json"
    if (state / "test_result.json").exists():
        # Recovery only: metadata validation/publication, never another predict.
        from .verification import validate_state
        validate_state(state, identity, final_required=True, test_required=True)
        if read(claim_path) != read(state / "TEST_CLAIM.json"):
            raise Blocked("Durable/local Test claims differ")
        return read(state / "test_result.json")
    if claim_path.exists() or (state / "TEST_CLAIM.json").exists():
        raise Blocked("Test was already claimed without a verified result; automatic retry forbidden")
    if inspect(request) != identity["dataset_audit"]:
        raise Blocked("Dataset fingerprint changed before Test")
    # Completed final model/scaler must already be durable before the Test claim.
    checkpoint(force=True, reason="final_refit_before_test")
    model, scaler = joblib.load(state / "model.joblib"), joblib.load(state / "final_scaler.joblib")
    if (any(model.get_params().get(k) != v for k, v in module.SVM_PARAMS.items())
            or model.n_features_in_ != len(frozen["features"]) or scaler.n_features_in_ != len(frozen["features"])):
        raise Blocked("Final model/scaler contract mismatch")
    claim = {"format": FORMAT, "identity": identity["sha256"], "final_state": digest(final),
             "evaluation_number": 1, "started": utc(), "retry_forbidden": True}
    drive_json_once(claim_path, claim)  # Must finish and read back BEFORE any Test values.
    atomic(state / "TEST_CLAIM.json", claim, exclusive=True)
    progress.event("test_started", evaluation_number=1)
    T, ty = raw_split(request, identity["dataset_audit"], local, identity, "test", test_claim=claim)
    indices = [identity["dataset"]["names"].index(n) for n in frozen["features"]]
    started = time.perf_counter()
    prediction = model.predict(scaler.transform(T[:, indices]))
    result = {"identity": identity["sha256"], "claim": digest(claim), "test_evaluation_calls": 1,
              "test_rows_evaluated": len(ty), "metrics": module.metrics(ty, prediction),
              "predict_seconds": time.perf_counter()-started, "finished": utc()}
    atomic(state / "test_result.json", result, exclusive=True)
    del model, scaler, T, ty, prediction
    progress.event("test_completed", evaluation_number=1)
    return result


def execute(request):
    mode = request.get("mode", "preflight")
    if mode not in ("preflight", "run", "resume", "verify", "finalize"):
        raise Blocked("Unknown mode")
    local, remote = paths(request)
    if mode == "verify":
        if (remote / "COMPLETE.json").exists() and read(remote / "run_identity.json").get("format") == core(request["source"]).FORMAT:
            return {**core(request["source"]).verify_directory(remote), **SCIENTIFIC}
        from .verification import verify
        return verify(remote)
    if (remote / "RUN_COMPLETE.json").exists() and mode != "run":
        from .verification import verify
        return verify(remote)
    # A previously claimed Test can only be published or blocked, never refit.
    plan = review(request, need_fit=not (remote / "TEST_CLAIM.json").exists())
    if mode == "preflight" or plan["status"] != "PASS":
        return plan
    identity, module = plan["identity"], core(request["source"])
    if request.get("confirmation") != f"{mode.upper()} PROVISIONAL {remote.name}":
        raise Blocked("Exact run/mode confirmation required")
    local.mkdir(parents=True, exist_ok=True)
    with module.run_lock(local):
        manager = DriveCheckpoint(local, remote, identity, takeover=request.get("takeover_confirmation"))
        state = restore(local, remote, identity)
        if request.get("acknowledge_local_stop") and mode in ("resume", "finalize"):
            # Explicit acknowledgement affects only this owned local control flag.
            (local / "STOP_REQUESTED").unlink(missing_ok=True)
        for name, value in (("hyperparameters.json", identity["contract"]), ("preflight_audit.json", plan)):
            if not (state / name).exists():
                atomic(state / name, value, exclusive=True)
        # Snapshot user-run synthetic test logs once per worker, never per k.
        test_records = request.get("unit_test_artifacts", {})
        if test_records:
            test_root = state / ("unit_tests_" + uuid.uuid4().hex)
            test_root.mkdir()
            for name, record in test_records.items():
                source = Path(record["path"]).resolve()
                if (Path(name).name != name or Path(name).suffix not in (".json", ".log")
                        or info(source) != {"bytes": record["bytes"], "sha256": record["sha256"]}):
                    raise Blocked("Unit test log artifact identity differs")
                with source.open("rb") as src, (test_root / name).open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                if info(test_root / name) != info(source):
                    raise Blocked("Unit test log snapshot failed")
        with (local / "console.log").open("a", encoding="utf-8") as log, Monitor(request["budget"]) as monitor:
            monitor.local = local
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                progress = LocalProgress(state, identity, check=monitor.check)
                attempt, started = uuid.uuid4().hex, time.perf_counter()

                def checkpoint(*, force=False, reason):
                    if (reason == "final_refit_before_test" and manager.latest_record
                            and manager.latest_record.get("final_state_sha256") == sha(state / "final_state.json")):
                        return manager.latest_record  # Already verified durable; avoid copying model bundle twice.
                    manual_request = local / "CHECKPOINT_REQUESTED"
                    if manual_request.exists():
                        force, reason = True, "user_requested_safe_boundary"
                    if not force and not manager.due():
                        previous = manager.latest_record["finished"] if manager.latest_record else "NOT_YET_COMMITTED"
                        due_in = max(0, DRIVE_CHECKPOINT_INTERVAL_SECONDS - (manager.clock()-manager.last_success))
                        print(f"LAST_DRIVE_CHECKPOINT: {previous}; NEXT_CHECKPOINT_DUE_IN_SECONDS: {due_in:.0f}", flush=True)
                        return None
                    log.flush()
                    shutil.copyfile(local / "console.log", state / "console_snapshot.log")
                    atomic(state / "resource_usage.json", {"attempt": attempt, "peak_rss_bytes_sampled_this_attempt": monitor.peak,
                        "min_available_ram_bytes_this_attempt": monitor.low, "seconds_this_attempt": time.perf_counter()-started,
                        "measurement_scope": "this worker attempt; not reconstructed historical RAM", "utc": utc()})
                    record = manager.sync(state, force=force, reason=reason, safe_boundary=True)
                    if record is not None:
                        manual_request.unlink(missing_ok=True)
                    return record

                def stop_requested():
                    return (local / "STOP_REQUESTED").exists()

                try:
                    progress.event("attempt_started", mode=mode, config=request["config"], **SCIENTIFIC)
                    if mode == "finalize" and not (state / "selection_frozen.json").exists():
                        raise Blocked("Complete k=1..100 and freeze selection before finalize")
                    if (remote / "TEST_CLAIM.json").exists():
                        if not (state / "test_result.json").exists():
                            raise Blocked("Ambiguous Test claim: no automatic re-evaluation")
                        test_once(request, state, local, remote, identity, progress, checkpoint)
                        from .verification import publish, result_record
                        if not (state / "result.json").exists():
                            atomic(state / "result.json", result_record(state, identity), exclusive=True)
                        checkpoint(force=True, reason="completed_test_recovery")
                        result = publish(state, remote, identity)
                        manager.release("COMPLETE_RECOVERED")
                        return result

                    if mode in ("run", "resume") and (state / "selection_frozen.json").exists():
                        from .verification import validate_state
                        validate_state(state, identity)
                        manager.release("SELECTION_ALREADY_FROZEN")
                        return {"status": "SELECTION_FROZEN", "directory": str(remote), **SCIENTIFIC}
                    if mode == "finalize" and request.get("phase", "final_refit") == "test" and not (state / "final_state.json").exists():
                        raise Blocked("Run final_refit first; Test phase cannot start a refit")
                    needs_data = mode != "finalize" or not (state / "final_state.json").exists()
                    if needs_data:
                        monitor.check()
                        train, y = raw_split(request, plan["audit"], local, identity, "train")
                        if stop_requested():
                            checkpoint(force=True, reason="graceful_stop_after_loading")
                            manager.release("PAUSED")
                            return {"status": "PAUSED_AT_SAFE_CHECKPOINT", "completed": False, **SCIENTIFIC}
                        # MI/RFE fit on Train only; Validation is transformed, never fitted.
                        import_selection(state, identity, identity["reuse"])
                        top, mi = module.saved_mi(state, identity, train, y, identity["dataset"]["names"], progress)
                        if not manager.latest_record or manager.latest_record.get("mi_state_sha256") != sha(state / "mi_state.json"):
                            checkpoint(force=True, reason="mi_complete")
                        if stop_requested():
                            checkpoint(force=True, reason="graceful_stop_after_mi")
                            manager.release("PAUSED")
                            return {"status": "PAUSED_AT_SAFE_CHECKPOINT", "completed": False, **SCIENTIFIC}
                        # Fit Train Top100 once; Validation transform later. Reuse scaler/cache on resume.
                        from sklearn.preprocessing import StandardScaler
                        import joblib
                        import numpy as np
                        if mode != "finalize":
                            # RFE uses the same Train scaler output as the reference.
                            # scaled_top needs Validation only for transform, never for fitting MI/RFE.
                            val, vy = raw_split(request, plan["audit"], local, identity, "val")
                            X, V = scaled_top(request, state, local, identity, train, y, val, vy, top)
                            ranked, rfe = module.saved_rfe(state, identity, top, mi, X, y, progress)
                            validate_selection(state, identity)
                            if (not manager.latest_record
                                    or manager.latest_record.get("rfe_state_sha256") != sha(state / "rfe_state.json")
                                    or manager.latest_record.get("top100_scaler_sha256") != sha(state / "top100_scaler_state.json")):
                                checkpoint(force=True, reason="rfe_complete_or_imported")
                            Xr, Vr = ranked_arrays(local, identity, X, V, top, ranked)
                            del X, V
                            def fit_one(model, matrix, labels):
                                monitor.fit_peak = monitor.sample()
                                fitted, measurement = module.timed_fit(model, matrix, labels)
                                monitor.sample()
                                measurement["peak_rss_bytes_sampled_during_fit"] = monitor.fit_peak
                                return fitted, measurement
                            rows = search(state, identity, ranked, Xr, y, Vr, vy, progress,
                                model_factory=lambda: module.make_model("svm"), fit_fn=fit_one,
                                metrics_fn=module.metrics, boundary=checkpoint, stop_requested=stop_requested, rss=monitor.sample)
                            del Xr, Vr
                            if rows is None:
                                manager.release("PAUSED")
                                return {"status": "PAUSED_AT_SAFE_CHECKPOINT", "completed": False, **SCIENTIFIC}
                            frozen = freeze(state, identity, rows, ranked, module)
                            progress.event("selection_frozen", k=frozen["best_k"])
                            checkpoint(force=True, reason="selection_complete")
                            manager.release("SELECTION_FROZEN")
                            return {"status": "SELECTION_FROZEN", "best_k": frozen["best_k"], "directory": str(remote), **SCIENTIFIC}
                        val, vy = raw_split(request, plan["audit"], local, identity, "val")
                    if mode == "finalize":
                        phase = request.get("phase", "final_refit")
                        if phase not in ("final_refit", "test", "all_final"):
                            raise Blocked("Unknown finalize phase")
                        from .verification import validate_state, publish
                        validate_state(state, identity)
                        frozen = read(state / "selection_frozen.json")
                        if not (state / "final_state.json").exists():
                            if phase == "test":
                                raise Blocked("Run final_refit phase first")
                            progress.check()
                            module.final_refit(state, identity, frozen, train, y, val, vy,
                                               identity["dataset"]["names"], "svm", progress)
                            del train, y, val, vy
                            checkpoint(force=True, reason="final_refit_complete")
                        validate_final(state, identity)
                        if stop_requested():
                            checkpoint(force=True, reason="graceful_stop_after_final_refit")
                            manager.release("PAUSED")
                            return {"status": "PAUSED_AT_SAFE_CHECKPOINT", "completed": False, **SCIENTIFIC}
                        if phase == "final_refit":
                            manager.release("FINAL_REFIT_SAVED")
                            return {"status": "FINAL_REFIT_SAVED", "test_evaluation_calls": 0, **SCIENTIFIC}
                        progress.check()
                        test = test_once(request, state, local, remote, identity, progress, checkpoint)
                        selection = read(state / "selection_frozen.json")
                        from .verification import result_record
                        result = result_record(state, identity)
                        atomic(state / "result.json", result, exclusive=True)
                        progress.event("run_validated", best_k=selection["best_k"], technical_status=result["status"])
                        checkpoint(force=True, reason="run_complete")
                        result = publish(state, remote, identity)
                        manager.release("COMPLETE")
                        return result
                except BaseException as exc:
                    if (remote / "RUN_COMPLETE.json").exists():
                        # Publication/cleanup errors must never append a new bundle
                        # after COMPLETE and thereby invalidate its final binding.
                        print("Post-publication error; completed artifacts preserved. Run verify: " + redact(exc),
                              file=sys.stderr, flush=True)
                        raise
                    try:
                        progress.event("attempt_stopped", error=redact(exc), completed=False)
                        checkpoint(force=True, reason="controlled_stop_or_exception")
                        manager.release("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
                    except BaseException as checkpoint_error:
                        print("CHECKPOINT NOT CONFIRMED: " + redact(checkpoint_error), file=sys.stderr, flush=True)
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Local JSON request; otherwise read JSON from stdin")
    args = parser.parse_args(argv)
    request = read(args.request) if args.request else json.load(sys.stdin)
    try:
        result = execute(request)
        print(MARKER + json.dumps(result, allow_nan=False), flush=True)
        return 0 if result.get("status") in ("PASS", "SELECTION_FROZEN", "FINAL_REFIT_SAVED", "PAUSED_AT_SAFE_CHECKPOINT") else 2
    except KeyboardInterrupt:
        print("INTERRUPTED: only completed, committed boundaries can resume", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
