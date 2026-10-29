"""Independent A1 orchestration; unchanged A0 MI/scaler/SVM functions are called."""
import contextlib
import csv
import json
import shutil
import sys
import uuid
from pathlib import Path
from .contract import (ALGORITHM, SCIENTIFIC, Blocked, atomic, check_files, confirmation,
                       core, digest, info, paths, read, sha)
from .dataset import load_selection
from .preflight import review
from .rfe import fit_step5, validate_steps, validate_result
from .results import OUTPUTS, create_selection, publish, validate_ranking, validate_selection
from .store import Store, ledger
from .telemetry import Capture, Pause, Telemetry
from mi_rfe_svm_colab.runner import scaled_top, ranked_arrays, search, freeze
from mi_rfe_svm_colab.local_progress import KProgress, choose


def run(request):
    mode = request.get("mode", "preflight")
    if mode not in ("metadata", "preflight", "run", "resume", "verify"):
        raise Blocked("A1 is selection-only; final refit and Test modes do not exist")
    plan = review(request, require_tests=mode not in ("metadata", "verify"))
    if mode in ("metadata", "preflight") or plan["status"] != "PASS":
        return plan
    local, remote = paths(request)
    if mode == "verify":
        return verify(remote, plan["identity"], local.parent)
    if request.get("confirmation") != confirmation(request):
        raise Blocked("Exact A1 mode/config/run authorization required: " + confirmation(request))
    module = core(request["source"])
    local.mkdir(parents=True, exist_ok=True)
    with module.run_lock(local):
        store = Store(local, remote, plan["identity"], resume=mode == "resume",
                      takeover=request.get("takeover_confirmation"))
        if store.complete:
            return verify(remote, store.identity, local.parent)
        state, identity = store.state, store.identity
        if request.get("acknowledge_local_stop") and mode == "resume":
            (local / "STOP_REQUESTED").unlink(missing_ok=True)
        if not (state / "run_identity.json").exists():
            atomic(state / "run_identity.json", identity, exclusive=True)
        attempt = uuid.uuid4().hex
        capture = Capture(local / "logs" / (attempt + ".log"), sys.stdout, sys.stderr)
        outcome, failure = None, None
        try:
            with contextlib.redirect_stdout(capture.stream()), contextlib.redirect_stderr(capture.stream(True)):
                with Telemetry(store, request) as progress:
                    def checkpoint(stage, files, **metadata):
                        if store.saved(stage):
                            return store.saved(stage)
                        progress.checkpoint_status = "SAVING"
                        stamp = f"{attempt}_{len(store.rows)+1:06d}"
                        snapshot = state / "telemetry" / (stamp + ".json")
                        atomic(snapshot, progress.snapshot(), exclusive=True)
                        log_part = capture.snapshot(state / "logs" / (stamp + ".log"))
                        payload = {**files, snapshot.relative_to(state).as_posix(): snapshot,
                                   log_part.relative_to(state).as_posix(): log_part}
                        ordered = (read(state / "rfe_state.json")["result"]["names"] if (state / "rfe_state.json").exists()
                            else read(state / "mi_top100.json") if (state / "mi_top100.json").exists() else identity["dataset"]["names"])
                        committed_k = [int(r["record"]["stage"][2:]) for r in store.rows
                                       if r["record"]["stage"].startswith("k_")]
                        durable_k = int(stage[2:]) if stage.startswith("k_") else max(committed_k, default=0)
                        metadata.update(feature_order_sha256=digest(ordered), feature_names_order=ordered,
                            scaler_identity=sha(state / "top100_scaler_state.json") if (state / "top100_scaler_state.json").exists() else None,
                            completed_k=durable_k, local_completed_k=len(list((state / "k").glob("k_*.json"))),
                            train_val_manifest_identity=digest({r: identity["dataset"]["splits"][r] for r in ("train", "val")}))
                        row = store.commit(stage, payload, metadata=metadata)
                        progress.last_checkpoint = row["record"]["directory"]
                        progress.checkpoint_status = "SAVED"
                        print("CHECKPOINT_STATUS: SAVED " + stage, flush=True)
                        return row

                    def cache_manifest():
                        values = {p.relative_to(local).as_posix(): read(p) for p in (local / "cache").glob("*/cache.json")}
                        manifest = {"algorithm": ALGORITHM, "identity": identity["sha256"],
                                    "dataset_fingerprint": identity["dataset_fingerprint"], "local_arrays": values,
                                    "arrays_uploaded": False, "availability_after_runtime_loss": "REBUILD_FROM_BOUND_TRAIN_VAL"}
                        store.mutable("cache_manifest.json", manifest)
                        target = state / "cache_snapshots" / (attempt + "_" + uuid.uuid4().hex + ".json")
                        atomic(target, manifest, exclusive=True)
                        return {target.relative_to(state).as_posix(): target}

                    try:
                        checkpoint("init", {"run_identity.json": state / "run_identity.json", **cache_manifest()}, runtime_policy=request["runtime"])
                        if store.saved("selection"):
                            validate_selection(state, identity)
                            progress.status, progress.completed = "SELECTION_COMPLETE", 100
                            outcome = "SELECTION_COMPLETE"
                        else:
                            progress.check()
                            train, y = load_selection(request, plan["audit"], local, identity, "train")
                            progress.check()
                            top, mi = module.saved_mi(state, identity, train, y, identity["dataset"]["names"], progress)
                            checkpoint("mi", {n: state / n for n in ("mi_state.json", "mi_top100.json", "mi_feature_scores.csv")}, fit_role="train")
                            progress.check()
                            val, vy = load_selection(request, plan["audit"], local, identity, "val")
                            X, V = scaled_top(request, state, local, identity, train, y, val, vy, top)
                            checkpoint("scaler", {**{n: state / n for n in ("top100_scaler.joblib", "top100_scaler_state.json")},
                                                   **cache_manifest()}, fit_role="train", validation_transform_only=True)
                            binding = {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
                                       "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}
                            if not (state / "rfe_state.json").exists():
                                steps, previous = module.load_chain(state / "rfe_steps", "a1rfe", binding)
                                validate_steps(steps, 100)
                                for number in range(1, len(steps)+1):
                                    name = f"a1rfe_{number:03d}"
                                    checkpoint(name, {f"rfe_steps/{name}.json": state / "rfe_steps" / (name + ".json")})
                                from sklearn.ensemble import RandomForestClassifier
                                def save_step(row):
                                    nonlocal previous
                                    previous = module.save_chain(state / "rfe_steps", "a1rfe", row["step"], binding, row, previous)
                                    name = f"a1rfe_{row['step']:03d}"
                                    checkpoint(name, {f"rfe_steps/{name}.json": state / "rfe_steps" / (name + ".json")})
                                result = fit_step5(X, y, top, RandomForestClassifier(**module.RFE_PARAMS["svm"]),
                                    steps=steps, checkpoint=save_step, check=progress.check, event=progress.event)
                                csv_path = state / "rfe_feature_ranking.csv"
                                if csv_path.exists():
                                    raise Blocked("Partial A1 ranking publication exists; preserve for review")
                                with csv_path.open("x", encoding="utf-8", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=list(result["records"][0]))
                                    writer.writeheader(); writer.writerows(result["records"])
                                atomic(state / "rfe_state.json", {"binding": binding, "result": result,
                                    "files": {csv_path.name: info(csv_path)}}, exclusive=True)
                            top, ranked = validate_ranking(state, identity)
                            checkpoint("rfe", {n: state / n for n in ("rfe_state.json", "rfe_feature_ranking.csv")})
                            progress.check()
                            Xr, Vr = ranked_arrays(local, identity, X, V, top, ranked)
                            del X, V, train, val
                            # A prior upload failure can leave valid newer LOCAL k.
                            # Re-commit that verified prefix before starting any new fit.
                            history = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
                            progress.completed = len(history.rows)
                            if history.rows:
                                best = choose(history.rows, ranked)
                                progress.best_k, progress.best_accuracy = best["k"], best["accuracy"]
                            def commit_k():
                                current = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
                                for row in current.rows:
                                    name = f"k_{row['k']:03d}"
                                    checkpoint(name, {f"k/{name}.json": state / "k" / (name + ".json")},
                                               k=row["k"], completed_k_values=list(range(1, row["k"]+1)))
                            commit_k()
                            def fit(model, matrix, labels):
                                progress.event("svm_fit_started", train_rows=len(labels), features=matrix.shape[1])
                                model, measurement = module.timed_fit(model, matrix, labels)
                                progress.event("svm_fit_completed", fit=measurement)
                                return model, measurement
                            def metrics(labels, prediction):
                                values = module.metrics(labels, prediction)
                                values["label_origin"] = "PROVISIONAL_FOLDER_BOUND_PRODUCER_NOT_GROUND_TRUTH"
                                return values
                            def boundary(**kwargs):
                                commit_k()  # Every completed k, never a 5-hour timer.
                            rows = search(state, identity, ranked, Xr, y, Vr, vy, progress,
                                model_factory=lambda: module.make_model("svm"), fit_fn=fit, metrics_fn=metrics,
                                boundary=boundary, stop_requested=progress.stop_requested,
                                rss=lambda: progress.proc.memory_info().rss)
                            del Xr, Vr, y, vy
                            if rows is None:
                                raise Pause("Stopped after committing the completed k prefix")
                            frozen = freeze(state, identity, rows, ranked, module)
                            create_selection(state, identity, rows, ranked, frozen)
                            progress.completed, progress.status = 100, "SELECTION_COMPLETE"
                            checkpoint("selection", {**{n: state / n for n in
                                (*OUTPUTS, "selection_frozen.json")}, **cache_manifest()}, completed_k_values=list(range(1, 101)))
                            outcome = "SELECTION_COMPLETE"
                    except Pause as exc:
                        progress.status = "PAUSED"
                        progress.event("paused_at_safe_boundary", reason=str(exc))
                        checkpoint("pause_" + attempt, {}, next_stage=progress.stage)
                        outcome = "PAUSED"
                    except BaseException as exc:
                        progress.status = "FAILED"
                        progress.event("attempt_failed", error=str(exc), completed=False)
                        try:
                            checkpoint("failed_" + attempt, {}, completed=False)
                        except Exception as upload_error:
                            print("FAILURE CHECKPOINT NOT CONFIRMED: " + str(upload_error), file=sys.stderr, flush=True)
                        failure = exc
                if progress.error and failure is None:
                    failure = Blocked("Telemetry error: " + progress.error)
                if failure:
                    store.release("FAILED")
                    raise failure
                if outcome == "SELECTION_COMPLETE":
                    result = publish(store)
                    store.release("SELECTION_COMPLETE")
                    return result
                store.release("PAUSED")
                return {"status": "PAUSED", "directory": str(remote), "completed_k": progress.completed, **SCIENTIFIC}
        finally:
            capture.close()


def verify(remote, identity, temporary_root):
    """Reconstruct only small committed artifacts in a fresh local workspace."""
    import tempfile
    if identity.get("algorithm") != ALGORITHM or identity.get("sha256") != digest({k: v for k, v in identity.items() if k != "sha256"}):
        raise Blocked("Not the same A1 identity")
    remote, temporary_root = Path(remote), Path(temporary_root)
    manifest = read(remote / "run_manifest.json")
    if manifest.get("identity") != identity:
        raise Blocked("Verify identity mismatch")
    rows = ledger(remote, identity)
    marker = read(remote / "results/SELECTION_COMPLETE.json")
    artifact = read(remote / "results/artifact_manifest.json")
    selected = next((r for r in rows if r["record"]["stage"] == "selection"), None)
    if (selected is None or marker.get("identity") != identity["sha256"] or marker.get("checkpoint") != selected["sha256"]
            or marker.get("manifest") != info(remote / "results/artifact_manifest.json")
            or marker.get("status") != "SELECTION_COMPLETE" or artifact.get("identity") != identity["sha256"]
            or artifact.get("algorithm") != ALGORITHM or set(artifact.get("files", {})) != set(OUTPUTS)
            or any(marker.get(k) != v for k, v in SCIENTIFIC.items())):
        raise Blocked("Final A1 selection marker/manifest differs")
    check_files(remote / "results", artifact["files"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a1_verify_", dir=temporary_root) as temporary:
        state = Path(temporary)
        for row in rows:
            for name, meta in row["record"]["files"].items():
                target = state / name
                if target.exists():
                    if info(target) != meta:
                        raise Blocked("A committed artifact was overwritten by a later checkpoint")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["path"] / name, target)
        outcome = validate_selection(state, identity)
        for name in OUTPUTS:
            if info(state / name) != artifact["files"][name]:
                raise Blocked("Published result differs from committed selection")
    return {**outcome, "directory": str(remote), "phase_status": "SELECTION_COMPLETE", "test_evaluation_calls": 0}
