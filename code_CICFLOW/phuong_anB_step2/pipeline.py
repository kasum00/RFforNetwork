"""Independent B selection orchestration; exact A1 numerical helpers are reused."""
import contextlib
import csv
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from .contract import (ALGORITHM, SCIENTIFIC, Blocked, atomic, confirmation, core,
                       digest, info, paths, read, sha, utc, check_files)
from .dataset import load_selection
from .history import History, check_committed_candidates, check_committed_search
from .preflight import review
from .rfe import fit_step2, validate_steps
from .results import (OUTPUTS, create_selection, json_once, publish,
                      validate_ranking, validate_selection)
from .search import search
from .store import Store, ledger
from .telemetry import Capture, Pause, Telemetry
from mi_rfe_svm_colab.runner import scaled_top, ranked_arrays


def run(request):
    mode = request.get("mode", "preflight")
    if mode not in ("metadata", "preflight", "run", "resume", "verify"):
        raise Blocked("B exposes selection only, never final refit or Test")
    plan = review(request, require_tests=mode not in ("metadata", "verify"))
    if mode in ("metadata", "preflight") or plan["status"] != "PASS":
        return plan
    local, remote = paths(request)
    if mode == "verify":
        return verify(remote, plan["identity"], local.parent)
    if request.get("confirmation") != confirmation(request):
        raise Blocked("Exact mode/config/run confirmation required: " + confirmation(request))
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
        json_once(state / "run_identity.json", identity)
        if any((state / "candidates").glob("candidate_*.json")) and not (state / "rfe_state.json").is_file():
            raise Blocked("Candidate records without their bound RFE state; no new fit")
        attempt, started_utc = uuid.uuid4().hex, utc()
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
                        ordered = (read(state / "rfe_state.json")["result"]["names"] if (state / "rfe_state.json").is_file()
                            else read(state / "mi_top100.json") if (state / "mi_top100.json").is_file()
                            else identity["dataset"]["names"])
                        metadata.update(feature_names_order=ordered, feature_order_sha256=digest(ordered),
                            train_val_manifest_identity=digest({r: identity["dataset"]["splits"][r] for r in ("train", "val")}),
                            scaler_identity=sha(state / "top100_scaler_state.json") if (state / "top100_scaler_state.json").is_file() else None)
                        payload = {**files, snapshot.relative_to(state).as_posix(): snapshot,
                                   log_part.relative_to(state).as_posix(): log_part}
                        committed = store.commit(stage, payload, metadata=metadata)
                        progress.last_checkpoint = committed["record"]["directory"]
                        progress.last_checkpoint_utc = committed["record"]["finished"]
                        progress.checkpoint_status = "SAVED"
                        print("CHECKPOINT_STATUS: SAVED " + stage, flush=True)
                        return committed

                    def save_session(status):
                        receipts = list((state / "sessions").glob("session_*.json"))
                        completed = [p for p in receipts if read(p)["outcome"] == "SELECTION_WORK_COMPLETE"]
                        if status == "SELECTION_WORK_COMPLETE" and completed:
                            if len(completed) != 1 or progress.rf_started or progress.svm_started:
                                raise Blocked("New model fits after a frozen work-complete receipt")
                            return  # Resume publication using the original measured selection cutoff.
                        path = state / "sessions" / ("session_" + attempt + ".json")
                        if path.exists():
                            previous = read(path)
                            if (previous["identity"] != identity["sha256"]
                                    or previous["sha256"] != digest({k: v for k, v in previous.items() if k != "sha256"})):
                                raise Blocked("Existing session receipt differs")
                            return  # A publication failure cannot rewrite a work-complete receipt.
                        value = {"identity": identity["sha256"], "attempt": attempt,
                            "outcome": status, "started": started_utc, "finished": utc(),
                            "wall_seconds": progress.clock()-progress.started,
                            "rf_started": progress.rf_started, "rf_completed": progress.rf_completed,
                            "svm_started": progress.svm_started, "svm_completed": progress.svm_completed,
                            "peak_rss_bytes": progress.peak_rss,
                            "prior_receipts": {p.name: info(p) for p in receipts} if status == "SELECTION_WORK_COMPLETE" else {}}
                        value["sha256"] = digest(value)
                        json_once(path, value)
                        checkpoint("session_" + attempt, {path.relative_to(state).as_posix(): path},
                                   outcome=status, timing_scope="selection session before freeze/publication")

                    try:
                        checkpoint("init", {"run_identity.json": state / "run_identity.json"},
                                   search_policy=identity["search_policy"], rfe_step=2)
                        # Local complete timing receipts may survive a failed upload.
                        # Rebind nothing: retain their exact identity and self-hash.
                        for path in sorted((state / "sessions").glob("session_*.json")):
                            receipt = read(path)
                            if (receipt["identity"] != identity["sha256"]
                                    or path.stem != "session_" + receipt["attempt"]
                                    or receipt["sha256"] != digest({k: v for k, v in receipt.items() if k != "sha256"})):
                                raise Blocked("Local session receipt identity/hash differs")
                            checkpoint("session_" + receipt["attempt"], {path.relative_to(state).as_posix(): path},
                                       outcome=receipt["outcome"])
                        if store.saved("selection"):
                            validate_selection(state, identity)
                            progress.status, progress.phase = "SELECTION_COMPLETE", "SELECTION"
                            outcome = "SELECTION_COMPLETE"
                        else:
                            progress.check()
                            train, y = load_selection(request, plan["audit"], local, identity, "train")
                            progress.check()
                            top, _ = module.saved_mi(state, identity, train, y, identity["dataset"]["names"], progress)
                            checkpoint("mi", {n: state / n for n in ("mi_state.json", "mi_top100.json", "mi_feature_scores.csv")},
                                       fit_role="train")
                            progress.check()
                            val, vy = load_selection(request, plan["audit"], local, identity, "val")
                            X, V = scaled_top(request, state, local, identity, train, y, val, vy, top)
                            checkpoint("scaler", {n: state / n for n in ("top100_scaler.joblib", "top100_scaler_state.json")},
                                       fit_role="train", validation_transform_only=True)
                            binding = {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
                                       "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}
                            if not (state / "rfe_state.json").is_file():
                                steps, previous = module.load_chain(state / "rfe_steps", "b2rfe", binding)
                                validate_steps(steps, len(top))
                                progress.set_phase("RFE_COARSE", 51, [r["fit_seconds"] for r in steps])
                                for number in range(1, len(steps)+1):
                                    name = f"b2rfe_{number:03d}"
                                    checkpoint(name, {f"rfe_steps/{name}.json": state / "rfe_steps" / (name + ".json")},
                                               completed_rf_eliminations=number)
                                def save_step(row):
                                    nonlocal previous
                                    previous = module.save_chain(state / "rfe_steps", "b2rfe", row["step"], binding, row, previous)
                                    name = f"b2rfe_{row['step']:03d}"
                                    checkpoint(name, {f"rfe_steps/{name}.json": state / "rfe_steps" / (name + ".json")},
                                               completed_rf_eliminations=row["step"])
                                from sklearn.ensemble import RandomForestClassifier
                                result = fit_step2(X, y, top, RandomForestClassifier(**module.RFE_PARAMS["svm"]),
                                    steps=steps, checkpoint=save_step, check=progress.check, event=progress.event)
                                csv_path = state / "rfe_feature_ranking.csv"
                                expected_rows = [{k: "" if v is None else str(v) for k, v in row.items()} for row in result["records"]]
                                if csv_path.exists():
                                    with csv_path.open(encoding="utf-8", newline="") as handle:
                                        if list(csv.DictReader(handle)) != expected_rows:
                                            raise Blocked("Partial/existing RFE CSV differs; preserve for review")
                                else:
                                    with csv_path.open("x", encoding="utf-8", newline="") as handle:
                                        writer = csv.DictWriter(handle, fieldnames=list(result["records"][0]))
                                        writer.writeheader(); writer.writerows(result["records"])
                                atomic(state / "rfe_state.json", {"binding": binding, "result": result,
                                    "files": {csv_path.name: info(csv_path)}}, exclusive=True)
                            top, ranked = validate_ranking(state, identity)
                            checkpoint("rfe", {n: state / n for n in ("rfe_state.json", "rfe_feature_ranking.csv")},
                                       rf_fits_in_completed_ranking=51)
                            history = History(state, identity, ranked, sha(state / "rfe_state.json"))
                            def commit_candidates(current):
                                durable = check_committed_candidates(store, current)
                                for sequence in range(durable+1, len(current.rows)+1):
                                    path, row = current.path(sequence), current.rows[sequence-1]
                                    checkpoint(current.stage(sequence), {path.relative_to(state).as_posix(): path},
                                               phase=row["phase"], k=row["k"], candidate_sequence=sequence)
                            commit_candidates(history)
                            progress.restore_history(history.rows, ranked)
                            progress.check()
                            Xr, Vr = ranked_arrays(local, identity, X, V, top, ranked)
                            del X, V, train, val
                            def fit(model, matrix, labels):
                                progress.event("svm_fit_started", train_rows=len(labels), features=matrix.shape[1])
                                model, measurement = module.timed_fit(model, matrix, labels)
                                progress.event("svm_fit_completed", fit=measurement)
                                return model, measurement
                            def metrics(labels, prediction):
                                values = module.metrics(labels, prediction)
                                values["label_origin"] = "PROVISIONAL_FOLDER_BOUND_PRODUCER_NOT_GROUND_TRUTH"
                                return values
                            search(state, history, Xr, y, Vr, vy, progress,
                                model_factory=lambda: module.make_model("svm"), fit_fn=fit, metrics_fn=metrics,
                                commit_candidates=commit_candidates, checkpoint=checkpoint,
                                rss=lambda: progress.proc.memory_info().rss)
                            del Xr, Vr, y, vy
                            progress.check()
                            save_session("SELECTION_WORK_COMPLETE")
                            create_selection(state, identity, history, module)
                            progress.status, progress.phase, progress.phase_total = "SELECTION_COMPLETE", "SELECTION", 1
                            checkpoint("selection", {n: state / n for n in OUTPUTS},
                                       evaluated_k=[row["k"] for row in history.rows],
                                       svm_fits=len(history.rows), search_policy=identity["search_policy"])
                            outcome = "SELECTION_COMPLETE"
                    except Pause as exc:
                        progress.status = "PAUSED"
                        progress.event("paused_at_safe_boundary", reason=str(exc))
                        save_session("PAUSED")
                        checkpoint("pause_" + attempt, {}, completed=False)
                        outcome = "PAUSED"
                    except BaseException as exc:
                        progress.status = "FAILED"
                        failure = exc
                        try:
                            progress.event("attempt_failed", error=str(exc), completed=False)
                            save_session("FAILED")
                            checkpoint("failed_" + attempt, {}, completed=False)
                        except BaseException as upload_error:
                            print("FAILURE CHECKPOINT NOT CONFIRMED: " + str(upload_error), file=sys.stderr, flush=True)
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
                return {"status": "PAUSED", "directory": str(remote),
                        "completed_candidates": len(progress.rows), **SCIENTIFIC}
        finally:
            capture.close()


def verify(remote, identity, temporary_root):
    """Rebuild only immutable small B artifacts; never read Parquet/model values here."""
    import tempfile
    if (identity.get("algorithm") != ALGORITHM or identity.get("rfe_step") != 2
            or identity.get("sha256") != digest({k: v for k, v in identity.items() if k != "sha256"})):
        raise Blocked("Not the same intact B step2 identity")
    remote, temporary_root = Path(remote), Path(temporary_root)
    if read(remote / "run_manifest.json")["identity"] != identity:
        raise Blocked("B verify identity mismatch")
    rows = ledger(remote, identity)
    marker, artifact = read(remote / "results/SELECTION_COMPLETE.json"), read(remote / "results/artifact_manifest.json")
    selected = next((r for r in rows if r["record"]["stage"] == "selection"), None)
    if (selected is None or marker["identity"] != identity["sha256"] or marker["checkpoint"] != selected["sha256"]
            or marker["manifest"] != info(remote / "results/artifact_manifest.json") or marker["status"] != "SELECTION_COMPLETE"
            or artifact["identity"] != identity["sha256"] or artifact["algorithm"] != ALGORITHM
            or set(artifact["files"]) != set(OUTPUTS)
            or any(marker.get(k) != v for k, v in SCIENTIFIC.items())):
        raise Blocked("B complete marker/publication binding differs")
    check_files(remote / "results", artifact["files"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="b2_verify_", dir=temporary_root) as temporary:
        state = Path(temporary)
        for row in rows:
            for name, meta in row["record"]["files"].items():
                target = state / name
                if target.exists():
                    if info(target) != meta:
                        raise Blocked("Immutable B artifact overwritten by a later checkpoint")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["path"] / name, target)
        outcome = validate_selection(state, identity)
        _, ranked = validate_ranking(state, identity)
        history = History(state, identity, ranked, sha(state / "rfe_state.json"))
        class Reconstructed:
            pass
        reconstructed = Reconstructed()
        reconstructed.rows, reconstructed.state = rows, state
        check_committed_search(reconstructed, history)
        for name in OUTPUTS:
            if info(state / name) != artifact["files"][name]:
                raise Blocked("Published B result differs from committed selection")
    return {**outcome, "directory": str(remote), "phase_status": "SELECTION_COMPLETE",
            "test_evaluation_calls": 0}
