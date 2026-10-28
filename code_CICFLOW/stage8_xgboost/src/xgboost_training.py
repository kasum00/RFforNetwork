"""One baseline config per fresh process; scheduler handles four sequentially."""
import contextlib
import gc
import json
import os
import time
import uuid
from pathlib import Path
from .common import Blocked, EXPECTED, SCIENTIFIC, check_files, confirmation, digest, file_record, hash_file, load_core, output_path, read, snapshot, utc, write
from .checkpoint_manager import CheckpointContext, CheckpointManager, ProtocolStateStore
from . import dataset_loader, evaluation, xgboost_preflight as admission
from .drive_artifact_manager import REQUIRED, verify

class Pulse:
    def __init__(self, request):
        self.request, self.directory = request, output_path(request)
        self.start, self.peak = time.perf_counter(), 0

    def __call__(self, event, **details):
        import psutil
        rss = 0
        for proc in [psutil.Process(), *psutil.Process().children(recursive=True)]:
            try:
                rss += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak = max(self.peak, rss)
        available, budget = psutil.virtual_memory().available, self.request["budget"]
        row = {"utc": utc(), "event": event, "elapsed_seconds_this_attempt": time.perf_counter()-self.start,
               "rss_bytes": rss, "available_ram_bytes": available, "details": details}
        if self.request["backend"] == "cuda":
            row["gpu"] = admission.gpu_inventory()
        log = self.directory / "logs/events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        write(self.directory / "logs/progress.json", row, exclusive=False)
        print(f"[{row['elapsed_seconds_this_attempt']:.1f}s] {event} RSS={rss/1024**3:.2f}GiB {json.dumps(details)}", flush=True)
        if rss > budget["max_tree_rss_bytes"] or available < budget["min_runtime_available_ram_bytes"]:
            raise Blocked("Runtime RAM budget exceeded; no data/parameter reduction")
        if row["elapsed_seconds_this_attempt"] > budget["max_elapsed_seconds"]:
            raise Blocked("Runtime time budget exceeded")
        if row.get("gpu", {}).get("free_bytes", budget["min_gpu_free_bytes"]) < budget["min_gpu_free_bytes"]:
            raise Blocked("Runtime GPU VRAM budget exceeded")

def context_for(request, identity):
    directory = output_path(request)
    names = identity["dataset"]["binding"]["names"]
    return CheckpointContext(checkpoint_root=directory / "checkpoints", run_name=directory.name, run_id="booster",
        experiment="baseline_xgb", config=request["config"], backend=request["backend"],
        xgboost_version=identity["environment"]["versions"]["xgboost"], git_commit=identity["git_commit"],
        source_hashes=identity["source"]["files"], dataset_identity=identity["dataset"],
        hyperparameters={"contract": identity["contract"], "effective": identity["effective_parameters"],
                         "environment": identity["environment"], "budget": identity["budget"]},
        feature_names=names, selected_features=names, checkpoint_interval_rounds=10, total_rounds=200)

def ensure_context(state, context):
    if any(state.get(key) != value for key, value in context.compatibility_identity().items()):
        raise Blocked("Checkpoint/protocol identity mismatch")

def run_stage(request, context, protocol, stage, X, y, pulse):
    manager = CheckpointManager(context)
    root = context.stage_root(stage)
    # A directory with orphan models is deliberately sent to latest(), which blocks.
    has_stage = root.is_dir() and any(root.iterdir())
    if has_stage and not request.get("resume"):
        raise Blocked("Checkpoint exists: explicit MODE=resume required")
    if has_stage:
        manager.latest(stage=stage)  # verify first; never silently fall back
    protocol.mark_stage(stage, "IN_PROGRESS")
    pulse(stage + "_started", resume=has_stage)
    model, fit = manager.fit_stage(model=admission.model_factory(request), X=X, y=y,
                                  stage=stage, resume=has_stage, progress=pulse)
    actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    admission.assert_backend(request["backend"], actual)
    protocol.mark_stage(stage, "STAGE_COMPLETE")
    pulse(stage + "_completed", completed_rounds=fit["completed_rounds"])
    return model, fit

def save_joblib(path, obj):
    import joblib
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise Blocked("Partial final artifact publication requires inspection: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, temporary, compress=3)
    temporary.rename(path)

def final_artifacts(request, context, protocol, identity, train, y, val, vy, pulse):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    directory = output_path(request)
    state_path = directory / "checkpoints/final_refit_artifacts.json"
    if state_path.exists():
        state = read(state_path)
        if state["identity_sha256"] != digest(identity) or state["completed_rounds"] != 200:
            raise Blocked("Final artifact identity mismatch")
        check_files(directory, state["files"])
        CheckpointManager(context).latest(stage="final_refit")
        return state
    model_path, scaler_path = directory / "models/final_model.ubj", directory / "models/final_scaler.joblib"
    if model_path.exists() or scaler_path.exists():
        raise Blocked("UNSUPPORTED: partial model/scaler export without final state; preserve files")
    combined = np.asarray(np.concatenate([train, val]), dtype=np.float32, order="C")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    del combined
    model, fit = run_stage(request, context, protocol, "final_refit", scaled, np.concatenate([y, vy]), pulse)
    del scaled
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name("final_model.partial.ubj")
    if temporary.exists():
        raise Blocked("Partial UBJ export exists")
    model.save_model(temporary)
    temporary.rename(model_path)
    save_joblib(scaler_path, scaler)
    write(directory / "models/booster_config.json", json.loads(model.get_booster().save_config()))
    state = {"identity_sha256": digest(identity), "completed_rounds": 200, "fit": fit,
             "rows": len(y)+len(vy), "fit_role": "Train+Validation", "frozen_features_sha256": hash_file(directory / "feature_selection.json"),
             "files": {name: file_record(directory / name) for name in
                       ("models/final_model.ubj", "models/final_scaler.joblib", "models/booster_config.json")}}
    write(state_path, state)
    return state

def write_report_and_validation(request, identity, context, pulse):
    directory = output_path(request)
    state = read(context.run_root / "protocol_state.json")
    ensure_context(state, context)
    test = read(directory / "metrics/test_metrics.json")
    if state["test_state"] != "TEST_COMPLETED" or state["test_metrics_sha256"] != hash_file(directory / "metrics/test_metrics.json"):
        raise Blocked("No completed, hash-bound Test result")
    final = read(directory / "checkpoints/final_refit_artifacts.json")
    check_files(directory, final["files"])
    config = read(directory / "models/booster_config.json")
    admission.assert_backend(request["backend"], config["learner"]["generic_param"]["device"])
    if dataset_loader.inspect(request)["fingerprint"] != identity["dataset"]["fingerprint"] or snapshot(request["source"]) != identity["source"]:
        raise Blocked("Source/dataset changed during run")
    resource = {"runtime_seconds_this_attempt": time.perf_counter()-pulse.start,
                "peak_rss_bytes_sampled_this_attempt": pulse.peak, "fit": final["fit"], **SCIENTIFIC}
    if not (directory / "reports/resource_usage.json").exists():
        write(directory / "reports/resource_usage.json", resource)
    report = directory / "reports/EXPERIMENT_REPORT.md"
    if not report.exists():
        text = (f"# {directory.name}\n\nBaseline XGBoost, all 380 features. No MI/RFE/RF artifacts reused.\n\n"
                f"PROVISIONAL_RESEARCH; OFFICIAL_BLOCKED; capture/session UNKNOWN; proxy labels.\n\n"
                f"Backend: {request['backend']}; fixed 200 rounds/depth10/min_child_weight3.\n\n"
                f"Test evaluations: 1. Test metrics:\n\n```json\n{json.dumps(test['metrics'], indent=2)}\n```\n\n"
                "Validation reports fixed-model behavior; it does not tune parameters or k.\n"
                "Final scaler/model refit Train+Validation after freezing all 380 columns.\n"
                "Inner checkpoint official_status=COMPLETED is a legacy protocol field, not scientific official approval.\n")
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as handle:
            handle.write(text)
    files = (set(REQUIRED) - {"RUN_VALIDATED.json"}) | {"checkpoints/final_refit_artifacts.json", "reports/resource_usage.json"}
    validation = {"identity_sha256": digest(identity), "completed_rounds": 200,
                  "checkpoint_identity": context.compatibility_identity(),
                  "files": {name: file_record(directory / name) for name in sorted(files)}, **SCIENTIFIC}
    if (directory / "RUN_VALIDATED.json").exists():
        check_files(directory, read(directory / "RUN_VALIDATED.json")["files"])
    else:
        write(directory / "RUN_VALIDATED.json", validation)
    return verify(directory, require_complete=False)

def execute(request):
    if request["mode"] not in ("full", "resume"):
        raise Blocked("Only full/resume may train")
    if request.get("confirmation") != confirmation(request):
        raise Blocked("Missing exact ALL4 confirmation")
    directory = output_path(request)
    with load_core(request["source"]).run_lock(directory):
        plan = admission.inspect(request)
        if plan["status"] != "PASS":
            raise Blocked("Preflight: " + str(plan["reasons"]))
        if plan.get("already_complete"):
            return plan["verification"]
        identity, pulse = plan["identity"], Pulse(request)
        context = context_for(request, identity)
        protocol = ProtocolStateStore(context)
        if protocol.path.exists():
            state = read(protocol.path)
            ensure_context(state, context)
            if state["test_state"] == "TEST_COMPLETED":
                # Recover publication only; never load rows or evaluate Test again.
                return write_report_and_validation(request, identity, context, pulse)
            if not request.get("resume"):
                raise Blocked("Explicit resume required")
            state = protocol.load_for_resume()
        else:
            state = protocol.create_fresh()
        audit = identity["dataset"]
        pulse("train_loading")
        train, y, _ = dataset_loader.load(request, audit, "train")
        pulse("validation_loading")
        val, vy, _ = dataset_loader.load(request, audit, "val")
        from sklearn.preprocessing import StandardScaler
        validation_path = directory / "metrics/validation_metrics.json"
        if not validation_path.exists():
            scaler = StandardScaler()
            X, V = scaler.fit_transform(train), scaler.transform(val)
            model, fit = run_stage(request, context, protocol, "validation_fit", X, y, pulse)
            value = {"identity_sha256": digest(identity), "metrics": evaluation.evaluate(request["source"], vy, model.predict(V)),
                     "rows": len(vy), "fit": fit, "used_for_hyperparameter_tuning": False,
                     "feature_count": 380, "best_k": None}
            write(validation_path, value)
            protocol.update(validation_metrics_sha256=hash_file(validation_path))
            del model, X, V, scaler
            gc.collect()
        elif read(protocol.path).get("validation_metrics_sha256") != hash_file(validation_path):
            raise Blocked("Validation artifact publication is incomplete or corrupted")
        selection = {"identity_sha256": digest(identity), "status": "NOT_APPLICABLE", "method": "baseline_all_380",
                     "selected_features": audit["binding"]["names"], "best_k": None, "mi_fit": False, "rfe_fit": False,
                     "rf_artifacts_reused": False, "test_used_for_selection": False, "frozen_before_final_refit": True}
        selection_path = directory / "feature_selection.json"
        if selection_path.exists():
            if read(selection_path) != selection:
                raise Blocked("Frozen baseline feature mapping changed")
        else:
            write(selection_path, selection)
        pulse("selection_frozen", features=380)
        final_artifacts(request, context, protocol, identity, train, y, val, vy, pulse)
        del train, val, y, vy
        gc.collect()
        protocol.request_test_once()  # persisted BEFORE opening Test values
        pulse("test_evaluation_started")
        T, ty, _ = dataset_loader.load(request, audit, "test", protocol=protocol)
        import joblib
        model = admission.model_factory(request)
        model.load_model(directory / "models/final_model.ubj")
        model.set_params(device=request["backend"])  # restore explicitly approved inference device after UBJ load
        admission.assert_backend(request["backend"], json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"])
        scaler = joblib.load(directory / "models/final_scaler.joblib")
        test_metrics = evaluation.evaluate(request["source"], ty, model.predict(scaler.transform(T)))
        write(directory / "metrics/test_metrics.json", {"identity_sha256": digest(identity), "rows": len(ty),
              "test_evaluation_calls": 1, "metrics": test_metrics, **SCIENTIFIC})
        cm = test_metrics["confusion_matrix_labels_0_benign_1_attack"]
        cm_path = directory / "metrics/confusion_matrix.csv"
        with cm_path.open("x", encoding="utf-8") as stream:
            stream.write("true\\pred,Benign,Attack\n" + "\n".join(name + "," + ",".join(map(str, row)) for name, row in zip(("Benign", "Attack"), cm)) + "\n")
        protocol.complete_test_once(metrics_sha256=hash_file(directory / "metrics/test_metrics.json"))
        pulse("test_evaluation_completed", accuracy=test_metrics["accuracy"])
        return write_report_and_validation(request, identity, context, pulse)
