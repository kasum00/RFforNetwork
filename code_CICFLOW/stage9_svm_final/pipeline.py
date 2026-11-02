"""Final-only worker. No MI, RF-RFE or k-search entry point is called here."""
import csv
import math
import time
import uuid
from pathlib import Path
from .common import (Blocked, SCIENTIFIC, confirmation, digest, files_checked, info, intact,
                     once, paths, read, sha, utc)
from .data import load
from .metrics import compute
from .preflight import review
from .store import bundle, ensure_identity, lease, publish_bundle
from .telemetry import Heartbeat


def validate_objects(root, identity):
    import joblib
    model = joblib.load(Path(root) / "model.joblib")
    scaler = joblib.load(Path(root) / "final_scaler.joblib")
    total = sum(identity["dataset"]["binding"]["splits"][r]["rows"] for r in ("train", "val"))
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import numpy as np
    if (type(model) is not SVC or model.get_params() != identity["svm_resolved"]
            or not np.array_equal(model.classes_, sorted((identity["labels"]["benign"], identity["labels"]["attack"])))
            or model.n_features_in_ != identity["best_k"] or type(scaler) is not StandardScaler
            or scaler.n_features_in_ != identity["best_k"] or int(scaler.n_samples_seen_) != total
            or scaler.get_params() != StandardScaler().get_params()):
        raise Blocked("Final model/scaler fitted contract differs")
    return model, scaler


def verify_final(root, identity):
    root = Path(root)
    intact(identity)
    if read(root / "identity_manifest.json") != identity:
        raise Blocked("Final identity mismatch")
    marker = read(root / "FINAL_COMPLETE.json")
    checkpoint = bundle(root / "final_model", "final_checkpoint.json", identity)
    state = read(root / "final_model/final_state.json")
    expected_rows = sum(identity["dataset"]["binding"]["splits"][r]["rows"] for r in ("train", "val"))
    if (marker != dict(status="FINAL_VERIFIED", identity=identity["sha256"],
                       checkpoint=info(root / "final_model/final_checkpoint.json"))
            or set(checkpoint["files"]) != {"model.joblib", "final_scaler.joblib", "final_state.json", "selection_binding.json", "resource_usage.json"}
            or read(root / "final_model/selection_binding.json") != dict(features=identity["features"],
                selection_identity=identity["selection_identity"], checkpoint_sha256=identity["selection_checkpoint_sha256"])
            or state["binding"] != {"identity": identity["sha256"], "frozen": checkpoint["frozen_digest"]}
            or state["fit_role"] != "train+val" or state["fit_rows"] != expected_rows
            or checkpoint["scaler_protocol"] != identity["scaler_protocol"]
            or checkpoint["frozen_digest"] != identity["frozen_digest"]
            or type(checkpoint["final_refit_elapsed_seconds"]) not in (int, float)
            or not math.isfinite(checkpoint["final_refit_elapsed_seconds"]) or checkpoint["final_refit_elapsed_seconds"] < 0):
        raise Blocked("Final checkpoint/model/selection protocol mismatch")
    files_checked(root / "final_model", state["files"])
    validate_objects(root / "final_model", identity)
    return dict(status="PASS", phase_status="FINAL_VERIFIED", directory=str(root),
                model_sha256=checkpoint["files"]["model.joblib"]["sha256"],
                final_fit_seconds=state["fit"]["seconds"], final_refit_elapsed_seconds=checkpoint["final_refit_elapsed_seconds"],
                test_values_read=False, **SCIENTIFIC)


def fit_final(request, plan):
    identity = plan["identity"]
    local, remote = paths(request)
    if not request.get("enable_final_fit") or request.get("final_confirmation") != confirmation(request, "FINAL_FIT"):
        raise Blocked("Final fit requires explicit method/config/run confirmation")
    if (remote / "FINAL_COMPLETE.json").exists():
        return verify_final(remote, identity)
    if claim_state(remote) != "UNCLAIMED":
        raise Blocked("Test claim/artifacts exist; final model must never be refitted")
    ensure_identity(remote, identity)
    local.mkdir(parents=True, exist_ok=True)
    with lease(remote, identity, previous_stopped=request.get("previous_worker_stopped")):
        # Restore completed local fit after publication failure, never fit it again.
        candidates = list(local.glob("fit_*/final_checkpoint.json"))
        if len(candidates) > 1:
            raise Blocked("Multiple local final bundles require manual reconciliation")
        if candidates:
            directory = candidates[0].parent
            bundle(directory, "final_checkpoint.json", identity)
            validate_objects(directory, identity)
        else:
            if (remote / "final_model").exists():
                # A fully durable bundle can be completed without local state.
                bundle(remote / "final_model", "final_checkpoint.json", identity)
                validate_objects(remote / "final_model", identity)
                directory = remote / "final_model"
            else:
                directory = local / ("fit_" + uuid.uuid4().hex)
                directory.mkdir()
                with Heartbeat(remote, identity, request.get("heartbeat_seconds", 15)) as progress:
                    progress.event("LOAD_TRAIN_VALIDATION")
                    train, y = load(request, identity, "train")
                    val, vy = load(request, identity, "val")
                    from mi_rfe_svm_colab.common import core
                    module = core(request["source"])
                    # Exact reference final scaler, layout, row order and SVC parameters.
                    fit_clock = time.perf_counter()
                    module.final_refit(directory, identity, plan["selected"]["frozen"], train, y, val, vy,
                                       identity["features"], "svm", progress)
                    elapsed = time.perf_counter()-fit_clock
                    del train, y, val, vy
                    validate_objects(directory, identity)
                    once(directory / "selection_binding.json", dict(features=identity["features"],
                         selection_identity=identity["selection_identity"], checkpoint_sha256=identity["selection_checkpoint_sha256"]))
                    once(directory / "resource_usage.json", progress.snapshot())
                    value = dict(identity=identity["sha256"], completed=utc(),
                        final_refit_elapsed_seconds=elapsed,
                        frozen_digest=digest(plan["selected"]["frozen"]), scaler_protocol=identity["scaler_protocol"],
                        files={n: info(directory / n) for n in
                               ("model.joblib", "final_scaler.joblib", "final_state.json", "selection_binding.json", "resource_usage.json")})
                    value["sha256"] = digest(value)
                    once(directory / "final_checkpoint.json", value)
                    progress.last = "LOCAL_FINAL_FIT_COMPLETE_NOT_YET_DRIVE_COMMITTED"
        if directory != remote / "final_model":
            publish_bundle(directory, remote / "final_model", "final_checkpoint.json", identity)
        once(remote / "FINAL_COMPLETE.json", dict(status="FINAL_VERIFIED", identity=identity["sha256"],
             checkpoint=info(remote / "final_model/final_checkpoint.json")))
        return verify_final(remote, identity)


def claim_state(root):
    root = Path(root)
    if (root / "TEST_COMPLETE.json").exists():
        return "COMPLETE_REQUIRES_VERIFY"
    if (root / "TEST_CLAIM.json").exists():
        return "BLOCKED_AMBIGUOUS_TEST_CLAIM"
    if any((root / n).exists() for n in ("test_metrics.json", "test_ledger.json", "test_checkpoint.json",
                                        "test_predictions.npz", "confusion_matrix.csv")):
        return "BLOCKED_ORPHAN_TEST_ARTIFACTS"
    return "UNCLAIMED"


def validate_claim(claim, root, identity):
    if (claim["identity"] != identity["sha256"] or claim["evaluation_number"] != 1
            or claim["retry_forbidden"] is not True or claim["method"] != identity["method"]
            or claim["config"] != identity["config"] or claim["selection_identity"] != identity["selection_identity"]
            or claim["authorization"] != (f"TEST_ONCE EXPERIMENTAL SVM {identity['method']} {identity['config']}/{identity['run_id']} "
                                          f"SELECTION {identity['selection_run_id']}")
            or claim["final_checkpoint"] != info(Path(root) / "final_model/final_checkpoint.json")
            or claim["model_sha256"] != sha(Path(root) / "final_model/model.joblib")):
        raise Blocked("Test claim binding differs")


def verify_test(root, identity):
    import numpy as np
    root = Path(root)
    verify_final(root, identity)
    claim = read(root / "TEST_CLAIM.json")
    validate_claim(claim, root, identity)
    checkpoint = bundle(root, "test_checkpoint.json", identity)
    marker = read(root / "TEST_COMPLETE.json")
    expected_files = {"test_metrics.json", "test_ledger.json", "confusion_matrix.csv", "test_predictions.npz"}
    if (set(checkpoint["files"]) != expected_files or checkpoint["claim_sha256"] != sha(root / "TEST_CLAIM.json")
            or marker != dict(identity=identity["sha256"], status="TEST_COMPLETE", checkpoint=info(root / "test_checkpoint.json"))):
        raise Blocked("Test completion/claim/files mismatch")
    value = read(root / "test_metrics.json")
    final = read(root / "final_model/final_state.json")
    final_checkpoint = read(root / "final_model/final_checkpoint.json")
    with np.load(root / "test_predictions.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"y", "prediction"}:
            raise Blocked("Prediction artifact schema differs")
        computed = compute(arrays["y"], arrays["prediction"], identity["labels"])
        rows = len(arrays["y"])
        labels, counts = np.unique(arrays["y"], return_counts=True)
        histogram = {str(int(label)): int(n) for label, n in zip(labels, counts)}
    if (value["identity"] != identity["sha256"] or value["metrics"] != computed
            or value["test_rows"] != rows or rows != identity["dataset"]["binding"]["splits"]["test"]["rows"]
            or histogram != identity["dataset"]["label_histograms"]["test"]
            or value["model_sha256"] != claim["model_sha256"] or value["test_evaluation_calls"] != 1
            or value["selection_checkpoint_sha256"] != identity["selection_checkpoint_sha256"]
            or value["features"] != identity["features"] or value["best_k"] != identity["best_k"]
            or any(value[k] != identity[k] for k in ("method", "config", "run_id", "selection_run_id"))
            or value["train_rows"] != identity["dataset"]["binding"]["splits"]["train"]["rows"]
            or value["validation_rows"] != identity["dataset"]["binding"]["splits"]["val"]["rows"]
            or value["final_fit_seconds"] != final["fit"]["seconds"] or value["final_fit_diagnostics"] != final["fit"]
            or value["final_refit_elapsed_seconds"] != final_checkpoint["final_refit_elapsed_seconds"]
            or value["metrics_percent"] != {k: computed[k]*100 for k in
                ("accuracy", "precision_attack", "recall_attack", "f1_attack", "macro_f1", "balanced_accuracy")}
            or type(value["prediction_seconds"]) not in (int, float)
            or not math.isfinite(value["prediction_seconds"]) or value["prediction_seconds"] < 0
            or any(value[k] != v for k, v in SCIENTIFIC.items())):
        raise Blocked("Test metrics/labels/counts/model identity differs")
    ledger = read(root / "test_ledger.json")
    if ledger != dict(identity=identity["sha256"], claim_sha256=sha(root / "TEST_CLAIM.json"),
                      status="COMPLETE", evaluation_number=1, metrics=info(root / "test_metrics.json")):
        raise Blocked("Test ledger completion differs")
    with (root / "confusion_matrix.csv").open(encoding="utf-8", newline="") as f:
        matrix = list(csv.reader(f))
    expected = [["true_label/predicted_label", *map(str, computed["confusion_matrix_labels"])]]
    expected += [[str(label), *map(str, row)] for label, row in zip(computed["confusion_matrix_labels"], computed["confusion_matrix"])]
    if matrix != expected:
        raise Blocked("Confusion matrix CSV differs")
    return dict(status="PASS", phase_status="TEST_COMPLETE", result=value, directory=str(root),
                test_dataset_reopened=False, **SCIENTIFIC)


def test_once(request, plan):
    import numpy as np
    identity = plan["identity"]
    local, remote = paths(request)
    state = claim_state(remote)
    if state == "COMPLETE_REQUIRES_VERIFY":
        return verify_test(remote, identity)  # No model.predict and no Test dataset read.
    if state != "UNCLAIMED":
        raise Blocked(state + ": automatic Test retry forbidden")
    if not request.get("enable_test_evaluation") or request.get("test_confirmation") != confirmation(request, "TEST_ONCE"):
        raise Blocked("Separate explicit Test confirmation is required for this method/config/run")
    verify_final(remote, identity)
    local.mkdir(parents=True, exist_ok=True)
    with lease(remote, identity, previous_stopped=request.get("previous_worker_stopped")):
        if claim_state(remote) != "UNCLAIMED":
            raise Blocked("Test was claimed by another worker")
        model, scaler = validate_objects(remote / "final_model", identity)
        claim = dict(identity=identity["sha256"], method=identity["method"], config=identity["config"],
            selection_identity=identity["selection_identity"], final_checkpoint=info(remote / "final_model/final_checkpoint.json"),
            model_sha256=sha(remote / "final_model/model.joblib"), evaluation_number=1,
            started=utc(), retry_forbidden=True, authorization=request["test_confirmation"])
        once(remote / "TEST_CLAIM.json", claim)  # Readback/fsync BEFORE any Test bytes, even hashing.
        validate_claim(read(remote / "TEST_CLAIM.json"), remote, identity)
        with Heartbeat(remote, identity, request.get("heartbeat_seconds", 15)) as progress:
            progress.event("TEST_CLAIM_DURABLE")
            T, y = load(request, identity, "test", claim=claim)
            progress.event("TEST_PREDICT_ONCE")
            started = time.perf_counter()
            prediction = model.predict(scaler.transform(T))
            seconds = time.perf_counter()-started
            final = read(remote / "final_model/final_state.json")
            final_checkpoint = read(remote / "final_model/final_checkpoint.json")
            counts = {role: identity["dataset"]["binding"]["splits"][role]["rows"] for role in ("train", "val", "test")}
            value = dict(identity=identity["sha256"], method=identity["method"], config=identity["config"],
                run_id=identity["run_id"], selection_run_id=identity["selection_run_id"], best_k=identity["best_k"],
                features=identity["features"], metrics=compute(y, prediction, identity["labels"]),
                train_rows=counts["train"], validation_rows=counts["val"], test_rows=len(y),
                final_fit_seconds=final["fit"]["seconds"], final_fit_diagnostics=final["fit"], prediction_seconds=seconds,
                final_refit_elapsed_seconds=final_checkpoint["final_refit_elapsed_seconds"],
                model_sha256=claim["model_sha256"], selection_checkpoint_sha256=identity["selection_checkpoint_sha256"],
                test_evaluation_calls=1, finished=utc(), **SCIENTIFIC)
            value["metrics_percent"] = {k: value["metrics"][k]*100 for k in
                ("accuracy", "precision_attack", "recall_attack", "f1_attack", "macro_f1", "balanced_accuracy")}
            directory = local / ("test_" + uuid.uuid4().hex)
            directory.mkdir()
            np.savez_compressed(directory / "test_predictions.npz", y=y, prediction=prediction)
            once(directory / "test_metrics.json", value)
            once(directory / "test_ledger.json", dict(identity=identity["sha256"],
                claim_sha256=sha(remote / "TEST_CLAIM.json"), status="COMPLETE", evaluation_number=1,
                metrics=info(directory / "test_metrics.json")))
            with (directory / "confusion_matrix.csv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["true_label/predicted_label", *value["metrics"]["confusion_matrix_labels"]])
                for label, row in zip(value["metrics"]["confusion_matrix_labels"], value["metrics"]["confusion_matrix"]):
                    writer.writerow([label, *row])
            checkpoint = dict(identity=identity["sha256"], claim_sha256=sha(remote / "TEST_CLAIM.json"),
                files={n: info(directory / n) for n in
                       ("test_metrics.json", "test_ledger.json", "confusion_matrix.csv", "test_predictions.npz")})
            checkpoint["sha256"] = digest(checkpoint)
            once(directory / "test_checkpoint.json", checkpoint)
            publish_bundle(directory, remote, "test_checkpoint.json", identity)
            once(remote / "TEST_COMPLETE.json", dict(identity=identity["sha256"], status="TEST_COMPLETE",
                 checkpoint=info(remote / "test_checkpoint.json")))
            progress.last = "TEST_COMPLETE"
        return verify_test(remote, identity)


def execute(request):
    mode = request.get("mode", "preflight")
    if mode not in ("verify-selection", "preflight", "final-fit", "resume-final", "verify-final", "test-readiness", "test", "verify-test"):
        raise Blocked("Unknown Stage9 mode; selection training is not exposed")
    # Every action re-verifies original selection/source/dataset without opening Test.
    _, remote = paths(request)
    plan = review(request, need_fit=mode in ("final-fit", "resume-final") and not (remote / "FINAL_COMPLETE.json").is_file()
                  or mode == "preflight" and request.get("preflight_for") == "final-fit")
    if mode == "test":
        # Same source-bound evidence requirement; no second RAM estimate is needed.
        tests = request.get("unit_tests", {})
        if (tests.get("status") != "PASS" or tests.get("source_sha256") != plan["identity"]["source"]["sha256"]
                or tests.get("environment") != plan["identity"]["environment"] or tests.get("tests_run", 0) <= 0
                or tests.get("skipped", 0) or tests.get("errors", 1) or tests.get("failures", 1)):
            raise Blocked("Current-source synthetic PASS evidence required before Test")
    _, remote = paths(request)
    if mode in ("verify-selection", "preflight"):
        return plan
    if mode in ("final-fit", "resume-final"):
        return fit_final(request, plan)
    if mode == "verify-final":
        return verify_final(remote, plan["identity"])
    if mode == "test-readiness":
        final = verify_final(remote, plan["identity"])
        state = claim_state(remote)
        if state == "COMPLETE_REQUIRES_VERIFY":
            return verify_test(remote, plan["identity"])
        return dict(status="PASS" if state == "UNCLAIMED" else "BLOCKED", test_state=state,
                    final=final, test_values_read=False, confirmation=confirmation(request, "TEST_ONCE"))
    if mode == "test":
        return test_once(request, plan)
    return verify_test(remote, plan["identity"])
