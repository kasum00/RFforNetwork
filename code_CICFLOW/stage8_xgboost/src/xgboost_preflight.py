"""Admission only; no fit or dataset values."""
import importlib
import importlib.metadata
import json
import os
import shutil
import platform
import math
from pathlib import Path
from .common import Blocked, SCIENTIFIC, clean_git, contract, digest, ensure_owned, output_path, read, snapshot, write
from . import dataset_loader

PROFILE = {"numpy": "2.2.6", "pandas": "2.3.3", "scipy": "1.15.3", "scikit-learn": "1.7.2",
           "joblib": "1.5.3", "psutil": "5.9.5", "xgboost": "3.4.1"}

def json_safe(value):
    import math
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return "NaN"  # audit representation only, never substituted into model params
    return value

def environment(backend="cpu"):
    imports = {"scikit-learn": "sklearn"}
    versions = {}
    for name in (*PROFILE, "pyarrow"):
        importlib.import_module(imports.get(name, name))
        versions[name] = importlib.metadata.version(name)
    mismatches = [name for name, expected in PROFILE.items() if versions[name] != expected]
    if mismatches:
        raise Blocked("Stage 8 reference dependency profile mismatch: " + str({n: versions[n] for n in mismatches}) + "; do not change packages automatically")
    try:
        versions["nvidia-ml-py"] = importlib.metadata.version("nvidia-ml-py")
    except importlib.metadata.PackageNotFoundError:
        versions["nvidia-ml-py"] = None
    if backend == "cuda":
        importlib.import_module("pynvml")
        if versions["nvidia-ml-py"] != "13.580.82":
            raise Blocked("CUDA reference requires nvidia-ml-py==13.580.82")
    return {"versions": versions, "python": platform.python_version(), "machine": platform.machine(), "os": os.name}

def gpu_inventory():
    import pynvml
    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() < 1:
            raise Blocked("CUDA requested but no GPU")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {"name": name.decode() if isinstance(name, bytes) else str(name), "index": 0,
                "free_bytes": int(mem.free), "total_bytes": int(mem.total)}
    finally:
        pynvml.nvmlShutdown()

def model_factory(request, rounds=200):
    import xgboost as xgb
    approved = contract(request["source"])
    if xgb.__version__ != approved["xgboost_version"] or rounds != 200:
        raise Blocked("XGBoost version or total rounds changed")
    if request["backend"] not in ("cpu", "cuda"):
        raise Blocked("Choose cpu or cuda explicitly")
    return xgb.XGBClassifier(**approved["parameters"], device=request["backend"])

def model_inventory(request, feature_count=380):
    import xgboost as xgb
    model = model_factory(request)
    build = xgb.build_info()
    if request["backend"] == "cuda" and str(build.get("USE_CUDA", "")).lower() not in ("true", "1", "on", "yes"):
        raise Blocked("XGBoost build has no CUDA support; CPU fallback forbidden")
    params = {k: v for k, v in model.get_xgb_params().items() if v is not None}
    # Native constructor configuration, NOT a training/benchmark probe.
    booster = xgb.Booster(params={**params, "num_feature": feature_count})
    resolved = json.loads(booster.save_config())
    device = resolved["learner"]["generic_param"]["device"]
    assert_backend(request["backend"], device)
    gradient = resolved["learner"]["gradient_booster"]
    tree = gradient.get("tree_train_param", {})
    defaults = {name: next((tree[key] for key in keys if key in tree), None) for name, keys in {
        "learning_rate": ("learning_rate", "eta"), "subsample": ("subsample",),
        "colsample_bytree": ("colsample_bytree",), "colsample_bylevel": ("colsample_bylevel",),
        "colsample_bynode": ("colsample_bynode",), "reg_alpha": ("reg_alpha", "alpha"),
        "reg_lambda": ("reg_lambda", "lambda"), "scale_pos_weight": ("scale_pos_weight",),
    }.items()}
    # scale_pos_weight belongs to the objective in some builds.
    if defaults["scale_pos_weight"] is None:
        defaults["scale_pos_weight"] = resolved["learner"].get("objective", {}).get("reg_loss_param", {}).get("scale_pos_weight")
    if any(value is None for value in defaults.values()):
        raise Blocked("Cannot resolve all approved native defaults before training; inspect native configuration, do not guess")
    defaults.update(booster=gradient["name"], objective=resolved["learner"]["objective"]["name"],
                    eval_metric="logloss", random_state=42, tree_method="hist", device=device)
    return {"constructor": json_safe(model.get_params()), "booster_parameters": json_safe(model.get_xgb_params()),
            "resolved_native_configuration_before_fit": resolved, "resolved_summary": defaults, "build": json_safe(build)}

def assert_backend(requested, actual):
    if not (actual == "cpu" if requested == "cpu" else str(actual).startswith("cuda")):
        raise Blocked(f"Backend changed: requested {requested}, actual {actual}; no fallback")

def resources(request, audit):
    import psutil
    budget = request["budget"]
    required = {"min_start_available_ram_bytes", "min_runtime_available_ram_bytes", "max_tree_rss_bytes",
                "min_free_disk_bytes", "max_elapsed_seconds", "min_gpu_free_bytes"}
    if set(budget) != required or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in budget.values()):
        raise Blocked("Invalid explicit resource budget")
    directory = output_path(request)
    memory = psutil.virtual_memory()
    free = shutil.disk_usage(directory).free
    rows = audit["binding"]["splits"]
    estimate = max(row["rows"] for row in rows.values()) * 380 * 4 * 6
    reasons = []
    if memory.available < max(estimate, budget["min_start_available_ram_bytes"]):
        reasons.append("Available RAM below budget/data estimate")
    if free < budget["min_free_disk_bytes"]:
        reasons.append("Disk below explicit budget")
    gpu = gpu_inventory() if request["backend"] == "cuda" else None
    if gpu and gpu["free_bytes"] < budget["min_gpu_free_bytes"]:
        reasons.append("GPU VRAM below budget")
    return {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons,
            "available_ram_bytes": memory.available, "required_ram_estimate_bytes": estimate, "disk_free_bytes": free,
            "gpu": gpu, "no_parameter_reduction": True, "not_a_runtime_forecast": True}

def inspect(request, *, require_tests=True, persist_identity=True):
    directory = ensure_owned(request)
    commit = clean_git(request["source"], request["git_commit"])
    source, env = snapshot(request["source"]), environment(request["backend"])
    approved = contract(request["source"])
    audit = dataset_loader.inspect(request)
    settings = model_inventory(request)
    identity = {"format": approved["format"], "experiment": request["experiment"], "config": request["config"],
                "run_name": directory.name, "git_commit": commit, "source": source, "environment": env,
                "backend": request["backend"], "contract": approved, "effective_parameters": settings,
                "dataset": audit, "budget": request["budget"], "checkpoint_interval_rounds": 10,
                "source_root": str(Path(request["source"]).resolve()), **SCIENTIFIC}
    if not persist_identity:
        # Early notebook preview happens before the final MODE/BACKEND selection.
        # It must not reject an old checkpoint merely because RESUME is not set
        # yet, nor mutate/bind that checkpoint to the preview configuration.
        resource = resources(request, audit)
        return {"status": resource["status"], "reasons": resource["reasons"], "resources": resource,
                "identity_preview": identity, "training_authorized": False,
                "final_identity_admission_required": True, "model_fit_calls": 0, "test_values_read": False, **SCIENTIFIC}
    reasons = []
    if (directory / "run_manifest.json").exists():
        prior = read(directory / "run_manifest.json")
        if prior.get("identity") != identity:
            raise Blocked("Source/environment/dataset/parameters/backend/run identity mismatch")
        if not (directory / "configuration.json").is_file():
            raise Blocked("Partial initial manifest/configuration publication; preserve files for inspection")
    if (directory / "RUN_COMPLETE.json").exists():
        from .drive_artifact_manager import verify
        return {"status": "PASS", "already_complete": True, "verification": verify(directory), "identity": identity, **SCIENTIFIC}
    if (directory / "RUN_VALIDATED.json").exists():
        from .drive_artifact_manager import verify
        verify(directory, require_complete=False)
        return {"status": "PASS", "already_validated": True, "identity": identity,
                "publication_only": True, **SCIENTIFIC}
    protocol_path = directory / "checkpoints/booster/protocol_state.json"
    sealed_test = protocol_path.exists() and read(protocol_path).get("test_state") == "TEST_COMPLETED"
    resource = {"status": "PASS", "reasons": [], "not_required": "publication-only recovery"} if sealed_test else resources(request, audit)
    reasons.extend(resource["reasons"])
    evidence = request.get("unit_tests", {})
    if require_tests and (evidence.get("status") != "PASS" or not evidence.get("tests_run")
            or evidence.get("source_sha256") != source["sha256"] or evidence.get("environment") != env):
        reasons.append("Missing/failed/stale unit test evidence")
    if protocol_path.exists():
        state = read(protocol_path)
        if state.get("test_state") in ("TEST_EVALUATION_REQUESTED", "TEST_STATE_UNKNOWN"):
            reasons.append("Test state ambiguous: cannot evaluate again")
        if not request.get("resume") and not (directory / "RUN_VALIDATED.json").exists():
            reasons.append("Fit state exists: set MODE=resume explicitly")
    if persist_identity and not (directory / "run_manifest.json").exists():
        write(directory / "run_manifest.json", {"format": approved["format"], "identity": identity,
                                              "identity_sha256": digest(identity), **SCIENTIFIC})
        write(directory / "configuration.json", {"identity_sha256": digest(identity), "parameters": settings,
                                                "contract": approved, **SCIENTIFIC})
    report = {"status": "BLOCKED" if reasons else "PASS", "reasons": reasons, "identity": identity,
              "resources": resource, "model_fit_calls": 0, "test_values_read": False, **SCIENTIFIC}
    report["training_authorized"] = False  # exact confirmation + fresh full admission still required
    if persist_identity:
        write(directory / "preflight.json", report, exclusive=False)
    return report
