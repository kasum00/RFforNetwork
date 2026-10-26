"""Stage 8 XGBoost extension for the validated CICIoMT2024 C+ windows.

This module reuses the Stage 5 loader, feature contract, MI implementation and
RF-RFE implementation.  It does not modify the RF/SVM paths.  Full execution is
fail-closed: one experiment/config/run, explicit budget, exact confirmation,
fresh output, no resume claim, durable Test claim, and exactly one Test
evaluation after final Train+Validation refit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler

try:
    import ciciomt24_training as core
except ModuleNotFoundError:  # Package import used by unit tests.
    from huyen_code import ciciomt24_training as core

try:
    import ciciomt24_xgboost_checkpoint as xgb_checkpoint
except ModuleNotFoundError:  # Package import used by unit tests.
    from huyen_code import ciciomt24_xgboost_checkpoint as xgb_checkpoint


STAGE8_SCHEMA = "stage8.xgboost.official.v1"
SUPPORTED_XGBOOST_VERSION = "3.4.1"
EXPERIMENTS = ("baseline_xgb", "mi_rfe_xgb")
CONFIGS = core.CONFIGS
XGB_REQUIRED_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_child_weight": 3,
}
XGB_FIXED_POLICY = {
    **XGB_REQUIRED_PARAMS,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": core.SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}
RFE_POLICY_SCHEMA = "stage8.xgboost.rf-rfe-policy.v1"
RFE_POLICY_ID = "reuse_mi_rfe_rf_estimator_for_xgboost_v1"
CAPTURE_SESSION_INDEPENDENCE = "NOT_ESTABLISHED"


def strict_json_value(value: object, *, path: str = "$") -> object:
    """Return JSON-native data and reject every non-finite numeric value.

    XGBoost's default missing-value sentinel is normalized separately by
    effective_xgb_parameters().  This generic validator must never silently
    rewrite NaN/Inf to zero or null.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(
                f"Non-finite numeric value is forbidden in strict JSON at "
                f"{path}: {number!r}"
            )
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(
                    f"Unsupported JSON object key at {path}: "
                    f"{type(key).__name__}"
                )
            if isinstance(key, float) and not np.isfinite(key):
                raise ValueError(
                    f"Non-finite JSON object key is forbidden at {path}: "
                    f"{key!r}"
                )
            child = f"{path}.{key}"
            result[str(key)] = strict_json_value(item, path=child)
        return result
    if isinstance(value, (list, tuple)):
        return [
            strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported strict JSON value at {path}: {type(value).__name__}"
    )


def strict_json_text(value: object) -> str:
    normalized = strict_json_value(value)
    return json.dumps(
        normalized,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    # Serialize and validate the complete payload before opening the target.
    # A NaN/Inf/unsupported value therefore cannot leave a truncated JSON file.
    payload = strict_json_text(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def load_xgboost():
    return importlib.import_module("xgboost")


def cuda_device_inventory() -> dict:
    """Verify a real CUDA device and expose VRAM without loading dataset rows."""

    try:
        pynvml = importlib.import_module("pynvml")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CUDA preflight requires pynvml (package nvidia-ml-py); "
            "no CPU fallback is permitted"
        ) from exc
    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        if count < 1:
            raise RuntimeError("CUDA backend requested but NVML reports no GPU")
        devices = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "vram_total_bytes": int(memory.total),
                    "vram_free_bytes": int(memory.free),
                    "vram_used_bytes": int(memory.used),
                }
            )
        if not any(item["vram_free_bytes"] > 0 for item in devices):
            raise RuntimeError("CUDA devices were found but no free VRAM was reported")
        return {"available": True, "device_count": count, "devices": devices}
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"CUDA/NVML device verification failed: {exc}") from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def xgboost_runtime(backend: str) -> dict:
    if backend not in ("cpu", "cuda"):
        raise ValueError("XGBoost backend must be cpu or cuda")
    xgb = load_xgboost()
    version = str(xgb.__version__)
    if version != SUPPORTED_XGBOOST_VERSION:
        raise RuntimeError(
            f"Stage 8 requires xgboost=={SUPPORTED_XGBOOST_VERSION}; found {version}"
        )
    build = dict(xgb.build_info()) if hasattr(xgb, "build_info") else {}
    cuda_value = build.get("USE_CUDA", False)
    cuda_built = cuda_value is True or str(cuda_value).strip().upper() in {
        "1",
        "ON",
        "TRUE",
        "YES",
    }
    if backend == "cuda" and not cuda_built:
        raise RuntimeError("CUDA backend requested but this XGBoost build has no CUDA support")
    gpu = cuda_device_inventory() if backend == "cuda" else {
        "available": False,
        "device_count": 0,
        "devices": [],
    }
    return {
        "version": version,
        "backend": backend,
        "tree_method": "hist",
        "device": "cuda" if backend == "cuda" else "cpu",
        "build_info": {str(k): str(v) for k, v in build.items()},
        "gpu_inventory": gpu,
        "cpu_is_reference_backend": True,
        "gpu_may_not_be_bitwise_equivalent_to_cpu": backend == "cuda",
        "automatic_backend_fallback": False,
    }


def make_xgb(backend: str = "cpu"):
    runtime = xgboost_runtime(backend)
    xgb = load_xgboost()
    return xgb.XGBClassifier(
        **XGB_FIXED_POLICY,
        device=runtime["device"],
    )


def effective_xgb_parameters(model) -> dict:
    sklearn_params = model.get_params(deep=False)
    booster_params = model.get_xgb_params()

    def audit_parameters(parameters: dict, section: str) -> dict:
        audited = {}
        for name, value in parameters.items():
            parameter_path = (
                f"$.xgboost_effective_parameters.{section}.{name}"
            )
            if (
                name == "missing"
                and isinstance(value, (float, np.floating))
                and np.isnan(float(value))
            ):
                # This is only a human-readable representation in audit JSON.
                # The estimator retains its real np.nan missing-value sentinel.
                audited[name] = "NaN"
            else:
                audited[name] = strict_json_value(
                    value,
                    path=parameter_path,
                )
        return audited

    audited_sklearn = audit_parameters(sklearn_params, "sklearn")
    audited_booster = audit_parameters(booster_params, "booster")
    return {
        "required": dict(XGB_REQUIRED_PARAMS),
        "sklearn": audited_sklearn,
        "booster": audited_booster,
        "class_balancing": {
            "class_weight": None,
            "scale_pos_weight": audited_booster.get("scale_pos_weight", 1.0),
            "oversampling": False,
            "undersampling": False,
        },
    }


def validate_model_contract(model) -> None:
    params = model.get_params(deep=False)
    for name, expected in XGB_REQUIRED_PARAMS.items():
        if params.get(name) != expected:
            raise ValueError(f"XGBoost {name} mismatch: {params.get(name)!r} != {expected!r}")
    if params.get("objective") != "binary:logistic":
        raise ValueError("XGBoost objective must be binary:logistic")
    if params.get("tree_method") != "hist":
        raise ValueError("XGBoost tree_method must be hist")
    if params.get("scale_pos_weight") not in (None, 1, 1.0):
        raise ValueError("Automatic class balancing is forbidden")


def metrics(y_true, y_pred) -> dict:
    result = core.metrics(y_true, y_pred)
    report = result["classification_report"]
    result.update(
        precision_benign=float(report["Benign_proxy"]["precision"]),
        recall_benign=float(report["Benign_proxy"]["recall"]),
        f1_benign=float(report["Benign_proxy"]["f1-score"]),
        precision_attack_per_class=float(report["Attack_proxy"]["precision"]),
        recall_attack_per_class=float(report["Attack_proxy"]["recall"]),
        f1_attack_per_class=float(report["Attack_proxy"]["f1-score"]),
    )
    return result


def load_budget(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "max_elapsed_seconds",
        "max_tree_rss_bytes",
        "min_start_available_ram_bytes",
        "min_runtime_available_ram_bytes",
        "min_free_disk_bytes",
        "poll_interval_seconds",
    }
    if set(value) != required:
        raise ValueError(f"Budget must contain exactly: {sorted(required)}")
    for key in required:
        if (
            not isinstance(value[key], (int, float))
            or isinstance(value[key], bool)
            or not np.isfinite(float(value[key]))
            or value[key] <= 0
        ):
            raise ValueError(f"Budget {key} must be a positive number")
    if not 1 <= value["poll_interval_seconds"] <= 60:
        raise ValueError("poll_interval_seconds must be between 1 and 60")
    return value


def load_rfe_policy(path: Path | None, experiment: str) -> dict:
    if experiment == "baseline_xgb":
        return {
            "status": "NOT_APPLICABLE",
            "approved": True,
            "policy_id": None,
            "rf_rfe_parameters": None,
        }
    if path is None or not path.is_file():
        return {
            "status": "BLOCKED_MISSING_POLICY_APPROVAL",
            "approved": False,
            "policy_id": RFE_POLICY_ID,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    approved = (
        value.get("schema") == RFE_POLICY_SCHEMA
        and value.get("policy_id") == RFE_POLICY_ID
        and value.get("approved") is True
        and value.get("status") == "APPROVED"
        and value.get("rf_rfe_parameters") == core.RF_RFE_PARAMS
    )
    return {
        **value,
        "approved": approved,
        "status": "APPROVED" if approved else "BLOCKED_POLICY_MISMATCH",
        "source_rf_rfe_parameters": dict(core.RF_RFE_PARAMS),
    }


def git_identity(repo: Path) -> dict:
    def call(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = call("rev-parse", "HEAD")
        status = call("status", "--porcelain")
        return {"commit": commit, "clean": not bool(status), "status_porcelain": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "clean": False, "error": str(exc)}


def dataset_identity(windows: Path, store: core.WindowStore) -> dict:
    summary = windows / "window_run_summary.json"
    split_root = windows.parent / "CSV_SPLIT_CPLUS"
    validation = split_root / "validation.json"
    snapshot = store.snapshot()
    return {
        "window_summary_sha256": sha256(summary),
        "window_snapshot_size_mtime_sha256": object_sha256(snapshot),
        "window_snapshot_files": len(snapshot),
        "split_validation_sha256": sha256(validation) if validation.is_file() else None,
        "split_validation_status": "PRESENT" if validation.is_file() else "MISSING",
        "fingerprint_scope": "window summary + split validation when present + sorted file size/mtime",
    }


def source_identity(repo: Path, original: Path) -> dict:
    implementation = Path(__file__).resolve()
    core_source = Path(core.__file__).resolve()
    expected_implementation = repo / "huyen_code" / "ciciomt24_xgboost.py"
    expected_core = repo / "huyen_code" / "ciciomt24_training.py"
    checkpoint_source = Path(xgb_checkpoint.__file__).resolve()
    expected_checkpoint = repo / "huyen_code" / "ciciomt24_xgboost_checkpoint.py"
    if implementation != expected_implementation.resolve():
        raise ValueError("Imported Stage 8 module is not from the selected repository")
    if core_source != expected_core.resolve():
        raise ValueError("Imported training core is not from the selected repository")
    if checkpoint_source != expected_checkpoint.resolve():
        raise ValueError("Imported checkpoint module is not from the selected repository")
    return {
        "stage8_sha256": sha256(implementation),
        "training_core_sha256": sha256(core_source),
        "checkpoint_module_sha256": sha256(checkpoint_source),
        "original_ctu13_sha256": core.source_hashes(original),
        "source_audit": core.source_audit(original),
        "git": git_identity(repo),
    }


def expected_counts(store: core.WindowStore, config: str) -> dict:
    return {
        role: sum(item.count for item in store.files[config] if item.role == role)
        for role in ("train", "val", "test")
    }


def build_checkpoint_context(
    *,
    checkpoint_root: Path,
    output: Path,
    experiment: str,
    config: str,
    backend: str,
    source: dict,
    dataset: dict,
    selected_features: list[str],
    checkpoint_interval_rounds: int,
) -> xgb_checkpoint.CheckpointContext:
    return xgb_checkpoint.CheckpointContext(
        checkpoint_root=checkpoint_root,
        run_name=output.name,
        run_id=output.name,
        experiment=experiment,
        config=config,
        backend=backend,
        xgboost_version=SUPPORTED_XGBOOST_VERSION,
        git_commit=source["git"]["commit"],
        source_hashes={
            "stage8_sha256": source["stage8_sha256"],
            "training_core_sha256": source["training_core_sha256"],
            "checkpoint_module_sha256": source.get(
                "checkpoint_module_sha256", source["stage8_sha256"]
            ),
            "original_ctu13_sha256": source["original_ctu13_sha256"],
        },
        dataset_identity=dataset,
        hyperparameters={
            **XGB_FIXED_POLICY,
            "device": "cuda" if backend == "cuda" else "cpu",
            "automatic_backend_fallback": False,
        },
        feature_names=list(core.FEATURE_NAMES),
        selected_features=selected_features,
        checkpoint_interval_rounds=checkpoint_interval_rounds,
        total_rounds=XGB_REQUIRED_PARAMS["n_estimators"],
    )


def preflight(
    repo: Path,
    windows: Path,
    original: Path,
    output: Path,
    cache_root: Path,
    config: str,
    experiment: str,
    backend: str,
    budget_path: Path,
    unit_test_evidence_path: Path,
    rfe_policy_path: Path | None,
    *,
    checkpoint_root: Path | None = None,
    checkpoint_interval_rounds: int = 10,
    resume: bool = False,
) -> dict:
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    if config not in CONFIGS:
        raise ValueError(f"Unknown config: {config}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    resolved_checkpoint_root = (
        checkpoint_root if checkpoint_root is not None else output.parent / "checkpoints"
    ).resolve()
    checks["baseline_xgb_checkpoint_scope"] = experiment == "baseline_xgb"
    if not checks["baseline_xgb_checkpoint_scope"]:
        errors["baseline_xgb_checkpoint_scope"] = (
            "Official checkpoint/resume is not approved for mi_rfe_xgb"
        )
    checks["checkpoint_interval"] = (
        isinstance(checkpoint_interval_rounds, int)
        and not isinstance(checkpoint_interval_rounds, bool)
        and checkpoint_interval_rounds > 0
        and XGB_REQUIRED_PARAMS["n_estimators"] % checkpoint_interval_rounds == 0
    )
    if not checks["checkpoint_interval"]:
        errors["checkpoint_interval"] = (
            "checkpoint interval must be a positive divisor of 200 rounds"
        )

    try:
        runtime = xgboost_runtime(backend)
        model = make_xgb(backend)
        validate_model_contract(model)
        effective = effective_xgb_parameters(model)
        checks["xgboost_dependency_and_parameters"] = True
    except Exception as exc:
        runtime, effective = None, None
        checks["xgboost_dependency_and_parameters"] = False
        errors["xgboost_dependency_and_parameters"] = str(exc)

    staged = output.with_name(output.name + ".incomplete")
    supervisor_record = output.parent / f"{output.name}.supervisor.json"
    if resume:
        checks["output_is_fresh"] = not output.exists() and staged.is_dir()
        # The local cache may disappear between Colab sessions.  It is derived
        # from the immutable, identity-checked window dataset and may be rebuilt.
        checks["cache_is_fresh"] = (
            not cache_root.exists() or cache_root.is_dir()
        )
    else:
        checks["output_is_fresh"] = (
            not output.exists()
            and not staged.exists()
            and not supervisor_record.exists()
        )
        checks["cache_is_fresh"] = not cache_root.exists()
    checks["paths_are_distinct"] = not (
        output.resolve() == cache_root.resolve()
        or output.resolve() in cache_root.resolve().parents
        or cache_root.resolve() in output.resolve().parents
    )
    checkpoint_run_root = resolved_checkpoint_root / output.name
    checks["checkpoint_paths_are_distinct"] = not (
        checkpoint_run_root == output.resolve()
        or checkpoint_run_root == cache_root.resolve()
        or checkpoint_run_root in output.resolve().parents
        or output.resolve() in checkpoint_run_root.parents
        or checkpoint_run_root in cache_root.resolve().parents
        or cache_root.resolve() in checkpoint_run_root.parents
    )

    try:
        store = core.WindowStore(windows)
        counts = expected_counts(store, config)
        identity = dataset_identity(windows, store)
        checks["dataset_manifest_schema"] = True
    except Exception as exc:
        store, counts, identity = None, None, None
        checks["dataset_manifest_schema"] = False
        errors["dataset_manifest_schema"] = str(exc)

    try:
        sources = source_identity(repo, original)
        checks["approved_original_scripts"] = True
        checks["clean_git_checkout"] = bool(sources["git"]["clean"])
    except Exception as exc:
        sources = None
        checks["approved_original_scripts"] = False
        checks["clean_git_checkout"] = False
        errors["approved_original_scripts"] = str(exc)

    try:
        evidence = json.loads(unit_test_evidence_path.read_text(encoding="utf-8"))
        evidence_ok = (
            evidence.get("status") == "PASS"
            and evidence.get("returncode") == 0
            and sources is not None
            and evidence.get("stage8_sha256") == sources["stage8_sha256"]
            and evidence.get("training_core_sha256") == sources["training_core_sha256"]
        )
        checks["unit_tests_current_source"] = bool(evidence_ok)
    except Exception as exc:
        evidence = None
        checks["unit_tests_current_source"] = False
        errors["unit_tests_current_source"] = str(exc)

    policy = load_rfe_policy(rfe_policy_path, experiment)
    checks["xgboost_rfe_policy"] = bool(policy["approved"])

    try:
        budget = load_budget(budget_path)
        available = psutil.virtual_memory().available
        disk_anchor = cache_root.parent
        while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
            disk_anchor = disk_anchor.parent
        disk_free = psutil.disk_usage(str(disk_anchor)).free
        checks["ram_admission"] = (
            available >= budget["min_start_available_ram_bytes"]
            and budget["max_tree_rss_bytes"]
            <= available - budget["min_runtime_available_ram_bytes"]
        )
        checks["disk_admission"] = disk_free >= budget["min_free_disk_bytes"]
    except Exception as exc:
        budget, available, disk_free = None, None, None
        checks["ram_admission"] = False
        checks["disk_admission"] = False
        errors["resource_budget"] = str(exc)

    checkpoint_state = None
    resume_from_stage = None
    resume_from_round = 0
    try:
        if sources is None or identity is None or runtime is None:
            raise RuntimeError("source, dataset and XGBoost identity must pass first")
        context = build_checkpoint_context(
            checkpoint_root=resolved_checkpoint_root,
            output=output,
            experiment=experiment,
            config=config,
            backend=backend,
            source=sources,
            dataset=identity,
            selected_features=list(core.FEATURE_NAMES),
            checkpoint_interval_rounds=checkpoint_interval_rounds,
        )
        state_store = xgb_checkpoint.ProtocolStateStore(context)
        if resume:
            checkpoint_state = state_store.load_for_resume()
            manager = xgb_checkpoint.CheckpointManager(context)
            validation_status = checkpoint_state["validation_fit_status"]
            final_status = checkpoint_state["final_refit_status"]
            if validation_status in {"IN_PROGRESS", "STAGE_COMPLETE"}:
                verified = manager.latest(stage="validation_fit")
                resume_from_stage = "validation_fit"
                resume_from_round = verified.completed_rounds
            if final_status in {"IN_PROGRESS", "STAGE_COMPLETE"}:
                verified = manager.latest(stage="final_refit")
                resume_from_stage = "final_refit"
                resume_from_round = verified.completed_rounds
            if final_status != "NOT_STARTED" and validation_status != "STAGE_COMPLETE":
                raise xgb_checkpoint.CheckpointCompatibilityError(
                    "final_refit cannot exist before validation_fit is complete"
                )
        elif context.run_root.exists():
            raise FileExistsError(
                "checkpoint run already exists; use explicit RESUME=True or a new run name"
            )
        checks["checkpoint_resume_identity"] = True
    except Exception as exc:
        checks["checkpoint_resume_identity"] = False
        errors["checkpoint_resume_identity"] = str(exc)

    return {
        "schema": STAGE8_SCHEMA,
        "status": "PASS" if checks and all(checks.values()) else "BLOCKED",
        "checks": checks,
        "errors": errors,
        "experiment": experiment,
        "config": config,
        "backend": backend,
        "rows_by_split": counts,
        "xgboost_runtime": runtime,
        "xgboost_effective_parameters": effective,
        "rf_rfe_policy": policy,
        "resource_budget": budget,
        "ram_available_bytes": available,
        "disk_free_bytes": disk_free,
        "source_identity": sources,
        "dataset_identity": identity,
        "unit_test_evidence": evidence,
        "test_feature_rows_read": 0,
        "model_fit_calls": 0,
        "resume_supported": True,
        "resume_requested": bool(resume),
        "resume_accepted": bool(resume and checks["checkpoint_resume_identity"]),
        "checkpoint_root": str(resolved_checkpoint_root),
        "checkpoint_interval_rounds": checkpoint_interval_rounds,
        "checkpoint_verified": bool(checks["checkpoint_resume_identity"]),
        "checkpoint_state": checkpoint_state,
        "resume_from_stage": resume_from_stage,
        "resume_from_round": resume_from_round,
        "total_rounds": XGB_REQUIRED_PARAMS["n_estimators"],
        "automatic_backend_fallback": False,
        "capture_session_independence": CAPTURE_SESSION_INDEPENDENCE,
        "label_origin": "SCENARIO_PROXY",
    }


def search_k_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    top_names: list[str],
    ranked_names: list[str],
    backend: str,
    *,
    progress=None,
    checkpoint_dir: Path | None = None,
) -> dict:
    results: list[dict] = []
    best_k, best_accuracy = None, -1.0
    for k in core.K_GRID:
        core.emit(progress, "k_fit_started", k=k, k_total=len(core.K_GRID))
        indices = [top_names.index(name) for name in ranked_names[:k]]
        model = make_xgb(backend)
        validate_model_contract(model)
        model, fit = core.timed_fit(model, X_train[:, indices], y_train)
        started = time.perf_counter()
        prediction = model.predict(X_val[:, indices])
        row = {
            "k": k,
            **metrics(y_val, prediction),
            "fit": fit,
            "predict_seconds": time.perf_counter() - started,
        }
        results.append(row)
        if checkpoint_dir is not None:
            write_json(
                checkpoint_dir / "k_search_results.partial.json",
                {
                    "status": "INCOMPLETE_EVIDENCE_ONLY",
                    "resume_supported": False,
                    "completed_k": k,
                    "results": results,
                },
            )
        if row["accuracy"] > best_accuracy:
            best_k, best_accuracy = k, row["accuracy"]
        core.emit(
            progress,
            "k_completed",
            k=k,
            k_total=len(core.K_GRID),
            best_k=best_k,
            best_validation_accuracy=best_accuracy,
            result=row,
        )
    return {
        "best_k": best_k,
        "best_validation_accuracy": best_accuracy,
        "features": ranked_names[:best_k],
        "results": results,
        "tie_rule": "strict_greater_keeps_smallest_k",
    }


def select_mi_rfe_xgb(
    train: core.PreparedSplit,
    val: core.PreparedSplit,
    backend: str,
    checkpoint_dir: Path,
    *,
    progress=None,
) -> dict:
    core.require_selection_roles(train, val)
    core.emit(progress, "mi_started", train_rows=len(train.y), features=380)
    mi = core.select_mi(train)
    mi["scores"].to_csv(checkpoint_dir / "mi_feature_scores.csv")
    write_json(checkpoint_dir / "mi_top100.json", mi["names"])
    joblib.dump(mi["scaler"], checkpoint_dir / "mi_scaler.joblib")
    core.emit(progress, "mi_completed", **mi["measurement"], top_features=100)

    X_train, X_val, top100_scaler = core.scale_top100(train, val, mi)
    joblib.dump(top100_scaler, checkpoint_dir / "top100_scaler.joblib")
    core.emit(
        progress,
        "rfe_started",
        target=1,
        step=1,
        expected_forest_fits=100,
        branch="xgboost_extension_reuses_rf_branch_policy",
    )
    ranking = core.rank_rfe(X_train, train.y, mi["names"], "rf", target=1)
    ranking["ranks"].to_csv(checkpoint_dir / "rfe_feature_ranking.csv")
    core.emit(
        progress,
        "rfe_completed",
        **ranking["measurement"],
        forest_fits=ranking["forest_fits"],
    )
    selection = search_k_xgb(
        X_train,
        train.y,
        X_val,
        val.y,
        mi["names"],
        ranking["names"],
        backend,
        progress=progress,
        checkpoint_dir=checkpoint_dir,
    )
    write_json(checkpoint_dir / "k_search_results.json", selection["results"])
    write_json(
        checkpoint_dir / "k_search_results.partial.json",
        {
            "status": "COMPLETE_EVIDENCE_ONLY",
            "resume_supported": False,
            "completed_k": 100,
            "results_sha256": sha256(checkpoint_dir / "k_search_results.json"),
        },
    )
    return {
        "mi": mi,
        "ranking": ranking,
        **selection,
    }


def validation_baseline_xgb(
    train: core.PreparedSplit,
    val: core.PreparedSplit,
    backend: str,
    checkpoint_manager: xgb_checkpoint.CheckpointManager,
    protocol_state: xgb_checkpoint.ProtocolStateStore,
    *,
    resume_stage: bool,
    progress=None,
) -> dict:
    core.require_selection_roles(train, val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.X)
    X_val = scaler.transform(val.X)
    core.emit(progress, "baseline_validation_fit_started", train_rows=len(train.y))
    model = make_xgb(backend)
    protocol_state.mark_stage("validation_fit", "IN_PROGRESS")
    model, fit = checkpoint_manager.fit_stage(
        model=model,
        X=X_train,
        y=train.y,
        stage="validation_fit",
        resume=resume_stage,
        progress=progress,
    )
    protocol_state.mark_stage("validation_fit", "STAGE_COMPLETE")
    started = time.perf_counter()
    result = {
        "metrics": metrics(val.y, model.predict(X_val)),
        "fit": fit,
        "predict_seconds": time.perf_counter() - started,
        "used_for_hyperparameter_tuning": False,
        "features": 380,
    }
    core.emit(progress, "baseline_validation_completed", **result["metrics"])
    return result


def fit_final_xgb(
    train: core.PreparedSplit,
    val: core.PreparedSplit,
    selected_names: list[str],
    backend: str,
    checkpoint_manager: xgb_checkpoint.CheckpointManager | None = None,
    protocol_state: xgb_checkpoint.ProtocolStateStore | None = None,
    *,
    resume_stage: bool = False,
    progress=None,
) -> dict:
    core.require_selection_roles(train, val)
    if (
        not selected_names
        or len(selected_names) != len(set(selected_names))
        or not set(selected_names) <= set(core.FEATURE_NAMES)
    ):
        raise ValueError("Invalid frozen XGBoost feature set")
    indices = [core.FEATURE_NAMES.index(name) for name in selected_names]
    rows = len(train.y) + len(val.y)
    combined = np.empty((rows, len(indices)), dtype=np.float32)
    offset = 0
    for split in (train, val):
        for start in range(0, len(split.y), 8192):
            end = min(start + 8192, len(split.y))
            width = end - start
            np.take(
                split.X[start:end],
                indices,
                axis=1,
                out=combined[offset : offset + width],
            )
            offset += width
    labels = np.concatenate([train.y, val.y])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    del combined
    model = make_xgb(backend)
    validate_model_contract(model)
    if (checkpoint_manager is None) != (protocol_state is None):
        raise ValueError(
            "checkpoint_manager and protocol_state must be supplied together"
        )
    if checkpoint_manager is None:
        # Compatibility path for the bounded scientific unit test/API.  The
        # official worker always supplies both durable checkpoint objects and
        # therefore never enters this branch.
        if resume_stage:
            raise ValueError("resume_stage requires durable checkpoint state")
        model, fit = core.timed_fit(model, scaled, labels)
    else:
        protocol_state.mark_stage("final_refit", "IN_PROGRESS")
        model, fit = checkpoint_manager.fit_stage(
            model=model,
            X=scaled,
            y=labels,
            stage="final_refit",
            resume=resume_stage,
            progress=progress,
        )
        protocol_state.mark_stage("final_refit", "STAGE_COMPLETE")
    del scaled
    return {
        "model": model,
        "scaler": scaler,
        "features": list(selected_names),
        "fit": fit,
        "fit_rows": rows,
        "fit_role": "train+val",
        "mi_rfe_refit": False,
        "effective_parameters": effective_xgb_parameters(model),
    }


@dataclass
class TestOnceGuard:
    run_directory: Path
    evaluations: int = 0

    def claim(self, final_model_sha256: str) -> None:
        write_json(
            self.run_directory / "TEST_CLAIM.json",
            {
                "schema": STAGE8_SCHEMA,
                "status": "CLAIMED_NOT_YET_VALIDATED",
                "evaluation_number": 1,
                "final_model_sha256": final_model_sha256,
                "automatic_retry_forbidden": True,
            },
            exclusive=True,
        )

    def evaluate(self, fitted: dict, test: core.PreparedSplit) -> dict:
        if self.evaluations:
            raise RuntimeError("Test evaluation has already occurred")
        if not (self.run_directory / "TEST_CLAIM.json").is_file():
            raise RuntimeError("Durable Test claim is required before evaluation")
        self.evaluations += 1
        indices = [core.FEATURE_NAMES.index(name) for name in fitted["features"]]
        scaled = fitted["scaler"].transform(test.X[:, indices])
        started = time.perf_counter()
        prediction = fitted["model"].predict(scaled)
        return {
            "metrics": metrics(test.y, prediction),
            "predict_seconds": time.perf_counter() - started,
            "test_rows_evaluated": len(test.y),
            "test_evaluation_calls": self.evaluations,
        }


def close_split(split: core.PreparedSplit | None) -> None:
    if split is None:
        return
    for array in (split.X, split.y):
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def artifact_manifest(directory: Path) -> dict:
    excluded = {"artifact_manifest.json"}
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {
        "schema": STAGE8_SCHEMA,
        "files": files,
        "manifest_sha256": object_sha256(files),
    }


def run_worker(
    repo: Path,
    windows: Path,
    original: Path,
    output: Path,
    cache_root: Path,
    config: str,
    experiment: str,
    backend: str,
    budget_path: Path,
    unit_test_evidence_path: Path,
    rfe_policy_path: Path | None,
    checkpoint_root: Path,
    checkpoint_interval_rounds: int,
    resume: bool,
    confirmation: str,
) -> dict:
    action = "RESUME" if resume else "RUN"
    expected = f"{action} STAGE8 {experiment} {config} {output.name} {backend}"
    if confirmation != expected:
        raise PermissionError(f"Type exactly: {expected}")
    if experiment != "baseline_xgb":
        raise PermissionError(
            "Official Stage 8 CUDA/checkpoint execution is currently approved only "
            "for baseline_xgb; mi_rfe_xgb remains blocked"
        )
    audit = preflight(
        repo,
        windows,
        original,
        output,
        cache_root,
        config,
        experiment,
        backend,
        budget_path,
        unit_test_evidence_path,
        rfe_policy_path,
        checkpoint_root=checkpoint_root,
        checkpoint_interval_rounds=checkpoint_interval_rounds,
        resume=resume,
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"Stage 8 preflight is BLOCKED: {audit['checks']}")

    staged = output.with_name(output.name + ".incomplete")
    if resume:
        staged.mkdir(parents=True, exist_ok=True)
    else:
        staged.mkdir(parents=True)
    cache_root.mkdir(parents=True, exist_ok=resume)
    context = build_checkpoint_context(
        checkpoint_root=checkpoint_root.resolve(),
        output=output,
        experiment=experiment,
        config=config,
        backend=backend,
        source=audit["source_identity"],
        dataset=audit["dataset_identity"],
        selected_features=list(core.FEATURE_NAMES),
        checkpoint_interval_rounds=checkpoint_interval_rounds,
    )
    checkpoint_manager = xgb_checkpoint.CheckpointManager(context)
    protocol_state = xgb_checkpoint.ProtocolStateStore(context)
    if resume:
        durable_state = protocol_state.load_for_resume()
        write_json(
            staged / f"resume_audit_{time.time_ns()}.json",
            audit,
            exclusive=True,
        )
    else:
        durable_state = protocol_state.create_fresh()
        write_json(staged / "preflight_audit.json", audit)
        write_json(
            staged / "run_identity.json",
            {
                "schema": STAGE8_SCHEMA,
                "run_name": output.name,
                "run_id": context.run_id,
                "experiment": experiment,
                "config": config,
                "backend": backend,
                "source_identity": audit["source_identity"],
                "dataset_identity": audit["dataset_identity"],
                "resume_supported": True,
                "checkpoint_root": str(context.run_root),
                "checkpoint_interval_rounds": checkpoint_interval_rounds,
            },
        )
    validation_resume = resume and durable_state["validation_fit_status"] in {
        "IN_PROGRESS",
        "STAGE_COMPLETE",
    }
    final_refit_resume = resume and durable_state["final_refit_status"] in {
        "IN_PROGRESS",
        "STAGE_COMPLETE",
    }
    store = core.WindowStore(windows)
    before = store.snapshot()
    started = time.perf_counter()
    train = val = test = None
    baseline_validation = None
    test_guard = TestOnceGuard(staged)
    try:
        with core.PeakRSS() as peak, core.RunProgress(
            staged, config, experiment
        ) as progress:
            train = store.load_full(config, "train", cache_root, progress=progress)
            val = store.load_full(config, "val", cache_root, progress=progress)
            write_json(
                staged / "stage_checkpoint.json",
                {
                    "stage": "train_validation_loaded",
                    "resume_supported": True,
                    "resume_requested": resume,
                    "resume_accepted": bool(resume),
                    "checkpoint_root": str(context.run_root),
                    "checkpoint_interval_rounds": checkpoint_interval_rounds,
                    "train_rows": len(train.y),
                    "validation_rows": len(val.y),
                    "test_feature_rows_read": 0,
                },
            )

            if experiment == "baseline_xgb":
                baseline_validation = validation_baseline_xgb(
                    train,
                    val,
                    backend,
                    checkpoint_manager,
                    protocol_state,
                    resume_stage=validation_resume,
                    progress=progress,
                )
                selected = list(core.FEATURE_NAMES)
                best_k = 380
                best_validation_accuracy = baseline_validation["metrics"]["accuracy"]
                selection_summary = {
                    "criterion": "fixed_380_baseline",
                    "validation_is_reporting_only": True,
                    "validation": baseline_validation,
                }
            else:
                selection = select_mi_rfe_xgb(
                    train,
                    val,
                    backend,
                    staged,
                    progress=progress,
                )
                selected = list(selection["features"])
                best_k = int(selection["best_k"])
                best_validation_accuracy = float(
                    selection["best_validation_accuracy"]
                )
                selection_summary = {
                    "criterion": "validation_accuracy_strict_greater_smallest_k_tie",
                    "mi_top100": selection["mi"]["names"],
                    "rfe_policy_id": RFE_POLICY_ID,
                    "rfe_parameters": dict(core.RF_RFE_PARAMS),
                    "forest_fits": selection["ranking"]["forest_fits"],
                }

            if baseline_validation is None:
                raise RuntimeError("baseline_xgb validation result is missing")

            write_json(
                staged / "selection_frozen.json",
                {
                    "best_k": best_k,
                    "features": selected,
                    "best_validation_accuracy": best_validation_accuracy,
                    **selection_summary,
                },
            )
            progress(
                "selection_frozen",
                best_k=best_k,
                validation_accuracy=best_validation_accuracy,
            )
            progress(
                "final_refit_started",
                train_val_rows=len(train.y) + len(val.y),
                features=len(selected),
            )
            fitted = fit_final_xgb(
                train,
                val,
                selected,
                backend,
                checkpoint_manager,
                protocol_state,
                resume_stage=final_refit_resume,
                progress=progress,
            )
            fitted["model"].save_model(str(staged / "model.ubj"))
            joblib.dump(fitted["scaler"], staged / "final_scaler.joblib")
            write_json(staged / "features.json", selected)
            write_json(
                staged / "hyperparameters.json",
                {
                    "xgboost_version": SUPPORTED_XGBOOST_VERSION,
                    "backend": backend,
                    "effective": fitted["effective_parameters"],
                    "rf_rfe": (
                        None
                        if experiment == "baseline_xgb"
                        else dict(core.RF_RFE_PARAMS)
                    ),
                    "no_class_balancing": True,
                },
            )
            booster_config = json.loads(fitted["model"].get_booster().save_config())
            write_json(staged / "booster_config.json", booster_config)
            final_model_hash = sha256(staged / "model.ubj")
            write_json(
                staged / "stage_checkpoint.json",
                {
                    "stage": "final_refit_completed",
                    "resume_supported": True,
                    "resume_requested": resume,
                    "resume_accepted": bool(resume),
                    "checkpoint_root": str(context.run_root),
                    "completed_rounds": fitted["fit"]["completed_rounds"],
                    "remaining_rounds": fitted["fit"]["remaining_rounds"],
                    "total_rounds": fitted["fit"]["total_rounds"],
                    "checkpoint_sha256": fitted["fit"]["checkpoint_sha256"],
                    "model_sha256": final_model_hash,
                    "fit_rows": fitted["fit_rows"],
                    "test_feature_rows_read": 0,
                },
            )
            progress("final_refit_completed", **fitted["fit"])

            protocol_state.request_test_once()
            test_guard.claim(final_model_hash)
            progress("test_claimed", evaluation_number=1)
            test = store.load_full(config, "test", cache_root, progress=progress)
            progress(
                "test_evaluation_started",
                rows=len(test.y),
                evaluation_number=1,
            )
            evaluated = test_guard.evaluate(fitted, test)
            write_json(
                staged / "test_metrics.json",
                {
                    "schema": STAGE8_SCHEMA,
                    "metrics": evaluated["metrics"],
                    "test_rows_evaluated": evaluated["test_rows_evaluated"],
                    "test_evaluation_calls": evaluated["test_evaluation_calls"],
                },
                exclusive=True,
            )
            protocol_state.complete_test_once(
                metrics_sha256=sha256(staged / "test_metrics.json")
            )
            progress("test_evaluation_completed", **evaluated["metrics"])

            rows = {
                "train": len(train.y),
                "val": len(val.y),
                "test": len(test.y),
                "final_fit_train_val": fitted["fit_rows"],
            }
            labels = {
                split.role: {
                    "benign": int((split.y == 0).sum()),
                    "attack": int((split.y == 1).sum()),
                }
                for split in (train, val, test)
            }
            if {
                key: rows[key] for key in ("train", "val", "test")
            } != audit["rows_by_split"]:
                raise ValueError("Rows differ from the approved window manifest")
            if store.snapshot() != before:
                raise ValueError("Window size/mtime snapshot changed during Stage 8")
            if core.source_hashes(original) != audit["source_identity"][
                "original_ctu13_sha256"
            ]:
                raise ValueError("CTU-13 originals changed during Stage 8")

            peak._sample()
            result = {
                "schema": STAGE8_SCHEMA,
                "status": "PASS",
                "experiment": experiment,
                "config": config,
                "run_name": output.name,
                "backend": backend,
                "xgboost_version": SUPPORTED_XGBOOST_VERSION,
                "rows_by_split": rows,
                "label_counts_by_split": labels,
                "best_k": best_k,
                "selected_features": selected,
                "feature_reduction_from_380": 1.0 - len(selected) / 380.0,
                "best_validation_accuracy": best_validation_accuracy,
                "metrics": evaluated["metrics"],
                "test_rows_evaluated": evaluated["test_rows_evaluated"],
                "test_evaluation_calls": evaluated["test_evaluation_calls"],
                "test_read_after_final_refit": True,
                "final_fit": fitted["fit"],
                "test_predict_seconds": evaluated["predict_seconds"],
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes_sampled": peak.peak,
                "effective_hyperparameters": fitted["effective_parameters"],
                "source_identity": audit["source_identity"],
                "dataset_identity": audit["dataset_identity"],
                "label_origin": "SCENARIO_PROXY",
                "capture_session_independence": CAPTURE_SESSION_INDEPENDENCE,
                "sampled_training": False,
                "resume_supported": True,
                "resume_requested": resume,
                "resume_accepted": bool(resume),
                "checkpoint_root": str(context.run_root),
                "checkpoint_interval_rounds": checkpoint_interval_rounds,
                "checkpoint_verified": True,
                "validation_fit_checkpoint": baseline_validation["fit"],
                "final_refit_checkpoint": fitted["fit"],
                "resume_from_stage": fitted["fit"]["resume_from_stage"],
                "resume_from_round": fitted["fit"]["resume_from_round"],
                "completed_rounds": fitted["fit"]["completed_rounds"],
                "remaining_rounds": fitted["fit"]["remaining_rounds"],
                "total_rounds": fitted["fit"]["total_rounds"],
                "checkpoint_sha256": fitted["fit"]["checkpoint_sha256"],
            }
            write_json(staged / "result.json", result)
            progress("run_validated", status="PASS", **audit["rows_by_split"])
            write_json(
                staged / "RUN_VALIDATED.json",
                {
                    "status": "PASS",
                    "test_evaluation_calls": 1,
                    "result_sha256": sha256(staged / "result.json"),
                    "model_sha256": final_model_hash,
                },
            )
            write_json(staged / "artifact_manifest.json", artifact_manifest(staged))

        close_split(train)
        close_split(val)
        close_split(test)
        train = val = test = None
        os.replace(staged, output)
        return result
    except BaseException as exc:
        failure_path = staged / "FAILED.json"
        if failure_path.exists():
            failure_path = staged / f"FAILED_{time.time_ns()}.json"
        write_json(
            failure_path,
            {
                "status": "INCOMPLETE_DO_NOT_USE",
                "error": str(exc),
                "test_claim_exists": (staged / "TEST_CLAIM.json").is_file(),
                "test_evaluation_calls_in_process": test_guard.evaluations,
                "automatic_retry_forbidden": True,
            },
        )
        raise
    finally:
        close_split(train)
        close_split(val)
        close_split(test)


def tree_rss(process: psutil.Process) -> int:
    total = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def stop_owned_tree(process: psutil.Process) -> None:
    owned = []
    try:
        owned = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for item in reversed(owned):
        try:
            item.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    _, alive = psutil.wait_procs([*owned, process], timeout=10)
    for item in alive:
        try:
            item.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def supervise(args) -> int:
    resume_requested = bool(
        args.resume or args.confirmation.startswith("RESUME STAGE8 ")
    )
    checkpoint_root = (
        args.checkpoint_root
        if args.checkpoint_root is not None
        else args.output.parent.parent / "checkpoints" / "stage8_xgboost"
    )
    audit = preflight(
        args.repo,
        args.windows,
        args.original,
        args.output,
        args.cache_root,
        args.config,
        args.experiment,
        args.backend,
        args.budget,
        args.unit_test_evidence,
        args.rfe_policy,
        checkpoint_root=checkpoint_root,
        checkpoint_interval_rounds=args.checkpoint_interval_rounds,
        resume=resume_requested,
    )
    if args.preflight_report.exists():
        raise FileExistsError(args.preflight_report)
    write_json(args.preflight_report, audit, exclusive=True)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Stage 8 preflight is BLOCKED: {audit['checks']}")
    action = "RESUME" if resume_requested else "RUN"
    expected = (
        f"{action} STAGE8 {args.experiment} {args.config} "
        f"{args.output.name} {args.backend}"
    )
    if args.confirmation != expected:
        raise PermissionError(f"Type exactly: {expected}")

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--repo",
        str(args.repo),
        "--windows",
        str(args.windows),
        "--original",
        str(args.original),
        "--output",
        str(args.output),
        "--cache-root",
        str(args.cache_root),
        "--config",
        args.config,
        "--experiment",
        args.experiment,
        "--backend",
        args.backend,
        "--budget",
        str(args.budget),
        "--unit-test-evidence",
        str(args.unit_test_evidence),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-interval-rounds",
        str(args.checkpoint_interval_rounds),
        "--confirmation",
        args.confirmation,
    ]
    if resume_requested:
        command.append("--resume")
    if args.rfe_policy is not None:
        command.extend(["--rfe-policy", str(args.rfe_policy)])
    child = subprocess.Popen(command)
    process = psutil.Process(child.pid)
    budget = load_budget(args.budget)
    started = time.monotonic()
    peak = 0
    stop_reason = None
    while child.poll() is None:
        rss = tree_rss(process)
        peak = max(peak, rss)
        elapsed = time.monotonic() - started
        available = psutil.virtual_memory().available
        disk_anchor = args.cache_root.parent
        while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
            disk_anchor = disk_anchor.parent
        disk_free = psutil.disk_usage(str(disk_anchor)).free
        if rss > budget["max_tree_rss_bytes"]:
            stop_reason = "MAX_TREE_RSS_EXCEEDED"
        elif elapsed > budget["max_elapsed_seconds"]:
            stop_reason = "MAX_ELAPSED_EXCEEDED"
        elif available < budget["min_runtime_available_ram_bytes"]:
            stop_reason = "MIN_AVAILABLE_RAM_VIOLATED"
        elif disk_free < budget["min_free_disk_bytes"]:
            stop_reason = "MIN_FREE_DISK_VIOLATED"
        if stop_reason:
            stop_owned_tree(process)
            break
        time.sleep(budget["poll_interval_seconds"])

    returncode = child.wait()
    supervisor_record = {
        "schema": STAGE8_SCHEMA,
        "status": "PASS" if returncode == 0 and stop_reason is None else "STOPPED_OR_FAILED",
        "returncode": returncode,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "peak_tree_rss_bytes": peak,
        "automatic_retry": False,
        "backend": args.backend,
        "resume_supported": True,
        "resume_requested": resume_requested,
        "resume_accepted": bool(resume_requested and audit["resume_accepted"]),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_interval_rounds": args.checkpoint_interval_rounds,
        "checkpoint_verified": audit["checkpoint_verified"],
        "resume_from_stage": audit["resume_from_stage"],
        "resume_from_round": audit["resume_from_round"],
        "total_rounds": XGB_REQUIRED_PARAMS["n_estimators"],
        "output": str(args.output),
    }
    record_name = (
        f"{args.output.name}.resume-{time.time_ns()}.supervisor.json"
        if resume_requested
        else f"{args.output.name}.supervisor.json"
    )
    record_path = args.output.parent / record_name
    write_json(record_path, supervisor_record, exclusive=True)
    if supervisor_record["status"] != "PASS":
        raise RuntimeError(f"Stage 8 worker failed/stopped: {supervisor_record}")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-interval-rounds", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--unit-test-evidence", type=Path, required=True)
    parser.add_argument("--rfe-policy", type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("preflight")
    add_common(inspect_parser)
    inspect_parser.add_argument("--report", type=Path, required=True)
    supervisor = commands.add_parser("supervise")
    add_common(supervisor)
    supervisor.add_argument("--preflight-report", type=Path, required=True)
    supervisor.add_argument("--confirmation", required=True)
    worker = commands.add_parser("_worker")
    add_common(worker)
    worker.add_argument("--confirmation", required=True)
    args = parser.parse_args()

    if args.command == "preflight":
        if args.report.exists():
            raise FileExistsError(args.report)
        report = preflight(
            args.repo,
            args.windows,
            args.original,
            args.output,
            args.cache_root,
            args.config,
            args.experiment,
            args.backend,
            args.budget,
            args.unit_test_evidence,
            args.rfe_policy,
            checkpoint_root=args.checkpoint_root,
            checkpoint_interval_rounds=args.checkpoint_interval_rounds,
            resume=args.resume,
        )
        write_json(args.report, report, exclusive=True)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["status"] != "PASS":
            raise SystemExit(2)
    elif args.command == "supervise":
        raise SystemExit(supervise(args))
    else:
        checkpoint_root = (
            args.checkpoint_root
            if args.checkpoint_root is not None
            else args.output.parent.parent / "checkpoints" / "stage8_xgboost"
        )
        run_worker(
            args.repo,
            args.windows,
            args.original,
            args.output,
            args.cache_root,
            args.config,
            args.experiment,
            args.backend,
            args.budget,
            args.unit_test_evidence,
            args.rfe_policy,
            checkpoint_root,
            args.checkpoint_interval_rounds,
            args.resume,
            args.confirmation,
        )


if __name__ == "__main__":
    main()
