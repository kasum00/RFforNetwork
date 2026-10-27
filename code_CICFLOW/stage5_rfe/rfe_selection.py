"""Stage 5A Train-only Random-Forest RFE after a Stage 4 MI Top-100 run.

The implementation is deliberately separate from Stage 4.  It consumes the
immutable Stage 4 binding, reads only the corresponding Train Parquet columns,
and performs two independent RFE branches.  The manual step-one loop mirrors
the public sklearn RFE elimination semantics so a checkpoint can be written
after every elimination without pretending that an interrupted sklearn RFE
object is resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from stage4_mi.mi_selection import (
    CONFIG_EXPECTED_ROWS,
    LABEL_COLUMN,
    Stage4ContractError,
    expected_full_schema,
    inspect_train_footer,
    load_feature_schema,
    resolve_train_path,
    source_identity as stage4_source_identity,
    stable_digest,
)


ORIGINAL_FEATURE_COUNT = 380
MI_TOP_K = 100
RFE_N_FEATURES_TO_SELECT = 1
RFE_STEP = 1
REFERENCE_STATUS = "PROVISIONAL_REFERENCE_UNVERIFIED"
SCIENTIFIC_STATUS = "PROVISIONAL_RESEARCH"

BRANCH_CONTRACTS: dict[str, dict[str, Any]] = {
    "rf": {
        "label": "RF",
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "unresolved_parameters": [],
    },
    "svm": {
        "label": "SVM-RFE",
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 15,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "unresolved_parameters": ["min_samples_split", "max_features"],
        "unresolved_policy": "sklearn constructor defaults are used provisionally; source reconciliation is required before promotion",
    },
}


class Stage5ContractError(ValueError):
    """Raised when a Stage 5A scientific or provenance contract is invalid."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Stage5ContractError(f"Expected a JSON object: {path}")
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    write_exclusive(path, json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def write_json_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def validate_run_id(run_id: str) -> str:
    value = str(run_id)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise Stage5ContractError("run-id must be a non-empty directory name without path separators")
    return value


def validate_branch(branch: str) -> str:
    value = str(branch).lower()
    if value not in {"rf", "svm", "xgboost"}:
        raise Stage5ContractError(f"Unsupported Stage 5A branch: {branch}")
    if value == "xgboost":
        raise Stage5ContractError("XGBoost Stage 5A is BLOCKED: ranking contract is not verified")
    return value


def validate_mi_top100_payload(
    payload: Mapping[str, Any],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate MI Top-100 order and name/index mapping without data values."""

    if len(feature_names) != ORIGINAL_FEATURE_COUNT:
        raise Stage5ContractError("The bound source schema must contain 380 features")
    if payload.get("top_k") != MI_TOP_K:
        raise Stage5ContractError("Stage 4 payload must declare top_k=100")
    rows = payload.get("features")
    if not isinstance(rows, list) or len(rows) != MI_TOP_K:
        raise Stage5ContractError("Stage 4 payload must contain exactly 100 features")

    result: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_names: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise Stage5ContractError("Stage 4 Top-100 entries must be JSON objects")
        rank = row.get("rank")
        index = row.get("feature_index")
        name = row.get("feature_name")
        score = row.get("mi_score")
        if rank != position or not isinstance(index, int) or not isinstance(name, str):
            raise Stage5ContractError("Stage 4 Top-100 rank/order fields are invalid")
        if index < 0 or index >= ORIGINAL_FEATURE_COUNT:
            raise Stage5ContractError("Stage 4 Top-100 feature index is outside 0..379")
        if feature_names[index] != name:
            raise Stage5ContractError("Stage 4 feature name/index mapping does not match source schema")
        if index in seen_indices or name in seen_names:
            raise Stage5ContractError("Stage 4 Top-100 contains duplicate features")
        if not np.isfinite(float(score)):
            raise Stage5ContractError("Stage 4 Top-100 contains a non-finite MI score")
        seen_indices.add(index)
        seen_names.add(name)
        result.append(
            {
                "rank": int(rank),
                "feature_index": int(index),
                "feature_name": name,
                "mi_score": float(score),
            }
        )
    return result


def _required_stage4_paths(mi_run_dir: Path) -> dict[str, Path]:
    return {
        "top100": mi_run_dir / "top100_features.json",
        "manifest": mi_run_dir / "mi_manifest.json",
        "complete": mi_run_dir / "MI_COMPLETE.json",
        "ranking": mi_run_dir / "mi_ranking.json",
    }


def load_stage4_binding(
    *,
    dataset_root: Path,
    config: str,
    mi_run_dir: Path,
) -> dict[str, Any]:
    """Validate the exact Stage 4 run and bind its ordered Top-100 input."""

    if config not in CONFIG_EXPECTED_ROWS:
        raise Stage5ContractError(f"Unsupported configuration: {config}")
    paths = _required_stage4_paths(mi_run_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise Stage5ContractError(f"Stage 4 binding files are missing: {missing}")

    manifest = read_json(paths["manifest"])
    complete = read_json(paths["complete"])
    ranking_payload = read_json(paths["ranking"])
    top_payload = read_json(paths["top100"])
    if complete.get("status") != "COMPLETE":
        raise Stage5ContractError("Stage 4 MI_COMPLETE marker is not COMPLETE")
    if complete.get("config") != config or manifest.get("config") != config:
        raise Stage5ContractError("Stage 4 config identity does not match Stage 5 config")
    if manifest.get("official_reproduction") is not False:
        raise Stage5ContractError("Stage 4 official_reproduction flag must remain false")

    config_dir = dataset_root / config
    feature_names = load_feature_schema(config_dir)
    top_features = validate_mi_top100_payload(top_payload, feature_names)
    ranking = ranking_payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != ORIGINAL_FEATURE_COUNT:
        raise Stage5ContractError("Stage 4 full ranking must contain all 380 features")
    if stable_digest(ranking) != complete.get("ranking_digest"):
        raise Stage5ContractError("Stage 4 full ranking digest does not match MI_COMPLETE")
    if ranking[:MI_TOP_K] != top_features:
        raise Stage5ContractError("Stage 4 Top-100 is not the prefix of the bound full ranking")
    if manifest.get("train_feature_count") != ORIGINAL_FEATURE_COUNT:
        raise Stage5ContractError("Stage 4 manifest does not bind 380 source features")
    if manifest.get("label_column") != LABEL_COLUMN:
        raise Stage5ContractError("Stage 4 label column does not match Window_Label")

    train_path = resolve_train_path(config_dir)
    footer = inspect_train_footer(train_path, feature_names)
    if footer["rows"] != CONFIG_EXPECTED_ROWS[config]:
        raise Stage5ContractError("Stage 4/Stage 5 Train row count does not match the contract")
    current_identity = stage4_source_identity(config, config_dir, feature_names, footer)
    bound_identity = manifest.get("source_identity", {})
    if bound_identity.get("identity_digest") != current_identity.get("identity_digest"):
        raise Stage5ContractError("Stage 4 source identity no longer matches the Train source")
    if bound_identity.get("feature_names_digest") != current_identity.get("feature_names_digest"):
        raise Stage5ContractError("Stage 4 feature schema identity no longer matches")

    return {
        "config": config,
        "mi_run_dir": str(mi_run_dir),
        "mi_run_id": mi_run_dir.name,
        "config_dir": str(config_dir),
        "train_path": str(train_path),
        "feature_names": feature_names,
        "top_features": top_features,
        "train_footer": footer,
        "source_identity": current_identity,
        "stage4_manifest_digest": hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
        "stage4_complete_digest": hashlib.sha256(paths["complete"].read_bytes()).hexdigest(),
        "stage4_ranking_digest": complete.get("ranking_digest"),
        "validation_test_values_read": False,
    }


def preprocess_top_matrix(X: Any, y: Any, expected_features: int = MI_TOP_K) -> tuple[np.ndarray, np.ndarray]:
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Stage5ContractError("Train RFE features must be numeric") from exc
    labels = np.asarray(y)
    if matrix.ndim != 2 or matrix.shape[1] != expected_features:
        raise Stage5ContractError(f"Train RFE matrix must have {expected_features} columns")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise Stage5ContractError("Train labels must be row-aligned")
    if np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels.astype(np.float64)).all():
        raise Stage5ContractError("Train labels contain NaN or Inf")
    if np.unique(labels).size != 2:
        raise Stage5ContractError("Window_Label must contain exactly two classes")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return matrix, labels.copy()


def read_train_top_values(
    train_path: Path,
    feature_names: Sequence[str],
    table_reader: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only selected Train columns and Window_Label; never Val/Test."""

    columns = list(feature_names) + [LABEL_COLUMN]
    if table_reader is None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise Stage5ContractError("pyarrow is required for Stage 5 Train value access") from exc
        table = parquet.read_table(train_path, columns=columns)
    else:
        table = table_reader(train_path, columns=columns)
    if list(table.schema.names) != columns:
        raise Stage5ContractError("Train selected-column order does not match Stage 4 Top-100 order")
    arrays = [table.column(name).to_numpy(zero_copy_only=False) for name in feature_names]
    X = np.column_stack(arrays)
    y = table.column(LABEL_COLUMN).to_numpy(zero_copy_only=False)
    return preprocess_top_matrix(X, y, expected_features=len(feature_names))


def branch_contract(branch: str) -> dict[str, Any]:
    branch = validate_branch(branch)
    return json.loads(json.dumps(BRANCH_CONTRACTS[branch]))


def make_estimator(branch: str) -> Any:
    contract = branch_contract(branch)
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Stage5ContractError("scikit-learn is required for Stage 5A") from exc
    return RandomForestClassifier(**contract["parameters"])


def _feature_importances(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=np.float64)
        if values.ndim > 1:
            values = np.sum(values**2, axis=0)
        else:
            values = values**2
        return values
    else:
        raise Stage5ContractError("RFE estimator has no feature_importances_ or coef_ attribute")
    if values.ndim != 1:
        raise Stage5ContractError("RFE feature importances must be one-dimensional")
    # sklearn RFE's default importance_getter='auto' uses square transform.
    return values**2


def manual_rfe_step1(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    *,
    n_features_to_select: int = RFE_N_FEATURES_TO_SELECT,
    step: int = RFE_STEP,
    initial_support: Sequence[bool] | None = None,
    initial_ranking: Sequence[int] | None = None,
    completed_steps: int = 0,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run sklearn-RFE-equivalent elimination with a checkpoint per step."""

    if step != 1:
        raise Stage5ContractError("Stage 5A requires RFE step=1")
    if X.ndim != 2 or X.shape[0] != len(y):
        raise Stage5ContractError("RFE matrix and labels are not row-aligned")
    feature_count = X.shape[1]
    if not 1 <= n_features_to_select <= feature_count:
        raise Stage5ContractError("Invalid n_features_to_select")
    support = np.ones(feature_count, dtype=bool) if initial_support is None else np.asarray(initial_support, dtype=bool).copy()
    ranking = np.ones(feature_count, dtype=int) if initial_ranking is None else np.asarray(initial_ranking, dtype=int).copy()
    if support.shape != (feature_count,) or ranking.shape != (feature_count,):
        raise Stage5ContractError("RFE checkpoint shape does not match selected feature count")

    while int(np.sum(support)) > n_features_to_select:
        active = np.arange(feature_count)[support]
        current_estimator = __import__("sklearn.base", fromlist=["clone"]).clone(estimator)
        current_estimator.fit(X[:, active], y)
        importances = _feature_importances(current_estimator)
        if importances.shape != (active.size,) or not np.isfinite(importances).all():
            raise Stage5ContractError("RFE estimator returned invalid feature importances")
        # sklearn RFE uses np.argsort(importances) here. Keep the same default
        # sort semantics instead of introducing a custom tie-break rule.
        elimination_order = np.argsort(importances)
        threshold = min(step, int(np.sum(support)) - n_features_to_select)
        eliminated = active[elimination_order[:threshold]]
        support[eliminated] = False
        ranking[~support] += 1
        completed_steps += 1
        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "completed_steps": completed_steps,
                    "active_indices": np.flatnonzero(support).tolist(),
                    "support": support.tolist(),
                    "ranking": ranking.tolist(),
                    "eliminated_indices": eliminated.tolist(),
                }
            )
    return ranking, support, completed_steps


def ranking_records(
    top_features: Sequence[Mapping[str, Any]],
    ranking: Sequence[int],
    support: Sequence[bool],
) -> list[dict[str, Any]]:
    if len(top_features) != len(ranking) or len(ranking) != len(support):
        raise Stage5ContractError("RFE ranking shape mismatch")
    ordered_indices = sorted(range(len(ranking)), key=lambda index: (int(ranking[index]), index))
    return [
        {
            "selection_order": order,
            "rfe_rank": int(ranking[index]),
            "selected": bool(support[index]),
            "mi_rank": int(top_features[index]["rank"]),
            "feature_index": int(top_features[index]["feature_index"]),
            "feature_name": str(top_features[index]["feature_name"]),
            "mi_score": float(top_features[index]["mi_score"]),
        }
        for order, index in enumerate(ordered_indices, start=1)
    ]


def _library_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "pyarrow", "psutil"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "MISSING"
    return result


def collect_resource_snapshot(config_dir: Path, run_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        "library_versions": _library_versions(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot["memory"] = {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "percent_used": float(memory.percent),
        }
    except ImportError:
        snapshot["memory"] = {"status": "NOT_CHECKED", "reason": "psutil unavailable"}
    usage = shutil.disk_usage(run_dir.parent if run_dir.parent.exists() else config_dir)
    snapshot["disk"] = {"free_bytes": int(usage.free), "total_bytes": int(usage.total)}
    return snapshot


def resource_gate(snapshot: Mapping[str, Any], rows: int, selected_features: int = MI_TOP_K) -> dict[str, Any]:
    matrix_bytes = int(rows) * int(selected_features) * 4
    labels_bytes = int(rows) * 8
    required_ram = int((matrix_bytes + labels_bytes) * 6.0)
    reasons: list[str] = []
    memory = snapshot.get("memory", {})
    available = memory.get("available_bytes")
    if not isinstance(available, int):
        reasons.append("available RAM could not be measured")
    elif available < required_ram:
        reasons.append(f"available RAM {available} is below Stage 5 estimate {required_ram}")
    free_disk = snapshot.get("disk", {}).get("free_bytes")
    if not isinstance(free_disk, int):
        reasons.append("free disk space could not be measured")
    elif free_disk < 2 * 1024**3:
        reasons.append("less than 2 GiB free disk space")
    versions = snapshot.get("library_versions", {})
    for package in ("numpy", "scikit-learn", "pyarrow"):
        if versions.get(package) in {None, "MISSING"}:
            reasons.append(f"required library missing: {package}")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "required_ram_estimate_bytes": required_ram,
        "available_ram_bytes": available,
        "free_disk_bytes": free_disk,
        "reasons": reasons,
        "no_sampling_or_parameter_reduction": True,
    }


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        name: run_dir / name
        for name in (
            "RUN_INCOMPLETE.json",
            "BLOCKED.json",
            "FAILED.json",
            "rfe_progress.json",
            "rfe_ranking.json",
            "rfe_ranking.csv",
            "rfe_manifest.json",
            "resource_usage.json",
            "rfe_report.md",
            "COMPLETE.json",
        )
    }


def _make_run_identity(
    *, config: str, branch: str, run_id: str, binding: Mapping[str, Any], git_commit: str
) -> dict[str, Any]:
    return {
        "stage": "5A",
        "config": config,
        "branch": branch,
        "run_id": run_id,
        "git_commit": git_commit or "NOT_CAPTURED",
        "stage4_mi_run_id": binding["mi_run_id"],
        "stage4_mi_identity_digest": binding["source_identity"]["identity_digest"],
        "stage4_mi_ranking_digest": binding["stage4_ranking_digest"],
        "dataset_fingerprint": binding["source_identity"]["identity_digest"],
        "top100_order_digest": stable_digest(binding["top_features"]),
        "branch_contract": branch_contract(branch),
        "rfe_parameters": {
            "n_features_to_select": RFE_N_FEATURES_TO_SELECT,
            "step": RFE_STEP,
            "importance_getter": "auto",
            "manual_loop_equivalent": "sklearn.RFE step=1",
        },
        "scientific_status": SCIENTIFIC_STATUS,
        "reference_status": REFERENCE_STATUS,
    }


def _validate_resume(run_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    paths = _run_paths(run_dir)
    if not paths["RUN_INCOMPLETE.json"].exists() or not paths["rfe_progress.json"].exists():
        raise Stage5ContractError("Resume requires RUN_INCOMPLETE.json and rfe_progress.json")
    if paths["COMPLETE.json"].exists():
        raise Stage5ContractError("Completed Stage 5A run cannot be resumed")
    manifest = read_json(paths["rfe_manifest.json"])
    if manifest.get("run_identity") != dict(identity):
        raise Stage5ContractError("Stage 5 resume identity does not match exactly")
    progress = read_json(paths["rfe_progress.json"])
    if progress.get("run_identity") != dict(identity):
        raise Stage5ContractError("Stage 5 checkpoint identity does not match exactly")
    return progress


def _write_ranking_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_order",
                "rfe_rank",
                "selected",
                "mi_rank",
                "feature_index",
                "feature_name",
                "mi_score",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def _render_report(config: str, branch: str, identity: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"# Stage 5A RF-RFE — {config} / {branch}",
        "",
        f"- Scientific status: `{SCIENTIFIC_STATUS}`",
        f"- Reference status: `{REFERENCE_STATUS}`",
        f"- Stage 4 MI identity: `{identity['stage4_mi_identity_digest']}`",
        f"- Dataset fingerprint: `{identity['dataset_fingerprint']}`",
        f"- RFE estimator: `{identity['branch_contract']['estimator']}`",
        f"- RFE parameters: `n_features_to_select=1`, `step=1`, `importance_getter=auto`",
        "- Values read: Train selected columns only; Validation/Test values were not opened.",
        "",
        "## First 10 RFE records",
        "",
        "| Order | RFE rank | Selected | Feature | MI rank |",
        "|---:|---:|:---:|---|---:|",
    ]
    lines.extend(
        f"| {row['selection_order']} | {row['rfe_rank']} | {row['selected']} | `{row['feature_name']}` | {row['mi_rank']} |"
        for row in records[:10]
    )
    lines.extend(
        [
            "",
            "This is a provisional Stage 5A result. It is not an official CTU-13 reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def run_rfe(
    *,
    dataset_root: Path,
    mi_runs_root: Path,
    stage5_runs_root: Path,
    config: str,
    branch: str,
    run_id: str,
    git_commit: str = "NOT_CAPTURED",
    resume: bool = False,
) -> Path:
    branch = validate_branch(branch)
    run_id = validate_run_id(run_id)
    mi_run_dir = mi_runs_root / config / run_id
    binding = load_stage4_binding(dataset_root=dataset_root, config=config, mi_run_dir=mi_run_dir)
    run_dir = stage5_runs_root / config / run_id / branch
    paths = _run_paths(run_dir)
    identity = _make_run_identity(
        config=config,
        branch=branch,
        run_id=run_id,
        binding=binding,
        git_commit=git_commit or "NOT_CAPTURED",
    )

    if resume:
        if not run_dir.is_dir():
            raise Stage5ContractError(f"Stage 5 run directory does not exist: {run_dir}")
        progress = _validate_resume(run_dir, identity)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        progress = {
            "status": "IN_PROGRESS",
            "run_identity": identity,
            "completed_steps": 0,
            "active_indices": list(range(MI_TOP_K)),
            "support": [True] * MI_TOP_K,
            "ranking": [1] * MI_TOP_K,
        }
        write_json_exclusive(paths["RUN_INCOMPLETE.json"], {"status": "INCOMPLETE", "run_identity": identity})
        write_json_exclusive(paths["rfe_manifest.json"], {"status": "INCOMPLETE", "run_identity": identity, "binding": binding})

    resource_snapshot = collect_resource_snapshot(Path(binding["config_dir"]), run_dir)
    resource_result = resource_gate(resource_snapshot, binding["train_footer"]["rows"])
    if not paths["resource_usage.json"].exists():
        write_json_exclusive(paths["resource_usage.json"], {"snapshot": resource_snapshot, "gate": resource_result})
    if resource_result["status"] != "PASS":
        if not paths["BLOCKED.json"].exists():
            write_json_exclusive(
                paths["BLOCKED.json"],
                {
                    "status": "BLOCKED",
                    "reason": "Stage 5A resource gate",
                    "run_identity": identity,
                    "resource_gate": resource_result,
                },
            )
        raise Stage5ContractError("Stage 5A resource gate BLOCKED; no Train values were read")
    try:
        X, y = read_train_top_values(
            Path(binding["train_path"]),
            [row["feature_name"] for row in binding["top_features"]],
        )
        estimator = make_estimator(branch)
    except Exception as exc:
        if not paths["FAILED.json"].exists():
            write_json_exclusive(paths["FAILED.json"], {"status": "FAILED", "error": repr(exc)})
        raise

    def checkpoint(state: dict[str, Any]) -> None:
        updated = {
            "status": "IN_PROGRESS",
            "run_identity": identity,
            **state,
        }
        write_json_checkpoint(paths["rfe_progress.json"], updated)

    if not paths["rfe_progress.json"].exists():
        write_json_exclusive(paths["rfe_progress.json"], progress)
    try:
        ranking, support, completed_steps = manual_rfe_step1(
            X,
            y,
            estimator,
            n_features_to_select=RFE_N_FEATURES_TO_SELECT,
            step=RFE_STEP,
            initial_support=progress.get("support"),
            initial_ranking=progress.get("ranking"),
            completed_steps=int(progress.get("completed_steps", 0)),
            checkpoint_callback=checkpoint,
        )
    except Exception as exc:
        if not paths["FAILED.json"].exists():
            write_json_exclusive(paths["FAILED.json"], {"status": "FAILED", "error": repr(exc)})
        raise

    records = ranking_records(binding["top_features"], ranking, support)
    final_manifest = {
        "status": "COMPLETE",
        "run_identity": identity,
        "binding": binding,
        "train_rows_observed": int(X.shape[0]),
        "selected_feature_count": int(X.shape[1]),
        "completed_steps": int(completed_steps),
        "validation_test_values_read": False,
        "official_reproduction": False,
        "ranking_digest": stable_digest(records),
    }
    write_json_exclusive(paths["rfe_ranking.json"], {"ranking": records, "raw_ranking": ranking, "support": support})
    _write_ranking_csv(paths["rfe_ranking.csv"], records)
    write_json_checkpoint(paths["rfe_progress.json"], {"status": "COMPLETE", "run_identity": identity, "completed_steps": completed_steps, "active_indices": np.flatnonzero(support).tolist(), "support": support.tolist(), "ranking": ranking.tolist()})
    write_json_checkpoint(paths["rfe_manifest.json"], final_manifest)
    write_exclusive(paths["rfe_report.md"], _render_report(config, branch, identity, records))
    write_json_exclusive(paths["COMPLETE.json"], {"status": "COMPLETE", "run_identity_digest": stable_digest(identity), "ranking_digest": final_manifest["ranking_digest"], "completed_steps": completed_steps})
    return run_dir


def verify_run(run_dir: Path) -> dict[str, Any]:
    paths = _run_paths(run_dir)
    required = [paths["COMPLETE.json"], paths["rfe_manifest.json"], paths["rfe_ranking.json"], paths["rfe_ranking.csv"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"status": "FAIL", "run_dir": str(run_dir), "missing": missing}
    manifest = read_json(paths["rfe_manifest.json"])
    ranking_payload = read_json(paths["rfe_ranking.json"])
    records = ranking_payload.get("ranking", [])
    errors: list[str] = []
    if manifest.get("status") != "COMPLETE":
        errors.append("manifest is not COMPLETE")
    if manifest.get("official_reproduction") is not False:
        errors.append("official_reproduction must remain false")
    if len(records) != MI_TOP_K:
        errors.append("RFE ranking must contain 100 features")
    if len({row.get("feature_name") for row in records}) != MI_TOP_K:
        errors.append("RFE ranking contains duplicate feature names")
    if stable_digest(records) != manifest.get("ranking_digest"):
        errors.append("RFE ranking digest mismatch")
    return {"status": "PASS" if not errors else "FAIL", "run_dir": str(run_dir), "errors": errors, "official_reproduction": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "resume", "verify"), required=True)
    parser.add_argument("--branch", choices=("rf", "svm", "xgboost"))
    parser.add_argument("--config", choices=sorted(CONFIG_EXPECTED_ROWS))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--mi-runs-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--git-commit", default="NOT_CAPTURED")
    parser.add_argument("--confirm-provisional-research", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "verify":
            if args.run_dir is None:
                parser.error("--run-dir is required for --mode verify")
            result = verify_run(args.run_dir)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "PASS" else 1
        if not args.confirm_provisional_research:
            parser.error("full/resume requires --confirm-provisional-research")
        required = ((args.branch, "--branch"), (args.config, "--config"), (args.dataset_root, "--dataset-root"), (args.mi_runs_root, "--mi-runs-root"), (args.runs_root, "--runs-root"), (args.run_id, "--run-id"))
        for value, name in required:
            if value is None:
                parser.error(f"{name} is required for --mode {args.mode}")
        result_dir = run_rfe(
            dataset_root=args.dataset_root,
            mi_runs_root=args.mi_runs_root,
            stage5_runs_root=args.runs_root,
            config=args.config,
            branch=args.branch,
            run_id=args.run_id,
            git_commit=args.git_commit,
            resume=args.mode == "resume",
        )
        print(json.dumps({"status": "PASS", "scientific_status": SCIENTIFIC_STATUS, "run_dir": str(result_dir)}, ensure_ascii=False))
        return 0
    except (Stage5ContractError, Stage4ContractError, FileExistsError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "scientific_status": SCIENTIFIC_STATUS, "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

