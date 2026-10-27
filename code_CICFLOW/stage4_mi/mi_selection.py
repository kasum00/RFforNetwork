"""Deterministic, Train-only Mutual Information Top-100 selection.

This module deliberately keeps Stage 4 separate from the existing Stage 0–3
artifacts.  It reads only the Train Parquet values after a metadata/resource
gate has passed.  Validation and Test files are never opened by the full-run
path.

The current implementation is explicitly provisional: the repository's
authoritative CTU-13 MI implementation has not yet been reconciled with this
branch.  Therefore generated outputs cannot be labelled as an official
COMPACT-MI-RFE reproduction until that reconciliation is completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_FEATURE_COUNT = 380
LABEL_COLUMN = "Window_Label"
TOP_K = 100
MI_RANDOM_STATE = 42
MI_N_NEIGHBORS = 3
MI_REFERENCE_STATUS = "PROVISIONAL_REFERENCE_UNVERIFIED"
TRAIN_PARQUET_CANDIDATES = (
    "train_windows.parquet",
    "train.parquet",
    "train_features.parquet",
)

CONFIG_EXPECTED_ROWS = {
    "FW256_FS8": 326157,
    "FW256_FS16": 163203,
    "FW512_FS8": 311423,
    "FW512_FS16": 155814,
}

PROVENANCE_TOKENS = (
    "source_file",
    "source_capture",
    "source_session",
    "capture_id",
    "session_id",
    "window_id",
    "segment_id",
    "start_time",
    "end_time",
    "timestamp",
    "row_id",
    "provenance",
)


class Stage4ContractError(ValueError):
    """Raised when the Stage 4 contract is not satisfied."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Stage4ContractError(f"Expected JSON object: {path}")
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_exclusive(path: Path, payload: str) -> None:
    """Create a file once; never silently overwrite an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    write_exclusive(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
    )


def write_json_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically refresh a mutable checkpoint inside an exclusive run dir."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _extract_schema_names(value: Any) -> list[str]:
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return list(value)
        if all(isinstance(item, Mapping) and "name" in item for item in value):
            return [str(item["name"]) for item in value]
    if isinstance(value, Mapping):
        for key in ("feature_names", "features", "columns", "schema", "fields"):
            if key in value:
                names = _extract_schema_names(value[key])
                if names:
                    return names
    return []


def validate_feature_names(feature_names: Sequence[str]) -> list[str]:
    names = [str(name) for name in feature_names]
    if len(names) != EXPECTED_FEATURE_COUNT:
        raise Stage4ContractError(
            f"Expected exactly {EXPECTED_FEATURE_COUNT} features; got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise Stage4ContractError("Feature schema contains duplicate names")
    if any(
        any(token in name.lower() for token in PROVENANCE_TOKENS)
        for name in names
    ):
        raise Stage4ContractError("Feature schema contains a provenance-like field")
    return names


def load_feature_schema(
    config_dir: Path,
    parquet_reader: Callable[[Path], Any] | None = None,
) -> list[str]:
    """Load the ordered 380-feature schema using metadata only.

    Stage 3 configurations in this workspace may not carry a sidecar schema
    JSON.  In that case the Train Parquet footer is the bound schema source;
    it is still metadata-only and is subsequently checked again by
    :func:`inspect_train_footer`.
    """

    candidates: list[tuple[Path, list[str]]] = []
    for filename in ("feature_schema.json", "window_feature_schema.json"):
        path = config_dir / filename
        if not path.exists():
            continue
        names = _extract_schema_names(read_json(path))
        if names:
            candidates.append((path, names))

    if not candidates:
        train_path = resolve_train_path(config_dir)
        if not train_path.exists():
            raise Stage4ContractError(
                f"No feature schema JSON and no Train Parquet found in {config_dir}"
            )
        reader = parquet_reader or _parquet_file
        footer_schema = list(reader(train_path).schema_arrow.names)
        if len(footer_schema) != EXPECTED_FEATURE_COUNT + 1 or footer_schema[-1] != LABEL_COLUMN:
            raise Stage4ContractError(
                "Train Parquet footer must contain 380 features followed by Window_Label"
            )
        return validate_feature_names(footer_schema[:-1])

    full_schema = next((names for _, names in candidates if LABEL_COLUMN in names), None)
    feature_schema = next(
        (names for _, names in candidates if len(names) == EXPECTED_FEATURE_COUNT), None
    )
    if full_schema is not None:
        if len(full_schema) != EXPECTED_FEATURE_COUNT + 1:
            raise Stage4ContractError("Window schema must contain 380 features plus Window_Label")
        if full_schema[-1] != LABEL_COLUMN:
            raise Stage4ContractError("Window_Label must be the final schema column")
        feature_schema = full_schema[:-1]
    if feature_schema is None:
        raise Stage4ContractError("Could not resolve an ordered 380-feature schema")
    return validate_feature_names(feature_schema)


def expected_full_schema(feature_names: Sequence[str]) -> list[str]:
    return validate_feature_names(feature_names) + [LABEL_COLUMN]


def _parquet_file(path: Path) -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Stage4ContractError("pyarrow is required for Parquet metadata/value access") from exc
    return parquet.ParquetFile(path)


def inspect_train_footer(
    train_path: Path,
    feature_names: Sequence[str],
    parquet_reader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Inspect only the Train Parquet footer and schema."""

    if not train_path.exists():
        raise Stage4ContractError(f"Train Parquet is missing: {train_path}")
    reader = parquet_reader or _parquet_file
    parquet_file = reader(train_path)
    schema = list(parquet_file.schema_arrow.names)
    expected = expected_full_schema(feature_names)
    if schema != expected:
        raise Stage4ContractError("Train Parquet schema/order does not match the official schema")
    metadata = parquet_file.metadata
    label_statistics: list[dict[str, Any]] = []
    label_index = schema.index(LABEL_COLUMN)
    for row_group_index in range(metadata.num_row_groups):
        column = metadata.row_group(row_group_index).column(label_index)
        statistics = column.statistics
        label_statistics.append(
            {
                "row_group": row_group_index,
                "has_statistics": statistics is not None,
                "min": None if statistics is None else statistics.min,
                "max": None if statistics is None else statistics.max,
            }
        )
    return {
        "path": str(train_path),
        "rows": int(metadata.num_rows),
        "columns": len(schema),
        "schema_digest": stable_digest(schema),
        "label_statistics": label_statistics,
        "file_size": train_path.stat().st_size,
        "mtime_ns": train_path.stat().st_mtime_ns,
    }


def resolve_config_dir(dataset_root: Path, config: str) -> Path:
    if config not in CONFIG_EXPECTED_ROWS:
        raise Stage4ContractError(f"Unsupported Stage 4 configuration: {config}")
    return dataset_root / config


def resolve_train_path(config_dir: Path) -> Path:
    """Resolve only known Train filenames; never search Validation/Test files."""

    for filename in TRAIN_PARQUET_CANDIDATES:
        path = config_dir / filename
        if path.exists():
            return path
    names = ", ".join(TRAIN_PARQUET_CANDIDATES)
    raise Stage4ContractError(
        f"No supported Train Parquet found in {config_dir}; expected one of: {names}"
    )


def validate_run_id(run_id: str) -> str:
    value = str(run_id)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise Stage4ContractError("run-id must be a non-empty directory name without path separators")
    return value


def source_identity(
    config: str,
    config_dir: Path,
    feature_names: Sequence[str],
    footer: Mapping[str, Any],
) -> dict[str, Any]:
    binding_files: list[dict[str, Any]] = []
    for filename in (
        "feature_schema.json",
        "window_feature_schema.json",
        "stage3_integrity_report.json",
        "integrity_report.json",
    ):
        path = config_dir / filename
        if path.exists():
            binding_files.append(
                {
                    "name": filename,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "content_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    identity = {
        "config": config,
        "config_dir": str(config_dir),
        "train_footer": dict(footer),
        "feature_names_digest": stable_digest(list(feature_names)),
        "binding_files": binding_files,
    }
    identity["identity_digest"] = stable_digest(identity)
    return identity


def collect_resource_snapshot(config_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_dir": str(config_dir),
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
    usage = shutil.disk_usage(config_dir)
    snapshot["disk"] = {
        "path": str(config_dir),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
    }
    return snapshot


def estimate_required_ram(rows: int, feature_count: int = EXPECTED_FEATURE_COUNT) -> int:
    # Matrix float32 + MI working memory + conversion/Parquet overhead.
    matrix_bytes = int(rows) * int(feature_count) * 4
    labels_bytes = int(rows) * 8
    return int((matrix_bytes + labels_bytes) * 3.0)


def resource_gate(snapshot: Mapping[str, Any], rows: int) -> dict[str, Any]:
    required = estimate_required_ram(rows)
    memory = snapshot.get("memory", {})
    available = memory.get("available_bytes")
    free_disk = snapshot.get("disk", {}).get("free_bytes")
    reasons: list[str] = []
    if not isinstance(available, int):
        reasons.append("available RAM could not be measured")
    elif available < required:
        reasons.append(f"available RAM {available} is below estimate {required}")
    if not isinstance(free_disk, int):
        reasons.append("free disk space could not be measured")
    elif free_disk < 2 * 1024**3:
        reasons.append("less than 2 GiB free disk space")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "required_ram_estimate_bytes": required,
        "available_ram_bytes": available,
        "free_disk_bytes": free_disk,
        "reasons": reasons,
        "no_sampling_or_retry": True,
    }


def preprocess_matrix(
    X: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the contract preprocessing without dropping rows or changing labels."""

    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Stage4ContractError("Train features must be numeric") from exc
    labels = np.asarray(y)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise Stage4ContractError("Train matrix must have exactly 380 columns")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise Stage4ContractError("Labels must be one-dimensional and row-aligned")
    if np.issubdtype(labels.dtype, np.number):
        if not np.isfinite(labels.astype(np.float64)).all():
            raise Stage4ContractError("Labels contain NaN or Inf")
    unique = np.unique(labels)
    if unique.size != 2:
        raise Stage4ContractError("Window_Label must contain exactly two classes")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return matrix, labels.copy()


def read_train_values(
    train_path: Path,
    feature_names: Sequence[str],
    table_reader: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only Train values; this function has no Validation/Test path input."""

    columns = list(feature_names) + [LABEL_COLUMN]
    if table_reader is None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise Stage4ContractError("pyarrow is required for Train value access") from exc
        table = parquet.read_table(train_path, columns=columns)
    else:
        table = table_reader(train_path, columns=columns)
    arrays = [table.column(name).to_numpy(zero_copy_only=False) for name in feature_names]
    X = np.column_stack(arrays)
    y = table.column(LABEL_COLUMN).to_numpy(zero_copy_only=False)
    return preprocess_matrix(X, y)


def compute_mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Iterable[int] | None = None,
    random_state: int = MI_RANDOM_STATE,
    n_neighbors: int = MI_N_NEIGHBORS,
) -> dict[int, float]:
    """Compute deterministic per-feature MI scores for the requested indices."""

    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Stage4ContractError("scikit-learn is required for MI selection") from exc
    indices = list(range(X.shape[1])) if feature_indices is None else list(feature_indices)
    scores: dict[int, float] = {}
    for index in indices:
        value = mutual_info_classif(
            X[:, [index]],
            y,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state + index,
        )[0]
        scores[index] = float(value)
    return scores


def rank_scores(scores: Mapping[int, float], feature_names: Sequence[str]) -> list[dict[str, Any]]:
    validate_feature_names(feature_names)
    if set(scores) != set(range(EXPECTED_FEATURE_COUNT)):
        raise Stage4ContractError("MI ranking must contain one score for every feature")
    if not all(np.isfinite(float(value)) for value in scores.values()):
        raise Stage4ContractError("MI ranking contains a non-finite score")
    ordered = sorted(scores, key=lambda index: (-float(scores[index]), int(index)))
    return [
        {
            "rank": rank,
            "feature_index": int(index),
            "feature_name": feature_names[index],
            "mi_score": float(scores[index]),
        }
        for rank, index in enumerate(ordered, start=1)
    ]


def _load_table_values_for_tests(table: Any, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [table.column(name).to_numpy(zero_copy_only=False) for name in feature_names]
    return preprocess_matrix(np.column_stack(arrays), table.column(LABEL_COLUMN).to_numpy(zero_copy_only=False))


def _run_parameters() -> dict[str, Any]:
    return {
        "algorithm": "sklearn.feature_selection.mutual_info_classif",
        "discrete_features": False,
        "n_neighbors": MI_N_NEIGHBORS,
        "random_state_base": MI_RANDOM_STATE,
        "score_seed_rule": "random_state_base + original_feature_index",
        "top_k": TOP_K,
        "preprocessing": {
            "numeric": True,
            "replace_inf_with_nan_then_fill_zero": True,
            "dtype": "float32",
            "row_drop": False,
            "label_change": False,
            "scaler": "none",
        },
        "reference_status": MI_REFERENCE_STATUS,
    }


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {name: run_dir / name for name in (
        "RUN_INCOMPLETE.json",
        "FAILED.json",
        "mi_progress.json",
        "mi_scores.csv",
        "mi_ranking.json",
        "top100_features.json",
        "mi_manifest.json",
        "resource_usage.json",
        "mi_report.md",
        "MI_COMPLETE.json",
    )}


def _write_scores_csv(path: Path, ranking: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_index", "feature_name", "mi_score"])
        writer.writeheader()
        writer.writerows(ranking)


def _load_progress(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "mi_progress.json"
    if not path.exists():
        return {"completed_indices": [], "scores": {}}
    return read_json(path)


def _validate_resume(
    run_dir: Path,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> None:
    paths = _run_paths(run_dir)
    if not paths["RUN_INCOMPLETE.json"].exists():
        raise Stage4ContractError("Resume requires RUN_INCOMPLETE.json")
    incomplete = read_json(paths["RUN_INCOMPLETE.json"])
    if incomplete.get("source_identity", {}).get("identity_digest") != identity.get("identity_digest"):
        raise Stage4ContractError("Resume source identity does not match")
    if incomplete.get("parameters") != parameters:
        raise Stage4ContractError("Resume scientific parameters do not match")
    manifest_path = paths["mi_manifest.json"]
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("source_identity", {}).get("identity_digest") != identity.get("identity_digest"):
            raise Stage4ContractError("Resume source identity does not match")
        if manifest.get("parameters") != parameters:
            raise Stage4ContractError("Resume scientific parameters do not match")
    progress = _load_progress(run_dir)
    completed = {int(index) for index in progress.get("completed_indices", [])}
    if not completed.issubset(set(range(EXPECTED_FEATURE_COUNT))):
        raise Stage4ContractError("Resume checkpoint contains an invalid feature index")


def _render_report(
    config: str,
    identity: Mapping[str, Any],
    ranking: Sequence[Mapping[str, Any]],
    resource_result: Mapping[str, Any],
) -> str:
    lines = [
        f"# Stage 4 MI run — {config}",
        "",
        f"- Scientific status: `PROVISIONAL_RESEARCH` ({MI_REFERENCE_STATUS})",
        "- Values read: Train only; Validation/Test values were not opened.",
        f"- Train rows: `{identity['train_footer']['rows']}`",
        f"- Feature count: `{EXPECTED_FEATURE_COUNT}`; Top-K: `{TOP_K}`",
        f"- Source identity: `{identity['identity_digest']}`",
        f"- Resource gate: `{resource_result['status']}`",
        "",
        "## Top 10 preview",
        "",
        "| Rank | Feature | MI score |",
        "|---:|---|---:|",
    ]
    lines.extend(
        f"| {row['rank']} | `{row['feature_name']}` | {row['mi_score']:.12g} |"
        for row in ranking[:10]
    )
    lines.extend(
        [
            "",
            "This run is not an official reproduction until the CTU-13 reference implementation and parameters are reconciled.",
            "",
        ]
    )
    return "\n".join(lines)


def run_mi(
    *,
    dataset_root: Path,
    config: str,
    runs_root: Path,
    run_id: str,
    resume: bool = False,
    batch_size: int = 20,
) -> Path:
    """Run or resume one configuration after metadata and resource gates."""

    run_id = validate_run_id(run_id)
    config_dir = resolve_config_dir(dataset_root, config)
    feature_names = load_feature_schema(config_dir)
    footer = inspect_train_footer(resolve_train_path(config_dir), feature_names)
    expected_rows = CONFIG_EXPECTED_ROWS[config]
    if footer["rows"] != expected_rows:
        raise Stage4ContractError(
            f"Unexpected Train row count for {config}: {footer['rows']} != {expected_rows}"
        )
    identity = source_identity(config, config_dir, feature_names, footer)
    parameters = _run_parameters()
    resource_snapshot = collect_resource_snapshot(config_dir)
    resource_result = resource_gate(resource_snapshot, footer["rows"])
    if resource_result["status"] != "PASS":
        raise Stage4ContractError("Resource gate blocked MI; no Train values were read")

    run_dir = runs_root / config / run_id
    if resume:
        if not run_dir.is_dir():
            raise Stage4ContractError(f"Run directory does not exist: {run_dir}")
        _validate_resume(run_dir, identity, parameters)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        paths = _run_paths(run_dir)
        write_json_exclusive(
            paths["RUN_INCOMPLETE.json"],
            {
                "status": "INCOMPLETE",
                "config": config,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_identity": identity,
                "parameters": parameters,
            },
        )
        write_json_exclusive(paths["resource_usage.json"], {"snapshot": resource_snapshot, "gate": resource_result})
        write_json_checkpoint(
            paths["mi_progress.json"],
            {"completed_indices": [], "scores": {}, "source_identity_digest": identity["identity_digest"]},
        )

    paths = _run_paths(run_dir)
    X, y = read_train_values(resolve_train_path(config_dir), feature_names)
    progress = _load_progress(run_dir)
    scores = {int(key): float(value) for key, value in progress.get("scores", {}).items()}
    completed = {int(index) for index in progress.get("completed_indices", [])}
    remaining = [index for index in range(EXPECTED_FEATURE_COUNT) if index not in completed]
    try:
        for start in range(0, len(remaining), max(1, int(batch_size))):
            batch = remaining[start : start + max(1, int(batch_size))]
            scores.update(compute_mi_scores(X, y, batch))
            completed.update(batch)
            write_json_checkpoint(
                paths["mi_progress.json"],
                {
                    "completed_indices": sorted(completed),
                    "scores": {str(key): scores[key] for key in sorted(scores)},
                    "source_identity_digest": identity["identity_digest"],
                },
            )
    except Exception as exc:
        if not paths["FAILED.json"].exists():
            write_json_exclusive(paths["FAILED.json"], {"error": repr(exc), "completed": len(completed)})
        raise

    ranking = rank_scores(scores, feature_names)
    if len(ranking) != EXPECTED_FEATURE_COUNT or len(ranking[:TOP_K]) != TOP_K:
        raise Stage4ContractError("MI ranking did not produce exactly 100 Top-K features")
    if not paths["mi_scores.csv"].exists():
        _write_scores_csv(paths["mi_scores.csv"], ranking)
    if not paths["mi_ranking.json"].exists():
        write_json_exclusive(paths["mi_ranking.json"], {"ranking": ranking})
    if not paths["top100_features.json"].exists():
        write_json_exclusive(paths["top100_features.json"], {"top_k": TOP_K, "features": ranking[:TOP_K]})
    if not paths["mi_manifest.json"].exists():
        write_json_exclusive(
            paths["mi_manifest.json"],
            {
                "status": "COMPLETE",
                "config": config,
                "source_identity": identity,
                "parameters": parameters,
                "train_rows_observed": int(X.shape[0]),
                "train_feature_count": int(X.shape[1]),
                "label_column": LABEL_COLUMN,
                "official_reproduction": False,
            },
        )
    if not paths["mi_report.md"].exists():
        write_exclusive(paths["mi_report.md"], _render_report(config, identity, ranking, resource_result))
    if not paths["MI_COMPLETE.json"].exists():
        write_json_exclusive(
            paths["MI_COMPLETE.json"],
            {
                "status": "COMPLETE",
                "config": config,
                "ranking_digest": stable_digest(ranking),
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    return run_dir


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Verify a completed run using result metadata only; never fit or read data."""

    paths = _run_paths(run_dir)
    required = [
        paths["MI_COMPLETE.json"],
        paths["mi_manifest.json"],
        paths["mi_ranking.json"],
        paths["top100_features.json"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"status": "FAIL", "run_dir": str(run_dir), "missing": missing}
    manifest = read_json(paths["mi_manifest.json"])
    ranking_payload = read_json(paths["mi_ranking.json"])
    top_payload = read_json(paths["top100_features.json"])
    ranking = ranking_payload.get("ranking", [])
    top = top_payload.get("features", [])
    errors: list[str] = []
    if manifest.get("official_reproduction") is not False:
        errors.append("official_reproduction must remain false")
    if len(ranking) != EXPECTED_FEATURE_COUNT:
        errors.append("ranking does not contain 380 features")
    if len(top) != TOP_K:
        errors.append("Top-K output does not contain 100 features")
    return {
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir),
        "errors": errors,
        "official_reproduction": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "resume", "verify"), required=True)
    parser.add_argument("--config", choices=sorted(CONFIG_EXPECTED_ROWS))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--confirm-provisional-research", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
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
        for value, name in ((args.config, "--config"), (args.dataset_root, "--dataset-root"), (args.runs_root, "--runs-root")):
            if value is None:
                parser.error(f"{name} is required for --mode {args.mode}")
        run_id = args.run_id or time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime())
        result_dir = run_mi(
            dataset_root=args.dataset_root,
            config=args.config,
            runs_root=args.runs_root,
            run_id=run_id,
            resume=args.mode == "resume",
            batch_size=args.batch_size,
        )
        print(json.dumps({"status": "PASS", "run_dir": str(result_dir)}, ensure_ascii=False))
        return 0
    except (Stage4ContractError, FileExistsError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

