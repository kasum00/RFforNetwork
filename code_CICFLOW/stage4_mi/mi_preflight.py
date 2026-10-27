"""Metadata-only and resource-only gates for Stage 4 MI.

This module deliberately does not load any Train feature/label values.  It
reads schema JSON and the Train Parquet footer only.  Validation and Test
paths are not inspected by this gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .mi_selection import (
    CONFIG_EXPECTED_ROWS,
    Stage4ContractError,
    collect_resource_snapshot,
    inspect_train_footer,
    load_feature_schema,
    resource_gate,
    resolve_config_dir,
    resolve_train_path,
    source_identity,
    validate_run_id,
)


def metadata_preflight(
    *, dataset_root: Path, config: str, runs_root: Path | None = None, run_id: str | None = None
) -> dict[str, Any]:
    config_dir = resolve_config_dir(dataset_root, config)
    if run_id is not None:
        run_id = validate_run_id(run_id)
    feature_names = load_feature_schema(config_dir)
    train_path = resolve_train_path(config_dir)
    footer = inspect_train_footer(train_path, feature_names)
    errors: list[str] = []
    if footer["rows"] != CONFIG_EXPECTED_ROWS[config]:
        errors.append(
            f"Train row count {footer['rows']} != expected {CONFIG_EXPECTED_ROWS[config]}"
        )
    run_dir = None
    if runs_root is not None and run_id is not None:
        run_dir = runs_root / config / run_id
        if run_dir.exists():
            errors.append(f"run directory already exists: {run_dir}")
    identity = source_identity(config, config_dir, feature_names, footer)
    return {
        "status": "PASS" if not errors else "FAIL",
        "gate": "METADATA_ONLY",
        "config": config,
        "config_dir": str(config_dir),
        "train_footer": footer,
        "feature_count": len(feature_names),
        "label_column": "Window_Label",
        "source_identity": identity,
        "validation_test_values_read": False,
        "errors": errors,
    }


def resource_preflight(*, dataset_root: Path, config: str) -> dict[str, Any]:
    config_dir = resolve_config_dir(dataset_root, config)
    feature_names = load_feature_schema(config_dir)
    footer = inspect_train_footer(resolve_train_path(config_dir), feature_names)
    snapshot = collect_resource_snapshot(config_dir)
    gate = resource_gate(snapshot, footer["rows"])
    return {
        "status": gate["status"],
        "gate": "RESOURCE_ONLY",
        "config": config,
        "train_rows": footer["rows"],
        "resource_snapshot": snapshot,
        "resource_gate": gate,
        "train_values_read": False,
        "validation_test_values_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "resource"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", choices=sorted(CONFIG_EXPECTED_ROWS), required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "metadata":
            result = metadata_preflight(
                dataset_root=args.dataset_root,
                config=args.config,
                runs_root=args.runs_root,
                run_id=args.run_id,
            )
        else:
            result = resource_preflight(dataset_root=args.dataset_root, config=args.config)
    except (Stage4ContractError, OSError, ValueError) as exc:
        result = {"status": "FAIL", "mode": args.mode, "message": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

