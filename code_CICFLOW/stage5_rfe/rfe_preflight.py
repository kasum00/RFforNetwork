"""Metadata-only and resource-only gates for Stage 5A RF-RFE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .rfe_selection import (
    CONFIG_EXPECTED_ROWS,
    Stage5ContractError,
    collect_resource_snapshot,
    load_stage4_binding,
    resource_gate,
    validate_branch,
    validate_run_id,
)


def metadata_preflight(
    *,
    dataset_root: Path,
    mi_runs_root: Path,
    stage5_runs_root: Path,
    config: str,
    branch: str,
    run_id: str,
) -> dict[str, Any]:
    branch = validate_branch(branch)
    run_id = validate_run_id(run_id)
    binding = load_stage4_binding(
        dataset_root=dataset_root,
        config=config,
        mi_run_dir=mi_runs_root / config / run_id,
    )
    prospective_run = stage5_runs_root / config / run_id / branch
    errors: list[str] = []
    if prospective_run.exists():
        errors.append(f"Stage 5 run directory already exists: {prospective_run}")
    if binding["train_footer"]["rows"] != CONFIG_EXPECTED_ROWS[config]:
        errors.append("Train row count does not match the configuration contract")
    return {
        "status": "PASS" if not errors else "FAIL",
        "gate": "METADATA_ONLY",
        "config": config,
        "branch": branch,
        "run_id": run_id,
        "stage4_mi_run_id": binding["mi_run_id"],
        "stage4_mi_identity_digest": binding["source_identity"]["identity_digest"],
        "dataset_fingerprint": binding["source_identity"]["identity_digest"],
        "train_footer": binding["train_footer"],
        "selected_feature_count": len(binding["top_features"]),
        "validation_test_values_read": False,
        "errors": errors,
    }


def resource_preflight(
    *,
    dataset_root: Path,
    mi_runs_root: Path,
    stage5_runs_root: Path,
    config: str,
    branch: str,
    run_id: str,
) -> dict[str, Any]:
    branch = validate_branch(branch)
    run_id = validate_run_id(run_id)
    binding = load_stage4_binding(
        dataset_root=dataset_root,
        config=config,
        mi_run_dir=mi_runs_root / config / run_id,
    )
    run_dir = stage5_runs_root / config / run_id / branch
    snapshot = collect_resource_snapshot(Path(binding["config_dir"]), run_dir)
    gate = resource_gate(snapshot, binding["train_footer"]["rows"])
    return {
        "status": gate["status"],
        "gate": "RESOURCE_ONLY",
        "config": config,
        "branch": branch,
        "run_id": run_id,
        "train_rows": binding["train_footer"]["rows"],
        "resource_snapshot": snapshot,
        "resource_gate": gate,
        "train_values_read": False,
        "validation_test_values_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "resource"), required=True)
    parser.add_argument("--branch", choices=("rf", "svm", "xgboost"), required=True)
    parser.add_argument("--config", choices=sorted(CONFIG_EXPECTED_ROWS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mi-runs-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "metadata":
            result = metadata_preflight(
                dataset_root=args.dataset_root,
                mi_runs_root=args.mi_runs_root,
                stage5_runs_root=args.runs_root,
                config=args.config,
                branch=args.branch,
                run_id=args.run_id,
            )
        else:
            result = resource_preflight(
                dataset_root=args.dataset_root,
                mi_runs_root=args.mi_runs_root,
                stage5_runs_root=args.runs_root,
                config=args.config,
                branch=args.branch,
                run_id=args.run_id,
            )
    except (Stage5ContractError, OSError, ValueError) as exc:
        result = {"status": "BLOCKED", "mode": args.mode, "message": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

