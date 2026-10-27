"""Google Colab orchestration for the CICFLOW MI-RFE pipeline.

The module is an orchestration layer only. Scientific implementation remains
in ``stage4_mi`` and ``stage5_rfe``. It never opens Validation/Test data and
does not copy dataset Parquet files during backup.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIGS = ("FW256_FS8", "FW256_FS16", "FW512_FS8", "FW512_FS16")
BRANCHES = ("rf", "svm")
MODES = ("dry_run", "full", "resume", "verify")
REFERENCE_STATUS = "PROVISIONAL_REFERENCE_UNVERIFIED"


class ColabContractError(ValueError):
    """Raised when a Colab path, mode, or scientific gate is invalid."""


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ColabContractError(f"Unsupported MODE: {mode}")
    return mode


def validate_config(config: str) -> str:
    if config not in CONFIGS:
        raise ColabContractError(f"Unsupported configuration: {config}")
    return config


def validate_branch(branch: str) -> str:
    if branch not in BRANCHES:
        raise ColabContractError(f"Unsupported Stage 5A branch: {branch}")
    return branch


def assert_single_dataset_root(dataset_root: Path) -> None:
    """Reject accidental mixing of CICFLOW roots and the CTU/Windows root."""

    normalized = str(dataset_root).replace("\\", "/").lower()
    if "ciciomt24/windows_cplus" in normalized:
        raise ColabContractError("CICIoMT24/WINDOWS_CPLUS cannot be used as the CICFLOW dataset root")
    if dataset_root.name != "CICFLOWMETER_FEATURE_CSV":
        raise ColabContractError(
            "DATASET_ROOT must point to the CICFLOWMETER_FEATURE_CSV root; do not mix dataset roots"
        )


def build_paths(
    *,
    project_root: Path,
    dataset_root: Path,
    drive_root: Path,
) -> dict[str, Path]:
    assert_single_dataset_root(dataset_root)
    return {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "stage4_runs_root": project_root / "code_CICFLOW" / "stage4_mi" / "runs",
        "stage5_runs_root": project_root / "code_CICFLOW" / "stage5_rfe" / "runs",
        "backup_root": drive_root / "COMPACT_MI_RFE_BACKUPS",
        "aggregate_log": project_root / "code_CICFLOW" / "logs" / "ALL_EXECUTION_RESULTS.log",
    }


def check_source_layout(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Metadata/path gate; this does not read Train values."""

    required = {
        "project_root": paths["project_root"],
        "dataset_root": paths["dataset_root"],
        "stage4_runs_root": paths["stage4_runs_root"],
        "stage5_runs_root": paths["stage5_runs_root"],
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    return {
        "status": "PASS" if not missing else "BLOCKED",
        "missing_paths": missing,
        "dataset_root": str(paths["dataset_root"]),
        "mixed_dataset_roots_rejected": True,
        "validation_test_values_read": False,
    }


def stage4_metadata_preflight(paths: Mapping[str, Path], config: str) -> dict[str, Any]:
    """Run the Stage 4 metadata/resource contract without fitting MI."""

    validate_config(config)
    try:
        from stage4_mi.mi_selection import (
            CONFIG_EXPECTED_ROWS,
            collect_resource_snapshot,
            inspect_train_footer,
            load_feature_schema,
            resource_gate,
            resolve_train_path,
        )

        config_dir = paths["dataset_root"] / config
        feature_names = load_feature_schema(config_dir)
        train_path = resolve_train_path(config_dir)
        footer = inspect_train_footer(train_path, feature_names)
        resource_snapshot = collect_resource_snapshot(config_dir)
        resource_result = resource_gate(resource_snapshot, footer["rows"])
        errors: list[str] = []
        if footer["rows"] != CONFIG_EXPECTED_ROWS[config]:
            errors.append("Train row count does not match the Stage 4 contract")
        if resource_result["status"] != "PASS":
            errors.extend(resource_result["reasons"])
        return {
            "status": "PASS" if not errors else "BLOCKED",
            "config": config,
            "feature_count": len(feature_names),
            "train_footer": footer,
            "resource_snapshot": resource_snapshot,
            "resource_gate": resource_result,
            "errors": errors,
            "validation_test_values_read": False,
        }
    except Exception as exc:
        return {"status": "BLOCKED", "config": config, "errors": [str(exc)], "validation_test_values_read": False}


def verify_stage4_run(paths: Mapping[str, Path], config: str, run_id: str) -> dict[str, Any]:
    from stage4_mi.mi_selection import verify_run

    validate_config(config)
    run_dir = paths["stage4_runs_root"] / config / run_id
    return verify_run(run_dir)


def run_stage4_mi(
    paths: Mapping[str, Path],
    config: str,
    run_id: str,
    *,
    mode: str,
    confirm_provisional_research: bool = False,
) -> dict[str, Any]:
    """Run or verify Stage 4 MI through its existing guarded implementation."""

    validate_config(config)
    validate_mode(mode)
    if mode in {"full", "resume"} and not confirm_provisional_research:
        raise ColabContractError("Full/resume MI requires explicit provisional-research confirmation")
    from stage4_mi.mi_selection import run_mi, verify_run

    run_dir = paths["stage4_runs_root"] / config / run_id
    if mode == "verify":
        return verify_run(run_dir)
    result = run_mi(
        dataset_root=paths["dataset_root"],
        config=config,
        runs_root=paths["stage4_runs_root"],
        run_id=run_id,
        resume=mode == "resume",
    )
    return {"status": "PASS", "scientific_status": "PROVISIONAL_RESEARCH", "run_dir": str(result)}


def run_stage5_rfe(
    paths: Mapping[str, Path],
    config: str,
    run_id: str,
    branch: str,
    *,
    mode: str,
    git_commit: str = "NOT_CAPTURED",
    confirm_provisional_research: bool = False,
) -> dict[str, Any]:
    """Run or verify Stage 5A through its guarded implementation."""

    validate_config(config)
    validate_branch(branch)
    validate_mode(mode)
    if mode in {"full", "resume"} and not confirm_provisional_research:
        raise ColabContractError("Full/resume RFE requires explicit provisional-research confirmation")
    from stage5_rfe.rfe_selection import run_rfe, verify_run

    run_dir = paths["stage5_runs_root"] / config / run_id / branch
    if mode == "verify":
        return verify_run(run_dir)
    result = run_rfe(
        dataset_root=paths["dataset_root"],
        mi_runs_root=paths["stage4_runs_root"],
        stage5_runs_root=paths["stage5_runs_root"],
        config=config,
        branch=branch,
        run_id=run_id,
        git_commit=git_commit,
        resume=mode == "resume",
    )
    return {"status": "PASS", "scientific_status": "PROVISIONAL_RESEARCH", "run_dir": str(result)}


def _small_artifact_names() -> tuple[str, ...]:
    return (
        "MI_COMPLETE.json",
        "mi_manifest.json",
        "mi_scores.csv",
        "mi_ranking.json",
        "top100_features.json",
        "resource_usage.json",
        "mi_report.md",
        "COMPLETE.json",
        "rfe_manifest.json",
        "rfe_ranking.json",
        "rfe_ranking.csv",
        "rfe_progress.json",
        "rfe_report.md",
        "BLOCKED.json",
        "FAILED.json",
    )


def backup_small_artifacts(source_dir: Path, backup_root: Path) -> dict[str, Any]:
    """Copy only small metadata/result files; never copy Parquet dataset files."""

    if not source_dir.exists():
        raise ColabContractError(f"Cannot back up missing run directory: {source_dir}")
    target = backup_root / source_dir.name
    if target.exists():
        raise ColabContractError(f"Backup target already exists; refusing overwrite: {target}")
    target.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for filename in _small_artifact_names():
        source = source_dir / filename
        if source.exists():
            destination = target / filename
            shutil.copy2(source, destination)
            copied.append(filename)
    marker = {
        "status": "BACKUP_COMPLETE",
        "source_dir": str(source_dir),
        "copied_files": copied,
        "dataset_files_copied": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    marker_path = target / "BACKUP_COMPLETE.json"
    with marker_path.open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return marker


def summarize_configs(
    paths: Mapping[str, Path],
    run_ids: Mapping[str, str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for config in CONFIGS:
        run_id = run_ids[config]
        stage4_dir = paths["stage4_runs_root"] / config / run_id
        row: dict[str, Any] = {
            "config": config,
            "run_id": run_id,
            "stage4_complete": (stage4_dir / "MI_COMPLETE.json").exists(),
        }
        for branch in BRANCHES:
            stage5_dir = paths["stage5_runs_root"] / config / run_id / branch
            row[f"stage5_{branch}_complete"] = (stage5_dir / "COMPLETE.json").exists()
        summary.append(row)
    return summary

