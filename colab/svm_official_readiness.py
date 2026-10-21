"""Stage 7.3A read-only SVM readiness review. This module cannot launch training.

The existing full-training gate in experiment_launcher remains closed. A Stage 7.3B
implementation and explicit per-run human authorization are required separately.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .dataset_validator import sha256
from .experiment_launcher import RunSpec, preflight

SVM_EXPERIMENTS = ("baseline_svm", "mi_rfe_svm")
IDENTITY_FIELDS = (
    "experiment", "config", "run_name", "git_commit", "training_code_sha256",
    "sliding_code_sha256", "manifest_sha256", "validation_sha256",
    "window_summary_sha256", "dataset_fingerprint_sha256",
)


def normalized_source_sha256(path: Path) -> str:
    """Compare Windows CRLF and Colab LF source bytes without masking other edits."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_contract(repo: Path) -> dict[str, bool]:
    try:
        training = (repo / "huyen_code" / "ciciomt24_training.py").read_text(encoding="utf-8")
        sliding = (repo / "huyen_code" / "ciciomt24_sliding_window.py").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {"relative_training_root": False, "windows_path_import": False,
                "two_windows_stem_sites": False, "baseline_and_compact_separate": False}
    return {
        "relative_training_root": "ROOT = Path(__file__).resolve().parents[1]" in training,
        "windows_path_import": "from pathlib import Path, PureWindowsPath" in sliding,
        "two_windows_stem_sites": sliding.count("PureWindowsPath(segment.output_relative_path).stem") == 2,
        "baseline_and_compact_separate": "BASELINE_SVM_PARAMS" in training and "COMPACT_SVM_PARAMS" in training,
    }


def git_checkout(repo: Path) -> dict:
    """Read-only; inaccessible Git or a dirty checkout is not proof of consistency."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=15, check=True).stdout
        return {"commit": head, "clean": not status.strip(),
                "dirty_paths_preview": status.splitlines()[:20]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "clean": False, "error": str(exc)}


def identity_matches(expected: dict | None, actual: dict) -> bool:
    """No missing field or cross-run approval may match by accident."""
    return bool(expected) and all(
        isinstance(expected.get(key), str) and bool(expected[key])
        and expected[key] == actual.get(key) for key in IDENTITY_FIELDS
    )


def approval_matches(approval: dict | None, actual: dict) -> bool:
    """Validate binding only; a matching approval does NOT unlock Stage 7.3A."""
    return bool(approval) and approval.get("decision") == "APPROVED_STAGE73B" and all((
        identity_matches(approval.get("identity"), actual),
        type(approval.get("max_elapsed_seconds")) is int
        and approval["max_elapsed_seconds"] > 0,
        type(approval.get("max_tree_rss_bytes")) is int
        and approval["max_tree_rss_bytes"] > 0,
    ))


def review(repo: Path, windows: Path, local_runs: Path, drive_root: Path,
           spec: RunSpec, *, unit_test_evidence: dict | None = None,
           svm_probe_path: Path | None = None,
           expected_identity: dict | None = None,
           approval: dict | None = None) -> dict:
    """Inspect metadata and headers, never Test rows, fit, create a run, or grant training.

    The Stage 6 preflight is deliberately reused; its SVM full-resource gate is
    always false at Stage 7.3A. File-hash errors are reported as FAIL, not waived.
    """
    spec.validate()
    if spec.experiment not in SVM_EXPERIMENTS:
        raise ValueError("Stage 7.3A review accepts SVM experiments only")
    contract = source_contract(repo)
    checkout = git_checkout(repo)
    base = preflight(repo, windows, local_runs, drive_root, spec,
                     unit_test_evidence=unit_test_evidence, svm_probe_path=svm_probe_path)
    try:
        from .dataset_validator import manifest_root_for
        manifest_root = manifest_root_for(windows, repo)
        identity = {
            "experiment": spec.experiment, "config": spec.config,
            "run_name": spec.run_name, "git_commit": checkout["commit"],
            "training_code_sha256": normalized_source_sha256(repo / "huyen_code" / "ciciomt24_training.py"),
            "sliding_code_sha256": normalized_source_sha256(repo / "huyen_code" / "ciciomt24_sliding_window.py"),
            "manifest_sha256": sha256(manifest_root / "split_manifest.csv"),
            "validation_sha256": sha256(manifest_root / "validation.json"),
            "window_summary_sha256": sha256(windows / "window_run_summary.json"),
            "dataset_fingerprint_sha256": (base.get("dataset") or {}).get("dataset_fingerprint_sha256"),
        }
        identity_error = None
    except (OSError, KeyError, ValueError) as exc:
        identity, identity_error = {}, str(exc)
    checks = {
        **base.get("checks", {}),
        "cross_platform_source_contract": all(contract.values()),
        "clean_git_checkout": checkout["clean"],
        "source_and_dataset_identity_approved": identity_matches(expected_identity, identity),
        "per_run_budget_approval_bound": approval_matches(approval, identity),
        "stage73b_implementation_and_authorization": False,
    }
    return {
        "status": "BLOCKED_STAGE73A", "full_training_authorized": False,
        "checks": checks, "source_contract": contract, "git": checkout,
        "identity": identity, "identity_error": identity_error,
        "base_preflight_status": base.get("status"),
        "ram_guard": base.get("ram_guard"),
        "local_disk_required_bytes": base.get("local_disk_required_bytes"),
        "runtime": base.get("runtime"),
        "dataset": base.get("dataset"),
        "test_feature_rows_read": 0, "model_fit_calls": 0,
        "technical_probe_is_full_authorization": False,
        "note": "Review only. Do not invoke Cell 10 or run_one for SVM in Stage 7.3A.",
    }
