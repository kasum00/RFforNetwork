"""Explicit opt-in extension to Stage 7.3B; legacy approvals cannot resume."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_launcher import RunSpec, _training
from .svm_run_authorization import (
    authorize_run, collect_identity, invalid_approval_template,
    validate_context_for_training, normalized_source_sha256,
)
from .svm_resumable_store import digest, read_json

POLICY_SCHEMA = "stage73b.mi_rfe_svm.resume-policy.v1"
REQUIRED_MODULES = (
    "colab/svm_resumable_store.py", "colab/svm_resumable_authorization.py",
    "colab/svm_resumable_training.py", "colab/svm_resumable_execution.py",
)
REQUIRED_TESTS = (
    "colab.test_svm_resumable_store", "colab.test_svm_resumable_training",
    "colab.test_svm_resumable_execution",
)
_SEAL = object()


def run_root(drive_root: Path, spec: RunSpec) -> Path:
    spec.validate()
    if spec.experiment != "mi_rfe_svm":
        raise PermissionError("Resumable execution is MI-RFE SVM only")
    return drive_root / "svm_resumable" / spec.experiment / spec.config / spec.run_name


def extension_hashes(repo: Path) -> dict:
    return {name: normalized_source_sha256(repo / name) for name in REQUIRED_MODULES}


def blocked_template(repo: Path, windows: Path, spec: RunSpec) -> dict:
    identity, _ = collect_identity(repo, windows, spec)
    template = invalid_approval_template(identity)
    template["execution_mode"] = POLICY_SCHEMA
    template["resume_policy"] = {
        "schema": POLICY_SCHEMA, "allow_completed_stage_resume": False,
        "repeat_only_uncommitted_work": False,
        "never_repeat_claimed_test": False,
        "session_id": "", "session_budget_seconds": 36000,
        "stop_starting_new_work_margin_seconds": 300,
        "heartbeat_interval_seconds": 60,
        "resumable_module_hashes": extension_hashes(repo),
    }
    return template


@dataclass(frozen=True)
class ResumeAuthorization:
    base: object
    policy: dict
    binding: dict
    root: Path
    resume: bool
    _seal: object
    deadline_epoch: float | None = None


def authorize_resume(repo: Path, windows: Path, local_runs: Path, drive_root: Path,
                     spec: RunSpec, approval_path: Path, technical_evidence: Path,
                     confirmation: str, *, unit_test_evidence: dict,
                     session_id: str, resume: bool, session_budget_seconds: int,
                     heartbeat_interval_seconds: int = 60) -> ResumeAuthorization:
    spec.validate()
    if spec.experiment != "mi_rfe_svm":
        raise PermissionError("This extension must never launch Baseline SVM or RF")
    approval = read_json(approval_path)
    policy = approval.get("resume_policy")
    if approval.get("execution_mode") != POLICY_SCHEMA or not isinstance(policy, dict):
        raise PermissionError("Legacy approval cannot authorize resumable execution")
    if policy.get("schema") != POLICY_SCHEMA or not all(policy.get(name) is True for name in (
        "allow_completed_stage_resume", "repeat_only_uncommitted_work",
        "never_repeat_claimed_test",
    )):
        raise PermissionError("Explicit completed-stage resume/restart policy is required")
    if not session_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                             for ch in session_id) or policy.get("session_id") != session_id:
        raise PermissionError("Approval is bound to a different or invalid session ID")
    for name in ("session_budget_seconds", "stop_starting_new_work_margin_seconds",
                 "heartbeat_interval_seconds"):
        if type(policy.get(name)) is not int or policy[name] <= 0:
            raise PermissionError(f"Explicit positive session budget required: {name}")
    if (policy["session_budget_seconds"] != session_budget_seconds
            or policy["heartbeat_interval_seconds"] != heartbeat_interval_seconds
            or policy["stop_starting_new_work_margin_seconds"] >= session_budget_seconds):
        raise PermissionError("Session time/heartbeat policy differs from the approval")
    modules = extension_hashes(repo)
    if policy.get("resumable_module_hashes") != modules:
        raise PermissionError("Resumable source hashes changed since approval")
    if not set(REQUIRED_TESTS) <= set(unit_test_evidence.get("modules", [])):
        raise PermissionError("Evidence must include the new resumable test suites")
    if (unit_test_evidence.get("git_commit") != approval.get("identity", {}).get("git_commit")
            or unit_test_evidence.get("resumable_module_hashes") != modules
            or unit_test_evidence.get("returncode") != 0):
        raise PermissionError("Unit-test evidence for the resumable source is stale")
    base = authorize_run(repo, windows, local_runs, drive_root, spec, approval_path,
                         technical_evidence, confirmation, unit_test_evidence=unit_test_evidence)
    if base.budgets["max_elapsed_seconds"] != session_budget_seconds:
        raise PermissionError("Supervisor hard elapsed budget must equal the approved session budget")
    training = _training(repo)
    from .dataset_validator import EXPECTED
    binding = {
        "identity": base.identity, "resumable_source_hashes": modules,
        "feature_names": list(training.FEATURE_NAMES),
        "scientific_contract": training.official_hyperparameters("mi_rfe_svm"),
        "rows_by_split": EXPECTED[spec.config], "seed": training.SEED,
        "preprocessing": "canonical float32 Inf/NaN->0; Train-only MI and Top100 scalers",
        "split_identity": {name: base.identity[name] for name in (
            "manifest_sha256", "validation_sha256", "window_summary_sha256",
            "dataset_fingerprint_sha256", "dataset_fingerprint_scope_sha256")},
    }
    root = run_root(drive_root, spec)
    if root.exists():
        if not resume:
            raise FileExistsError("Existing resumable run requires RESUME=True")
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("binding") != binding or manifest.get("binding_sha256") != digest(binding):
            raise PermissionError("Existing checkpoint run has another scientific identity")
    return ResumeAuthorization(base, policy.copy(), binding, root, resume, _SEAL)


def validate_authorization(context: object) -> ResumeAuthorization:
    if not isinstance(context, ResumeAuthorization) or context._seal is not _SEAL:
        raise PermissionError("A sealed Stage 7.3B resumable authorization is required")
    identity = context.base.identity
    validate_context_for_training(context.base, identity["config"], identity["experiment"],
                                  Path(identity["run_name"]))
    if identity["experiment"] != "mi_rfe_svm":
        raise PermissionError("Resumable mode is MI-RFE SVM only")
    if context.deadline_epoch is None or context.deadline_epoch <= 0:
        raise PermissionError("Execution context must be bound to its supervisor's session deadline")
    return context
