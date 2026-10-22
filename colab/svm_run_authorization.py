"""Fail-closed Stage 7.3B authorization for one official SVM run.

This module never grants approval from a probe or a resource estimate.  It only
verifies a human-authored approval against the current immutable run identity.
The returned in-memory context is required by every official SVM entry point.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import psutil

from .dataset_validator import manifest_root_for, sha256
from .experiment_launcher import RunSpec, preflight
from .setup_colab import runtime_info
from .svm_official_readiness import (SVM_EXPERIMENTS, git_checkout,
                                     normalized_source_sha256, source_contract)

SCHEMA = "stage7.3b.svm.authorization.v1"
CONTEXT_SCHEMA = "stage7.3b.svm.authorized-context.v1"
IDENTITY_FIELDS = (
    "experiment", "config", "run_name", "git_commit",
    "training_code_sha256", "sliding_code_sha256",
    "manifest_sha256", "validation_sha256", "window_summary_sha256",
    "dataset_fingerprint_sha256", "dataset_fingerprint_scope_sha256",
    "environment_profile_sha256", "original_script_hashes_sha256",
)
REQUIRED_BUDGET_FIELDS = (
    "max_elapsed_seconds", "max_tree_rss_bytes",
    "min_start_available_ram_bytes", "min_runtime_available_ram_bytes",
    "min_free_disk_bytes", "poll_interval_seconds",
)
_CONTEXT_SEAL = object()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def approval_sha256(path: Path) -> str:
    return sha256(path)


def expected_confirmation(identity: dict) -> str:
    return (f"AUTHORIZE STAGE7.3B {identity['experiment']} {identity['config']} "
            f"{identity['run_name']} {identity['git_commit']}")


def identity_matches(expected: object, actual: dict) -> bool:
    """Require every immutable field; missing/cross-run identities fail closed."""
    return isinstance(expected, dict) and all(
        isinstance(expected.get(field), str) and bool(expected[field])
        and expected[field] == actual.get(field) for field in IDENTITY_FIELDS)


def collect_identity(repo: Path, windows: Path, spec: RunSpec,
                     dataset: dict | None = None) -> tuple[dict, dict]:
    """Collect identity without reading Test feature rows or fitting a model."""
    spec.validate()
    if spec.experiment not in SVM_EXPERIMENTS:
        raise ValueError("Stage 7.3B authorization is only for SVM experiments")
    checkout = git_checkout(repo)
    if not checkout.get("clean") or not checkout.get("commit"):
        raise PermissionError("A clean, identifiable Git checkout is required")
    contract = source_contract(repo)
    if not all(contract.values()):
        raise PermissionError(f"Cross-platform source contract failed: {contract}")
    if dataset is None:
        from .dataset_validator import validate_dataset
        dataset = validate_dataset(windows, repo, all_headers=True)
    manifest_root = manifest_root_for(windows, repo)
    profile = runtime_info(Path("/content") if Path("/content").exists() else repo)
    selected_profile = {
        "python_major_minor": profile["python_major_minor"],
        "platform": profile["platform"],
        "machine": profile["machine"],
        "logical_cpus": profile["logical_cpus"],
        "physical_cpus": profile["physical_cpus"],
        "selected_scientific_profile": profile["selected_scientific_profile"],
    }
    original_hashes = {}
    original = repo / "CODE_ctu13"
    for name in (
        "2_labeling.py", "3_sliding_windown_and_aggregate_features.py",
        "7_senario1_svm.py", "[[CTU13]8_experiment2_svm.py",
        "[[CTU13]9_ex2_rf.py",
    ):
        original_hashes[name] = sha256(original / name)
    scope = dataset.get("fingerprint_scope")
    identity = {
        "experiment": spec.experiment,
        "config": spec.config,
        "run_name": spec.run_name,
        "git_commit": checkout["commit"],
        "training_code_sha256": normalized_source_sha256(
            repo / "huyen_code" / "ciciomt24_training.py"),
        "sliding_code_sha256": normalized_source_sha256(
            repo / "huyen_code" / "ciciomt24_sliding_window.py"),
        "manifest_sha256": sha256(manifest_root / "split_manifest.csv"),
        "validation_sha256": sha256(manifest_root / "validation.json"),
        "window_summary_sha256": sha256(windows / "window_run_summary.json"),
        "dataset_fingerprint_sha256": dataset["dataset_fingerprint_sha256"],
        "dataset_fingerprint_scope_sha256": _object_sha256(scope),
        "environment_profile_sha256": _object_sha256(selected_profile),
        "original_script_hashes_sha256": _object_sha256(original_hashes),
    }
    evidence = {
        "git": checkout, "source_contract": contract,
        "dataset_fingerprint_scope": scope,
        "environment_profile": selected_profile,
        "original_script_hashes": original_hashes,
    }
    return identity, evidence


def invalid_approval_template(identity: dict | None = None) -> dict:
    """Return a deliberately invalid template; this function cannot open a gate."""
    return {
        "schema": SCHEMA,
        "decision": "BLOCKED_TEMPLATE_REQUIRES_EXPLICIT_HUMAN_APPROVAL",
        "effective": False,
        "identity": identity or {field: None for field in IDENTITY_FIELDS},
        "confirmation": "",
        "budgets": {field: None for field in REQUIRED_BUDGET_FIELDS},
        "technical_evidence": {
            "policy": "REQUIRES_EXPLICIT_DECISION",
            "evidence_sha256": None,
            "evidence_identity": None,
            "approved_replacement_policy": None,
            "human_accepts_non_equivalence": False,
        },
        "scientific_contract_acknowledged": False,
        "test_is_single_final_evaluation_acknowledged": False,
        "capture_session_independence_not_established_acknowledged": False,
        "created_by_software_is_not_approval": True,
    }


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PermissionError(f"Approval budget {field} must be a positive integer")
    return value


def _validate_technical_evidence(approval: dict, technical_path: Path | None,
                                 identity: dict) -> dict:
    policy = approval.get("technical_evidence")
    if not isinstance(policy, dict):
        raise PermissionError("Missing explicit technical-evidence policy")
    mode = policy.get("policy")
    if mode not in ("EXACT_CONFIG_PROBE", "EXPLICIT_APPROVED_REPLACEMENT"):
        raise PermissionError("Technical evidence policy has not been approved")
    if technical_path is None or not technical_path.is_file():
        raise PermissionError("Approved technical evidence file is unavailable")
    actual_hash = sha256(technical_path)
    if actual_hash.casefold() != str(policy.get("evidence_sha256", "")).casefold():
        raise PermissionError("Technical evidence SHA256 differs from approval")
    evidence = json.loads(technical_path.read_text(encoding="utf-8"))
    if evidence.get("status") not in ("PASS_TECHNICAL_ONLY", "PASS_SCALING_PROBE"):
        raise PermissionError("Technical evidence is not a successful bounded probe")
    source_hashes = evidence.get("source_hashes") if isinstance(
        evidence.get("source_hashes"), dict) else {}
    observed = {
        "experiment": evidence.get("experiment"),
        "config": evidence.get("config"),
        "training_code_sha256": (evidence.get("training_code_sha256")
                                  or source_hashes.get("training")),
        "window_summary_sha256": (evidence.get("window_summary_sha256")
                                    or source_hashes.get("summary")),
    }
    if not all(isinstance(observed[key], str) and observed[key]
               for key in observed):
        raise PermissionError("Technical evidence lacks a complete source/config identity")
    evidence_identity = policy.get("evidence_identity")
    if not isinstance(evidence_identity, dict) or any(
            evidence_identity.get(key) != observed.get(key) for key in observed):
        raise PermissionError("Technical evidence identity is not exactly bound")
    if mode == "EXACT_CONFIG_PROBE":
        if evidence.get("experiment") != identity["experiment"] or evidence.get("config") != identity["config"]:
            raise PermissionError("Exact-config probe belongs to another experiment/config")
        if (observed["training_code_sha256"].casefold()
                != identity["training_code_sha256"].casefold()
                or observed["window_summary_sha256"].casefold()
                != identity["window_summary_sha256"].casefold()):
            raise PermissionError("Exact-config probe belongs to another source/dataset snapshot")
    else:
        statement = policy.get("approved_replacement_policy")
        if not isinstance(statement, str) or len(statement.strip()) < 40:
            raise PermissionError("Cross-config/replacement evidence needs an explicit rationale")
        if policy.get("human_accepts_non_equivalence") is not True:
            raise PermissionError("Replacement evidence non-equivalence was not acknowledged")
    if evidence.get("full_training_authorized") is not False:
        raise PermissionError("Probe evidence must not itself authorize full training")
    if (evidence.get("validation_csv_rows_read") != 0
            or evidence.get("test_csv_rows_read") != 0
            or ("scientific_result" in evidence
                and evidence.get("scientific_result") is not False)
            or ("official_best_k" in evidence
                and evidence.get("official_best_k") is not None)):
        raise PermissionError("Technical evidence violates Train-only/non-scientific probe rules")
    runs = evidence.get("runs")
    if not isinstance(runs, list) or not runs:
        raise PermissionError("Technical evidence has no completed bounded runs")
    if any(not isinstance(row, dict)
           or row.get("status") != "PASS_TECHNICAL_ONLY"
           or row.get("validation_csv_rows_read") != 0
           or row.get("test_csv_rows_read") != 0
           or row.get("scientific_result") is not False
           or row.get("official_best_k") is not None
           or ("full_training_authorized" in row
               and row.get("full_training_authorized") is not False)
           or row.get("worker_returncode") not in (None, 0)
           or row.get("supervisor_stop_reason") is not None
           for row in runs):
        raise PermissionError("One or more bounded technical runs are incomplete or unsafe")
    return {"policy": mode, "path": str(technical_path), "sha256": actual_hash,
            "observed_identity": observed}


def _output_conflicts(local_runs: Path, drive_root: Path, run_name: str) -> list[str]:
    candidates = (
        local_runs / run_name,
        local_runs / f"{run_name}.incomplete",
        drive_root / "experiment_results" / run_name,
        drive_root / "experiment_results" / f"{run_name}.incomplete",
        drive_root / "checkpoints" / f"{run_name}.incomplete",
        drive_root / "environment" / f"{run_name}.json",
    )
    return [str(path) for path in candidates if path.exists()]


@dataclass(frozen=True)
class AuthorizedRunContext:
    schema: str
    identity: dict
    budgets: dict
    approval_path: str
    approval_sha256: str
    confirmation: str
    audit: dict
    technical_evidence: dict
    _seal: object

    def public_record(self) -> dict:
        return {
            "schema": self.schema, "identity": self.identity,
            "budgets": self.budgets, "approval_path": self.approval_path,
            "approval_sha256": self.approval_sha256,
            "technical_evidence": self.technical_evidence,
            "authorization_is_run_specific": True,
        }


def authorize_run(repo: Path, windows: Path, local_runs: Path, drive_root: Path,
                  spec: RunSpec, approval_path: Path, technical_evidence_path: Path,
                  confirmation: str, *, unit_test_evidence: dict) -> AuthorizedRunContext:
    """Validate all gates. No model fit or Test feature row is performed here."""
    if not approval_path.is_file():
        raise PermissionError("A human-reviewed approval JSON is required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("schema") != SCHEMA or approval.get("decision") != "APPROVED_STAGE73B":
        raise PermissionError("Approval decision is not APPROVED_STAGE73B")
    if approval.get("effective") is not True:
        raise PermissionError("Approval is not effective")
    base = preflight(repo, windows, local_runs, drive_root, spec,
                     unit_test_evidence=unit_test_evidence,
                     svm_probe_path=technical_evidence_path)
    identity, identity_evidence = collect_identity(repo, windows, spec, base.get("dataset"))
    supplied_identity = approval.get("identity")
    if not identity_matches(supplied_identity, identity):
        raise PermissionError("Approval identity does not match this exact run/source/dataset/runtime")
    phrase = expected_confirmation(identity)
    if confirmation != phrase or approval.get("confirmation") != phrase:
        raise PermissionError(f"Exact authorization phrase required: {phrase}")
    if not all(approval.get(name) is True for name in (
        "scientific_contract_acknowledged",
        "test_is_single_final_evaluation_acknowledged",
        "capture_session_independence_not_established_acknowledged",
    )):
        raise PermissionError("Scientific/Test/proxy-data acknowledgements are incomplete")
    budgets_raw = approval.get("budgets")
    if not isinstance(budgets_raw, dict):
        raise PermissionError("Missing resource budgets")
    budgets = {field: _positive_integer(budgets_raw.get(field), field)
               for field in REQUIRED_BUDGET_FIELDS}
    if not 1 <= budgets["poll_interval_seconds"] <= 60:
        raise PermissionError("Polling interval must be between 1 and 60 seconds")
    technical = _validate_technical_evidence(approval, technical_evidence_path, identity)
    ignored = {"svm_technical_probe", "svm_full_resource_gate_reviewed"}
    failed_base = [name for name, passed in base.get("checks", {}).items()
                   if name not in ignored and not passed]
    if failed_base:
        raise PermissionError(f"Base preflight checks failed: {failed_base}")
    conflicts = _output_conflicts(local_runs, drive_root, spec.run_name)
    if conflicts:
        raise FileExistsError(f"Existing completed/incomplete run blocks execution: {conflicts}")
    available = psutil.virtual_memory().available
    free_disk = psutil.disk_usage(str(local_runs.parent if local_runs.parent.exists() else repo)).free
    if available < budgets["min_start_available_ram_bytes"]:
        raise MemoryError("Available RAM is below the explicitly approved start threshold")
    if budgets["max_tree_rss_bytes"] + budgets["min_runtime_available_ram_bytes"] > available:
        raise MemoryError("Approved process-tree RSS plus RAM reserve exceeds current available RAM")
    if free_disk < budgets["min_free_disk_bytes"]:
        raise OSError("Free local disk is below the explicitly approved threshold")
    audit = {
        **base,
        "status": "PASS_STAGE73B_AUTHORIZED",
        "full_training_authorized": True,
        "identity": identity,
        "identity_evidence": identity_evidence,
        "authorization_sha256": approval_sha256(approval_path),
        "authorization_checks": {
            "exact_identity": True, "exact_confirmation": True,
            "explicit_human_decision": True, "resource_budgets_bound": True,
            "technical_evidence_policy_bound": True,
            "no_existing_output_or_incomplete_run": True,
            "test_feature_rows_read": 0, "model_fit_calls": 0,
        },
    }
    return AuthorizedRunContext(
        CONTEXT_SCHEMA, identity, budgets, str(approval_path),
        approval_sha256(approval_path), confirmation, audit, technical, _CONTEXT_SEAL)


def validate_context_for_training(context: object, config: str, experiment: str,
                                  out: Path) -> AuthorizedRunContext:
    if not isinstance(context, AuthorizedRunContext) or context._seal is not _CONTEXT_SEAL:
        raise PermissionError("Official SVM requires a validated Stage 7.3B authorization context")
    if context.schema != CONTEXT_SCHEMA:
        raise PermissionError("Unknown authorization context schema")
    if (context.identity.get("config"), context.identity.get("experiment"),
            context.identity.get("run_name")) != (config, experiment, out.name):
        raise PermissionError("Authorization context belongs to another run")
    if context.confirmation != expected_confirmation(context.identity):
        raise PermissionError("Authorization confirmation no longer matches")
    if approval_sha256(Path(context.approval_path)) != context.approval_sha256:
        raise PermissionError("Approval file changed after authorization")
    return context
