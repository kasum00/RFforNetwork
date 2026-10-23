"""New per-session Windows authority; immutable Colab experiment binding retained."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

from colab.svm_resumable_store import atomic_json, digest, file_sha, read_json, utc_now
from .checkpoint_migration import blocked
from .paths import token
from .windows_preflight import preflight

SCHEMA = "mi_rfe_svm.windows_local.authorization.v1"
_SEAL = object()


def phrase(audit: dict, session_id: str) -> str:
    return (f"AUTHORIZE WINDOWS {audit['scan']['binding']['identity']['run_name']} "
            f"{session_id} {audit['git']['commit']} {audit['scan']['binding_sha256']}")


def prior_state(root: Path) -> dict:
    lock = root / "active_session.json"
    progress = root / "progress.json"
    results = {}
    sessions = root / "sessions"
    if sessions.is_dir():
        for directory in sessions.iterdir():
            if directory.is_dir():
                result = directory / "session_result.json"
                results[directory.name] = {"report_present": result.is_file(),
                                           "report_sha256": file_sha(result) if result.is_file() else None}
    return {"active_session": read_json(lock) if lock.exists() else None,
            "active_session_sha256": file_sha(lock) if lock.exists() else None,
            "progress": read_json(progress) if progress.exists() else None,
            "session_reports": results}


def comparable(audit: dict) -> dict:
    return {k: audit[k] for k in ("paths", "git", "environment", "source_hashes", "source_bridge",
                                  "scan", "origin_authorization", "dataset", "input_evidence_sha256")}


def verify_review(paths, args, report_path: Path, *, hash_train_val=False) -> dict:
    report = read_json(report_path)
    if report.get("status") != "PASS_LOCAL_PREFLIGHT" or not report.get("content_verification"):
        blocked("Preflight must have passed resources and full Train/Validation content checks")
    if report["content_verification"].get("roles") != ["train", "val"]:
        blocked("Train/Validation content evidence missing")
    current = preflight(paths, args, hash_train_val=hash_train_val)
    if current["status"] != "PASS_LOCAL_PREFLIGHT" or comparable(current) != comparable(report):
        blocked("Preflight identity/checkpoints/data/resources changed; create a fresh review")
    return current


def create_approval(paths, args, report_path: Path, target: Path, *, ask=input) -> dict:
    audit = verify_review(paths, args, report_path)
    session_id = token(args.session_id)
    if (paths.checkpoint_root / "sessions" / session_id).exists():
        blocked("Session ID already used; never restart under the same session authorization")
    state = prior_state(paths.checkpoint_root)
    statement = phrase(audit, session_id)
    print(statement, flush=True)
    if ask("Type the exact AUTHORIZE WINDOWS phrase: ").strip() != statement:
        raise PermissionError("Run/session/source confirmation mismatch")
    if ask("Type CONFIRM COLAB SESSION01 TERMINATED: ").strip() != "CONFIRM COLAB SESSION01 TERMINATED":
        raise PermissionError("Origin Colab session termination not confirmed")
    if ask("Type ACCEPT CROSS-OS CHECKPOINT REUSE AND SINGLE TEST: ").strip() != "ACCEPT CROSS-OS CHECKPOINT REUSE AND SINGLE TEST":
        raise PermissionError("Explicit migration/scientific/Test acknowledgement missing")
    previous = state["active_session"]
    takeover = {"previous_session_id": "", "previous_session_terminated_confirmed": False}
    if previous:
        old_id = previous.get("session_id")
        confirmation = f"CONFIRM {old_id} TERMINATED AND TAKE OVER"
        if ask(f"Type {confirmation}: ").strip() != confirmation:
            raise PermissionError("Explicit interrupted-session takeover not confirmed")
        takeover = {"previous_session_id": old_id, "previous_session_terminated_confirmed": True}
    budgets = audit["resources"]["budgets"]
    if budgets["max_elapsed_seconds"] > 36000:
        extended = f"APPROVE EXTENDED LOCAL BUDGET {budgets['max_elapsed_seconds']}"
        if ask(f"Type {extended}: ").strip() != extended:
            raise PermissionError("Extended local execution was not separately approved")
    approval = {"schema": SCHEMA, "decision": "APPROVED_WINDOWS_LOCAL_SESSION", "effective": True,
                "created_utc": utc_now(), "execution_environment": "windows_local",
                "session_id": session_id, "confirmation": statement,
                "review_sha256": file_sha(report_path), "review_identity_sha256": digest(comparable(audit)),
                "physical_paths": paths.record(), "implementation_commit": audit["git"]["commit"],
                "implementation_hashes": audit["source_hashes"],
                "origin_binding_sha256": audit["scan"]["binding_sha256"],
                "origin_manifest_sha256": audit["scan"]["run_manifest_sha256"],
                "budgets": budgets, "session_budget_seconds": budgets["max_elapsed_seconds"],
                "heartbeat_interval_seconds": 60, "console_status_interval_seconds": 300,
                "stop_starting_new_work_margin_seconds": 300,
                "colab_session01_terminated_confirmed": True,
                "trusted_colab_export_and_pickle_risk_accepted": True,
                "cross_os_non_bitwise_equivalence_accepted": True,
                "completed_stage_resume_only": True, "single_final_test": True,
                "scenario_proxy_and_capture_limitations_accepted": True,
                "prior_state": state, "interrupted_session_takeover": takeover}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, approval, exclusive=True)
    return approval


@dataclass(frozen=True)
class LocalBase:
    identity: dict  # scientific origin identity, NOT a claim about the current host
    budgets: dict
    approval_path: str
    approval_sha256: str
    audit: dict

    def public_record(self):
        return {"schema": SCHEMA, "origin_identity": self.identity, "budgets": self.budgets,
                "approval_path": self.approval_path, "approval_sha256": self.approval_sha256,
                "execution_environment": "windows_local", "not_a_colab_authorization": True}


@dataclass(frozen=True)
class LocalContext:
    base: LocalBase
    binding: dict
    policy: dict
    root: Path
    paths: object
    approval: dict
    resume: bool
    console_interval: int
    _seal: object
    deadline_epoch: float | None = None


def authorize_local(paths, args, *, verify_content_now=False) -> LocalContext:
    audit = verify_review(paths, args, args.preflight_report, hash_train_val=verify_content_now)
    approval = read_json(args.approval)
    if (approval.get("schema"), approval.get("decision"), approval.get("effective"),
            approval.get("execution_environment")) != (SCHEMA, "APPROVED_WINDOWS_LOCAL_SESSION", True, "windows_local"):
        blocked("A new effective Windows approval is required; legacy Colab approval cannot run here")
    expected = {"session_id": token(args.session_id), "confirmation": phrase(audit, args.session_id),
                "review_sha256": file_sha(args.preflight_report),
                "review_identity_sha256": digest(comparable(audit)), "physical_paths": paths.record(),
                "implementation_commit": audit["git"]["commit"], "implementation_hashes": audit["source_hashes"],
                "origin_binding_sha256": audit["scan"]["binding_sha256"],
                "origin_manifest_sha256": audit["scan"]["run_manifest_sha256"],
                "budgets": audit["resources"]["budgets"],
                "session_budget_seconds": audit["resources"]["budgets"]["max_elapsed_seconds"],
                "heartbeat_interval_seconds": 60, "console_status_interval_seconds": 300,
                "stop_starting_new_work_margin_seconds": 300}
    if any(approval.get(k) != v for k, v in expected.items()) or args.confirmation != expected["confirmation"]:
        blocked("Local approval differs from reviewed session/source/data/path/budget identity")
    flags = ("colab_session01_terminated_confirmed", "trusted_colab_export_and_pickle_risk_accepted",
             "cross_os_non_bitwise_equivalence_accepted", "completed_stage_resume_only", "single_final_test",
             "scenario_proxy_and_capture_limitations_accepted")
    if not all(approval.get(k) is True for k in flags):
        blocked("Explicit migration acknowledgements missing")
    base = LocalBase(copy.deepcopy(audit["scan"]["binding"]["identity"]), approval["budgets"].copy(),
                     str(args.approval), file_sha(args.approval), audit)
    policy = {"session_id": args.session_id, "session_budget_seconds": approval["session_budget_seconds"],
              "heartbeat_interval_seconds": 60, "stop_starting_new_work_margin_seconds": 300}
    return LocalContext(base, copy.deepcopy(audit["scan"]["binding"]), policy, paths.checkpoint_root,
                        paths, approval, True, 300, _SEAL)


def validate_local_context(context, repo, windows, local_runs):
    if not isinstance(context, LocalContext) or context._seal is not _SEAL:
        raise PermissionError("A sealed Windows Local authorization is required")
    if (repo.resolve(), windows.resolve(), local_runs.resolve()) != (
            context.paths.repo, context.paths.data_root, context.paths.output_root):
        raise PermissionError("Execution paths differ from local authorization")
    if file_sha(Path(context.base.approval_path)) != context.base.approval_sha256:
        raise PermissionError("Approval changed after validation")
    if context.deadline_epoch is None or context.deadline_epoch <= time.time():
        raise PermissionError("Live supervisor-bound local session deadline required")
    return context
