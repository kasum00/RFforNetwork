"""Explicit Windows MI-RFE SVM CLI. Default is help, never implicit training."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from colab.svm_resumable_store import atomic_json, file_sha, read_json, utc_now
from .checkpoint_migration import copy_checkpoint_tree, scan_checkpoints
from .paths import LocalPaths, absolute, token


def record_tests(repo: Path, output: Path) -> dict:
    """Only executes when the HUMAN invokes the tests subcommand; no dataset use."""
    from .windows_preflight import REQUIRED_TESTS, implementation_hashes
    from colab.svm_official_readiness import git_checkout
    before = implementation_hashes(repo)
    git = git_checkout(repo)
    if not git.get("clean") or not git.get("commit"):
        raise PermissionError("Commit source manually and use a clean checkout before recording test evidence")
    if output.exists():
        raise FileExistsError("Never overwrite unit evidence")
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(repo), str(repo / "huyen_code")]), PYTHONUTF8="1")
    command = [sys.executable, "-m", "unittest", "-v", *REQUIRED_TESTS]
    completed = subprocess.run(command, cwd=repo, env=env)
    if completed.returncode != 0 or implementation_hashes(repo) != before or git_checkout(repo) != git:
        raise RuntimeError("Unit tests failed or source changed: no successful evidence written")
    evidence = {"passed": True, "returncode": completed.returncode, "git_commit": git["commit"],
                "source_hashes": before, "modules": list(REQUIRED_TESTS), "completed_utc": utc_now(),
                "full_training_authorized": False, "scope": "synthetic fixtures/mocks; not full model fit"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, evidence, exclusive=True)
    return evidence


def load_configuration(args):
    config_file = absolute(args.config_file)
    config = read_json(config_file)
    args.config_file = config_file
    path_names = ("repo", "data_root", "manifest_root", "checkpoint_root", "output_root",
                  "origin_approval", "technical_evidence", "dataset_checksums", "unit_test_evidence", "budget_file")
    for name in path_names:
        value = getattr(args, name, None) or config.get(name)
        if value is None:
            raise ValueError(f"Missing config/CLI field: {name}")
        setattr(args, name, absolute(value))
    args.session_id = token(getattr(args, "session_id", None) or config.get("session_id", "session02"))
    paths = LocalPaths.from_args(args)
    if Path(__file__).resolve().parents[1] != paths.repo:
        raise PermissionError("Local runner was imported from another checkout; use the clone's cwd/PYTHONPATH")
    return paths


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tests = sub.add_parser("tests", help="User-invoked synthetic unit tests and fresh commit-bound evidence")
    tests.add_argument("--repo", type=Path, required=True)
    tests.add_argument("--output", type=Path, required=True)
    imported = sub.add_parser("import-checkpoints", help="Optional absent-target import; no network")
    imported.add_argument("--source", type=Path, required=True)
    imported.add_argument("--checkpoint-root", type=Path, required=True)
    stop = sub.add_parser("request-stop", help="Request safe boundary stop; does not interrupt SVC.fit")
    stop.add_argument("--checkpoint-root", type=Path, required=True)
    stop.add_argument("--session-id", required=True)
    for name in ("inspect", "preflight", "approve", "run", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--config-file", type=Path, required=True)
        for path in ("repo", "data-root", "manifest-root", "checkpoint-root", "output-root",
                     "origin-approval", "technical-evidence", "dataset-checksums", "unit-test-evidence", "budget-file"):
            p.add_argument("--" + path, type=Path)
        p.add_argument("--session-id")
        if name == "preflight":
            p.add_argument("--report", type=Path, required=True)
        if name in ("approve", "run", "_worker"):
            p.add_argument("--preflight-report", type=Path, required=True)
            p.add_argument("--approval", type=Path, required=True)
        if name in ("run", "_worker"):
            p.add_argument("--confirmation", required=True)
        if name == "_worker":
            p.add_argument("--worker-token", required=True)
    args = parser.parse_args()
    try:
        if args.command == "tests":
            result = record_tests(absolute(args.repo), absolute(args.output))
        elif args.command == "import-checkpoints":
            result = copy_checkpoint_tree(absolute(args.source), absolute(args.checkpoint_root))
        elif args.command == "request-stop":
            root, session_id = absolute(args.checkpoint_root), token(args.session_id)
            active = read_json(root / "active_session.json")
            if active.get("session_id") != session_id or active.get("execution_environment") != "windows_local":
                raise PermissionError("Request must name the current Windows session")
            destination = root / "sessions" / session_id / "STOP_REQUESTED.json"
            atomic_json(destination, {"requested_utc": utc_now(), "session_id": session_id,
                                      "supervisor_token": active["token"]}, exclusive=True)
            result = {"status": "STOP_REQUESTED_AFTER_CURRENT_WORK_UNIT", "optimizer_checkpoint": False}
        else:
            paths = load_configuration(args)
            if args.command == "inspect":
                from colab.experiment_launcher import _training
                training = _training(paths.repo)
                result = scan_checkpoints(paths.checkpoint_root, expected_names=training.FEATURE_NAMES,
                                          expected_contract=training.official_hyperparameters("mi_rfe_svm"))
                result = {**result, "status": "CHECKPOINTS_VERIFIED_NOT_AUTHORIZED", "model_fit_calls": 0}
            elif args.command == "preflight":
                from .windows_preflight import preflight
                target = absolute(args.report)
                if target.exists():
                    raise FileExistsError("Preflight report exists; choose a new report path")
                try:
                    result = preflight(paths, args)
                except Exception as exc:
                    result = {"status": "MIGRATION_BLOCKED", "error": f"{type(exc).__name__}: {exc}",
                              "full_training_authorized": False, "model_fit_calls": 0, "test_csv_contents_read": 0}
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(target, result, exclusive=True)
            elif args.command == "approve":
                from .local_authorization import create_approval
                result = create_approval(paths, args, absolute(args.preflight_report), absolute(args.approval))
            else:
                from .local_supervisor import supervise, worker
                args.approval, args.preflight_report = absolute(args.approval), absolute(args.preflight_report)
                result = supervise(paths, args) if args.command == "run" else worker(paths, args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if result.get("status") in ("MIGRATION_BLOCKED", "RESOURCE_ADMISSION_FAIL") or (
            args.command in ("run", "_worker") and result.get("status") not in (
                "COMPLETED_VERIFIED", "PAUSED_AT_SAFE_CHECKPOINT")):
            return 2
        return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
