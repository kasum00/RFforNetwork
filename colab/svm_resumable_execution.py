"""Single authorized session supervisor/worker for completed-stage SVM resume."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .experiment_launcher import RunSpec
from .svm_official_supervisor import resource_stop_reason, _stop_only_worker_tree
from .svm_resumable_authorization import authorize_resume
from .svm_resumable_store import (
    CheckpointStore, atomic_json, digest, file_sha, read_json, utc_now,
)
from .svm_resumable_training import run_resumable


def _authorize(args, *, child=False):
    unit = read_json(args.unit_test_evidence)
    spec = RunSpec("mi_rfe_svm", args.config, args.run_name)
    return authorize_resume(
        args.repo, args.windows, args.local_runs, args.drive_root, spec,
        args.approval, args.technical_evidence, args.confirmation,
        unit_test_evidence=unit, session_id=args.session_id,
        resume=True if child else args.resume,
        session_budget_seconds=args.session_budget_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds)


@contextmanager
def _session_claim_guard(root: Path):
    """Serialize takeover+exclusive claim; unsupported OS/filesystem locks fail closed."""
    with (root / ".session_claim.guard").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def claim_session(root: Path, context, supervisor_pid: int) -> dict:
    with _session_claim_guard(root):
        return _claim_session_locked(root, context, supervisor_pid)


def _claim_session_locked(root: Path, context, supervisor_pid: int) -> dict:
    """A lost Colab session requires an explicit per-session takeover statement."""
    session_id = context.policy["session_id"]
    session_dir = root / "sessions" / session_id
    if session_dir.exists():
        raise FileExistsError("Session ID already used; create a new explicitly approved session")
    lock_path = root / "active_session.json"
    if lock_path.exists():
        previous = read_json(lock_path)
        approval = read_json(Path(context.base.approval_path))
        takeover = approval.get("interrupted_session_takeover") or {}
        if (takeover.get("previous_session_id") != previous.get("session_id")
                or takeover.get("previous_session_terminated_confirmed") is not True):
            raise PermissionError("Previous session lock exists; explicit terminated-session confirmation required")
        progress_path = root / "progress.json"
        if progress_path.is_file():
            latest = read_json(progress_path)
            if latest.get("session_id") == previous.get("session_id"):
                last = datetime.fromisoformat(latest["utc"])
                age = (datetime.now(timezone.utc) - last).total_seconds()
                if age < max(180, context.policy["heartbeat_interval_seconds"] * 3):
                    raise PermissionError("Recent heartbeat from previous session; concurrent execution blocked")
        archive = root / "sessions" / ("interrupted_lock_" + uuid.uuid4().hex + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if read_json(lock_path) != previous:
            raise PermissionError("Previous session lock changed during takeover review")
        os.rename(lock_path, archive)
    session_dir.mkdir(parents=True, exist_ok=False)
    lock = {"session_id": session_id, "supervisor_pid": supervisor_pid,
            "supervisor_create_time": psutil.Process(supervisor_pid).create_time(),
            "token": uuid.uuid4().hex, "started_utc": utc_now(),
            "deadline_epoch": time.time() + context.policy["session_budget_seconds"],
            "binding_sha256": digest(context.binding)}
    atomic_json(lock_path, lock, exclusive=True)
    atomic_json(session_dir / "authorization.json", {
        "base": context.base.public_record(), "resume_policy": context.policy,
        "binding_sha256": digest(context.binding), "resume_requested": context.resume,
    }, exclusive=True)
    return lock


def monitor_worker(process, budgets: dict, root: Path, session_dir: Path,
                   *, local_disk_path: Path | None = None,
                   deadline_epoch: float | None = None,
                   clock=time.monotonic, sleeper=time.sleep) -> dict:
    started, peak, stop_reason = clock(), 0, None
    seen = {}
    error, termination = None, None
    try:
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            rss = 0
            for proc in processes:
                try:
                    seen[(proc.pid, proc.create_time())] = proc
                    rss += proc.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            free_ram = psutil.virtual_memory().available
            free_disk = psutil.disk_usage(str(local_disk_path or root)).free
            elapsed = clock() - started
            guard_elapsed = (elapsed if deadline_epoch is None else
                             budgets["max_elapsed_seconds"] - (deadline_epoch - time.time()))
            sample = {"utc": utc_now(), "elapsed_seconds": elapsed,
                      "worker_pid": process.pid, "tree_rss_bytes": rss,
                      "peak_tree_rss_bytes_sampled": peak,
                      "host_available_ram_bytes": free_ram,
                      "disk_free_bytes": free_disk}
            atomic_json(session_dir / "supervisor_progress.json", sample)
            stop_reason = resource_stop_reason(guard_elapsed, rss, free_ram, free_disk, budgets)
            if stop_reason:
                termination = _stop_only_worker_tree(process.pid)
                break
            sleeper(budgets["poll_interval_seconds"])
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "SUPERVISOR_EXCEPTION_OR_INTERRUPT"
        if process.poll() is None:
            termination = _stop_only_worker_tree(process.pid)
    finally:
        # Descendants observed before root exit remain in scope; never kill by name.
        for (pid, created), proc in seen.items():
            if pid == process.pid:
                continue
            try:
                if proc.is_running() and proc.create_time() == created:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    code = process.poll()
    if code is None:
        code = process.wait(timeout=20)
    return {"worker_returncode": code, "stop_reason": stop_reason,
            "supervisor_error": error, "termination": termination,
            "elapsed_seconds": clock() - started,
            "peak_tree_rss_bytes_sampled": peak,
            "rss_polling_is_not_hard_memory_guarantee": True}


def supervisor(args) -> dict:
    context = _authorize(args)
    checkpoints = CheckpointStore(context.root, context.binding, resume=context.resume)
    lock = claim_session(context.root, context, os.getpid())
    session_dir = context.root / "sessions" / args.session_id
    result_path = session_dir / "worker_result.json"
    atomic_json(session_dir / "preflight.json", context.base.audit, exclusive=True)
    command = [sys.executable, "-m", "colab.svm_resumable_execution", "worker",
               "--repo", str(args.repo), "--windows", str(args.windows),
               "--local-runs", str(args.local_runs), "--drive-root", str(args.drive_root),
               "--config", args.config, "--run-name", args.run_name,
               "--approval", str(args.approval), "--technical-evidence", str(args.technical_evidence),
               "--unit-test-evidence", str(args.unit_test_evidence),
               "--confirmation", args.confirmation, "--session-id", args.session_id,
               "--session-budget-seconds", str(args.session_budget_seconds),
               "--heartbeat-interval-seconds", str(args.heartbeat_interval_seconds),
               "--worker-token", lock["token"], "--resume"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(args.repo), str(args.repo / "huyen_code")])
    report, process = {}, None
    try:
        print(f"SESSION START: {args.session_id} config={args.config} "
              f"resume={args.resume} budget={args.session_budget_seconds}s "
              f"heartbeat={args.heartbeat_interval_seconds}s", flush=True)
        process = subprocess.Popen(command, cwd=args.repo, env=env,
                                   start_new_session=os.name != "nt")
        disk_anchor = args.local_runs
        while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
            disk_anchor = disk_anchor.parent
        report = monitor_worker(process, context.base.budgets, context.root, session_dir,
                                local_disk_path=disk_anchor, deadline_epoch=lock["deadline_epoch"])
        if report["worker_returncode"] == 0 and report["stop_reason"] is None:
            worker = read_json(result_path)
            if worker.get("binding_sha256") != checkpoints.binding_sha256:
                raise ValueError("Worker completion identity mismatch")
            report.update(worker)
            if worker.get("status") == "COMPLETED_VERIFIED":
                from .svm_resumable_training import publish_results
                report["result_directory"] = str(publish_results(checkpoints))
            elif worker.get("status") != "PAUSED_AT_SAFE_CHECKPOINT":
                raise ValueError("Unexpected worker result status")
        else:
            report["status"] = "INCOMPLETE_REQUIRES_EXPLICIT_NEW_SESSION_AUTHORIZATION"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _stop_only_worker_tree(process.pid)
        report.update(status="FAILED_REQUIRES_MANUAL_REVIEW",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(session_id=args.session_id, finished_utc=utc_now(),
                      binding_sha256=checkpoints.binding_sha256,
                      automatic_retry=False, automatic_new_session=False,
                      test_claim_exists=(context.root / "results" / "test_evaluation_claim.json").exists(),
                      final_test_committed=checkpoints.directory("final_test").exists())
        atomic_json(session_dir / "session_result.json", report, exclusive=True)
        # Release only this supervisor's own lock, retaining every checkpoint/log.
        lock_path = context.root / "active_session.json"
        if lock_path.is_file() and read_json(lock_path).get("token") == lock["token"]:
            lock_path.unlink()
    return report


def worker(args) -> dict:
    context = _authorize(args, child=True)
    lock = read_json(context.root / "active_session.json")
    if (lock.get("session_id") != args.session_id or lock.get("token") != args.worker_token
            or lock.get("supervisor_pid") != os.getppid()
            or lock.get("supervisor_create_time") != psutil.Process(os.getppid()).create_time()):
        raise PermissionError("Worker requires its live supervising parent's session token")
    if not isinstance(lock.get("deadline_epoch"), (int, float)):
        raise PermissionError("Supervised session deadline is missing")
    context = replace(context, deadline_epoch=lock["deadline_epoch"])
    result = run_resumable(args.repo, args.windows, args.local_runs, context)
    result["binding_sha256"] = digest(context.binding)
    atomic_json(context.root / "sessions" / args.session_id / "worker_result.json", result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supervise", "worker"))
    for name in ("repo", "windows", "local-runs", "drive-root", "approval",
                 "technical-evidence", "unit-test-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", required=True,
                        choices=("FW_256_FS_8", "FW_256_FS_16", "FW_512_FS_8", "FW_512_FS_16"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-budget-seconds", type=int, default=36000)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-token", default="")
    args = parser.parse_args()
    result = supervisor(args) if args.command == "supervise" else worker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result.get("status") not in ("COMPLETED_VERIFIED", "PAUSED_AT_SAFE_CHECKPOINT"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
