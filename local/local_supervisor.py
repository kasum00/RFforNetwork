"""One Windows worker, explicit session authority, held OS lock, psutil PID identity."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import psutil

from colab.svm_official_supervisor import resource_stop_reason
from colab.svm_resumable_execution import _session_claim_guard, _claim_session_locked
from colab.svm_resumable_store import atomic_json, digest, read_json, utc_now
from .local_authorization import authorize_local, prior_state
from .windows_job import WindowsJob


@contextmanager
def windows_run_lock(root: Path):
    """Held by supervisor for its whole lifetime, not a short claim-only lock."""
    if os.name != "nt":
        raise PermissionError("Windows supervisor requires Windows file locking")
    import msvcrt
    with (root / ".windows_worker.guard").open("a+b") as handle:
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


def same_process(pid: int, created: float) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.create_time() == created
    except psutil.NoSuchProcess:
        return False


def terminate_owned(pid: int, created: float, seen=None) -> dict:
    """No process-name matching, no POSIX signals, no recycled-PID termination."""
    captured = dict(seen or {})
    errors, unconfirmed = [], set()
    try:
        if pid > 0 and same_process(pid, created):
            parent = psutil.Process(pid)
            if parent.create_time() == created and parent.is_running():
                captured[(pid, created)] = parent
                for p in parent.children(recursive=True):
                    try:
                        captured[(p.pid, p.create_time())] = p
                    except psutil.NoSuchProcess:
                        pass
            else:
                errors.append("Root PID changed before capture; replacement was not targeted")
    except psutil.NoSuchProcess:
        pass
    except (psutil.AccessDenied, OSError) as exc:
        errors.append(f"Root capture: {type(exc).__name__}: {exc}")
        unconfirmed.add(pid)
    live = []
    for (ident, birth), process in reversed(list(captured.items())):
        try:
            if process.is_running() and process.create_time() == birth:
                process.terminate()
                live.append(process)
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError) as exc:
            errors.append(f"Terminate {ident}: {type(exc).__name__}: {exc}")
            unconfirmed.add(ident)
    try:
        _, alive = psutil.wait_procs(live, timeout=10)
    except (psutil.AccessDenied, OSError) as exc:
        errors.append(f"Soft wait: {type(exc).__name__}: {exc}")
        alive = live
    for process in alive:
        try:
            key = (process.pid, process.create_time())
            if key in captured and process.is_running():
                process.kill()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError) as exc:
            errors.append(f"Kill {process.pid}: {type(exc).__name__}: {exc}")
            unconfirmed.add(process.pid)
    try:
        _, survivors = psutil.wait_procs(alive, timeout=10)
        unconfirmed.update(p.pid for p in survivors)
    except (psutil.AccessDenied, OSError) as exc:
        errors.append(f"Hard wait: {type(exc).__name__}: {exc}")
        unconfirmed.update(p.pid for p in alive)
    return {"captured_processes": len(captured), "survivors": sorted(unconfirmed), "cleanup_errors": errors,
            "scope": "worker PID/create-time and observed descendants only"}


def claim_local(context):
    root = context.root
    with _session_claim_guard(root):
        current = prior_state(root)
        if current != context.approval["prior_state"]:
            raise PermissionError("Previous session/heartbeat/lock changed since human review")
        previous = current["active_session"]
        if previous and previous.get("host") == socket.gethostname():
            if same_process(previous["supervisor_pid"], previous["supervisor_create_time"]):
                raise PermissionError("Previous local supervisor still running; takeover prohibited")
            if previous.get("worker_pid") and same_process(previous["worker_pid"], previous["worker_create_time"]):
                raise PermissionError("Previous local worker is still alive after supervisor loss; takeover prohibited")
        # Existing implementation archives old lock only after explicit approval and stale heartbeat.
        lock = _claim_session_locked(root, context, os.getpid())
        lock.update(execution_environment="windows_local", host=socket.gethostname())
        atomic_json(root / "active_session.json", lock)
        return lock


def monitor(process, created, context, session_dir, *, sleeper=time.sleep, clock=time.monotonic):
    started, peak, seen = clock(), 0, {}
    reason, error, termination = None, None, None
    try:
        while process.poll() is None:
            if not same_process(process.pid, created):
                raise PermissionError("Worker PID identity changed")
            parent = psutil.Process(process.pid)
            rss = 0
            for p in [parent, *parent.children(recursive=True)]:
                try:
                    seen[(p.pid, p.create_time())] = p
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            free_ram = psutil.virtual_memory().available
            from .windows_preflight import disk_anchor
            disk_free = min(psutil.disk_usage(str(disk_anchor(p))).free
                            for p in (context.paths.output_root, context.root))
            guard_elapsed = context.base.budgets["max_elapsed_seconds"] - (context.deadline_epoch - time.time())
            sample = {"utc": utc_now(), "elapsed_seconds": clock() - started,
                      "worker_pid": process.pid, "worker_create_time": created,
                      "tree_rss_bytes": rss, "peak_tree_rss_bytes_sampled": peak,
                      "host_available_ram_bytes": free_ram, "minimum_disk_free_bytes": disk_free}
            atomic_json(session_dir / "supervisor_progress.json", sample)
            reason = resource_stop_reason(guard_elapsed, rss, free_ram, disk_free, context.base.budgets)
            if reason:
                termination = terminate_owned(process.pid, created, seen)
                break
            sleeper(context.base.budgets["poll_interval_seconds"])
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        reason = "SUPERVISOR_EXCEPTION_OR_INTERRUPT"
        termination = terminate_owned(process.pid, created, seen)
    if process.poll() is None:
        process.wait(timeout=25)
    # Capture/reap only descendants actually observed under this worker.
    leftover = {(pid, birth): p for (pid, birth), p in seen.items() if pid != process.pid}
    cleanup = terminate_owned(-1, -1, leftover) if leftover else {"survivors": []}
    if cleanup["survivors"] or termination and termination["survivors"]:
        reason = "OWNED_PROCESS_TERMINATION_UNCONFIRMED"
    return {"worker_returncode": process.returncode, "stop_reason": reason,
            "supervisor_error": error, "termination": termination,
            "elapsed_seconds": clock() - started, "peak_tree_rss_bytes_sampled": peak,
            "rss_polling_not_a_hard_oom_guarantee": True, "cleanup": cleanup}


def supervise(paths, args) -> dict:
    context = authorize_local(paths, args)
    with windows_run_lock(context.root), WindowsJob() as job:
        lock = claim_local(context)
        context = replace(context, deadline_epoch=lock["deadline_epoch"])
        directory = context.root / "sessions" / args.session_id
        result = {"status": "INCOMPLETE_REQUIRES_MANUAL_REVIEW"}
        process, created = None, None
        safe_to_release = False
        try:
            atomic_json(directory / "local_execution.json", {
                "paths": paths.record(), "origin_binding_sha256": digest(context.binding),
                "implementation_commit": context.base.audit["git"]["commit"],
                "environment": context.base.audit["environment"],
                "dataset_checksums": str(args.dataset_checksums),
                "dataset_checksums_sha256": context.base.audit["input_evidence_sha256"]["dataset_checksums"],
                "approval_sha256": context.base.approval_sha256,
                "cross_os_numerical_bitwise_equivalence": "NOT_ESTABLISHED"}, exclusive=True)
            atomic_json(directory / "local_preflight.json", context.base.audit, exclusive=True)
            command = [sys.executable, "-u", "-m", "local.run_mi_rfe_svm", "_worker",
                       "--config-file", str(args.config_file), "--repo", str(paths.repo),
                       "--data-root", str(paths.data_root), "--manifest-root", str(paths.manifest_root),
                       "--checkpoint-root", str(paths.checkpoint_root), "--output-root", str(paths.output_root),
                       "--session-id", args.session_id, "--preflight-report", str(args.preflight_report),
                       "--approval", str(args.approval), "--confirmation", args.confirmation,
                       "--worker-token", lock["token"]]
            for name in ("origin_approval", "technical_evidence", "dataset_checksums", "unit_test_evidence", "budget_file"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            env = os.environ.copy()
            env.update(PYTHONPATH=os.pathsep.join([str(paths.repo), str(paths.repo / "huyen_code")]),
                       PYTHONUNBUFFERED="1", PYTHONUTF8="1")
            print(f"LOCAL SESSION STARTED: {args.session_id}; {context.base.audit['scan']['completed_k']} k complete; "
                  f"next={context.base.audit['scan']['next_stage']}", flush=True)
            # Inherit terminal stdout/stderr; do not hide progress behind capture_output.
            process = subprocess.Popen(command, cwd=paths.repo, env=env)
            created = psutil.Process(process.pid).create_time()
            job.assign(process)
            with _session_claim_guard(context.root):
                current_lock = read_json(context.root / "active_session.json")
                if current_lock.get("token") != lock["token"]:
                    raise PermissionError("Own session lock changed before recording worker identity")
                current_lock.update(worker_pid=process.pid, worker_create_time=created)
                atomic_json(context.root / "active_session.json", current_lock)
            ready_path = directory / "job_assignment.json"
            temporary_ready = directory / (".job_assignment.incomplete-" + lock["token"])
            atomic_json(temporary_ready, {
                "token": lock["token"], "worker_pid": process.pid, "worker_create_time": created,
                "kill_on_supervisor_close": True}, exclusive=True)
            os.rename(temporary_ready, ready_path)  # Windows rename: atomic publish, no existing-target overwrite.
            result.update(monitor(process, created, context, directory))
            safe_to_release = process.poll() is not None and not (
                result.get("termination", {}).get("survivors") if isinstance(result.get("termination"), dict) else False)
            safe_to_release = safe_to_release and not result.get("cleanup", {}).get("survivors")
            if result["worker_returncode"] == 0 and result["stop_reason"] is None:
                worker = read_json(directory / "local_worker_result.json")
                if worker.get("binding_sha256") != digest(context.binding):
                    raise ValueError("Worker result binding mismatch")
                if worker.get("status") not in ("COMPLETED_VERIFIED", "PAUSED_AT_SAFE_CHECKPOINT"):
                    raise ValueError("Unexpected worker result status")
                result.update(worker)
            else:
                result["status"] = "INCOMPLETE_REQUIRES_EXPLICIT_NEW_SESSION"
        except BaseException as exc:
            result.update(status="FAILED_REQUIRES_MANUAL_REVIEW", error=f"{type(exc).__name__}: {exc}")
            if process is not None and created is not None:
                result["termination"] = terminate_owned(process.pid, created)
                safe_to_release = not result["termination"]["survivors"] and process.poll() is not None
            elif process is None:
                safe_to_release = True
        finally:
            result.update(session_id=args.session_id, finished_utc=utc_now(),
                          binding_sha256=digest(context.binding), execution_environment="windows_local",
                          automatic_retry=False, automatic_new_session=False,
                          origin_binding_unchanged=True, scientific_result=result.get("status") == "COMPLETED_VERIFIED")
            atomic_json(directory / "session_result.json", result, exclusive=True)
            active = context.root / "active_session.json"
            if safe_to_release and active.is_file() and read_json(active).get("token") == lock["token"]:
                active.unlink()  # Only this supervisor's own operational lock; never an old lock.
        return result


def worker(paths, args) -> dict:
    # No preflight, unpickle or fit until the parent has placed us in its kill-on-close Job.
    lock = read_json(paths.checkpoint_root / "active_session.json")
    directory = paths.checkpoint_root / "sessions" / args.session_id
    ready = directory / "job_assignment.json"
    end = time.monotonic() + 30
    while not ready.is_file():
        if time.monotonic() >= end or not same_process(lock["supervisor_pid"], lock["supervisor_create_time"]):
            raise PermissionError("Supervisor Job handoff failed; worker cannot begin")
        time.sleep(.1)
    assigned = read_json(ready)
    if (assigned.get("token") != args.worker_token or assigned.get("worker_pid") != os.getpid()
            or assigned.get("kill_on_supervisor_close") is not True):
        raise PermissionError("Worker is not bound to the supervisor Job handoff")
    context = authorize_local(paths, args, verify_content_now=True)
    lock = read_json(context.root / "active_session.json")
    if (lock.get("token") != args.worker_token or lock.get("session_id") != args.session_id
            or lock.get("supervisor_pid") != os.getppid()
            or not same_process(os.getppid(), lock.get("supervisor_create_time"))):
        raise PermissionError("Worker requires its live parent's token and process identity")
    context = replace(context, deadline_epoch=lock["deadline_epoch"])
    from colab.svm_resumable_training import run_windows_resumable
    result = run_windows_resumable(paths.repo, paths.data_root, paths.output_root, context)
    result["binding_sha256"] = digest(context.binding)
    directory = context.root / "sessions" / args.session_id
    if result.get("status") == "PAUSED_AT_SAFE_CHECKPOINT" and (directory / "STOP_REQUESTED.json").exists():
        result["local_stop_reason"] = "USER_REQUESTED_STOP_AT_SAFE_BOUNDARY"
    atomic_json(directory / "local_worker_result.json", result, exclusive=True)
    return result
