"""Isolated, fail-closed supervisor for one authorized Stage 7.3B SVM run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .dataset_validator import EXPECTED, sha256
from .experiment_launcher import RunSpec
from .svm_run_authorization import authorize_run


def _atomic_replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _process_tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _tree_rss(pid: int) -> int:
    total = 0
    for process in _process_tree(pid):
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def resource_stop_reason(elapsed: float, tree_rss: int, available_ram: int,
                         free_disk: int, budgets: dict) -> str | None:
    """Pure guard decision used by the supervisor and mock unit tests."""
    if elapsed > budgets["max_elapsed_seconds"]:
        return "MAX_ELAPSED_SECONDS_EXCEEDED"
    if tree_rss > budgets["max_tree_rss_bytes"]:
        return "MAX_PROCESS_TREE_RSS_EXCEEDED"
    if available_ram < budgets["min_runtime_available_ram_bytes"]:
        return "HOST_AVAILABLE_RAM_BELOW_APPROVED_RESERVE"
    if free_disk < budgets["min_free_disk_bytes"]:
        return "FREE_DISK_BELOW_APPROVED_THRESHOLD"
    return None


def _stop_only_worker_tree(pid: int, grace_seconds: float = 10.0) -> dict:
    """Terminate only the worker root and descendants captured from its PID."""
    tree = _process_tree(pid)
    if not tree:
        return {"captured_processes": 0, "soft_terminate_requested": 0,
                "hard_kill_requested": 0, "hard_kill_used": False}
    soft = 0
    for process in reversed(tree):
        try:
            process.terminate()
            soft += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(tree, timeout=grace_seconds)
    hard = 0
    for process in alive:
        try:
            process.kill()
            hard += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=grace_seconds)
    return {"captured_processes": len(tree), "soft_terminate_requested": soft,
            "hard_kill_requested": hard, "hard_kill_used": hard > 0}


def _event_names(local_runs: Path, run_name: str) -> list[str]:
    for candidate in (local_runs / run_name / "events.jsonl",
                      local_runs / f"{run_name}.incomplete" / "events.jsonl"):
        if candidate.is_file():
            names = []
            for line in candidate.read_text(encoding="utf-8").splitlines():
                try:
                    names.append(json.loads(line).get("event"))
                except ValueError:
                    names.append("MALFORMED_EVENT")
            return names
    return []


def _verify_completed(drive_root: Path, spec: RunSpec) -> dict:
    result_dir = drive_root / "experiment_results" / spec.run_name
    result_path = result_dir / "result.json"
    manifest_path = result_dir / "artifact_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Worker exited successfully but published result is incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("experiment") != spec.experiment or result.get("config") != spec.config:
        raise ValueError("Published result identity mismatch")
    if result.get("rows_by_split") != EXPECTED[spec.config]:
        raise ValueError("Published split counts mismatch")
    if result.get("test_evaluation_calls") != 1 or not result.get("test_read_after_final_refit"):
        raise ValueError("Final Test evaluation contract mismatch")
    if result.get("label_origin") != "SCENARIO_PROXY":
        raise ValueError("Label origin mismatch")
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"result.json", "model.joblib", "final_scaler.joblib",
                "authorization.json", "hyperparameters.json", "feature_schema.json"}
    if not required <= set(artifact_manifest.get("files", {})):
        raise ValueError("Published artifact manifest is incomplete")
    for name, expected in artifact_manifest["files"].items():
        artifact = result_dir / name
        if (not artifact.is_file() or artifact.stat().st_size != expected.get("bytes")
                or sha256(artifact).casefold() != str(expected.get("sha256", "")).casefold()):
            raise ValueError(f"Published artifact failed size/SHA256 verification: {name}")
    return {"result_dir": str(result_dir), "status": result["status"],
            "test_evaluation_calls": 1,
            "artifact_manifest_files": len(artifact_manifest["files"]),
            "artifact_manifest_sha256": sha256(manifest_path),
            "all_published_file_hashes_verified": True}


def supervise(args) -> dict:
    spec = RunSpec(args.experiment, args.config, args.run_name)
    unit_evidence = json.loads(args.unit_test_evidence.read_text(encoding="utf-8"))
    context = authorize_run(
        args.repo, args.windows, args.local_runs, args.drive_root, spec,
        args.approval, args.technical_evidence, args.authorization_confirmation,
        unit_test_evidence=unit_evidence)
    report_dir = args.drive_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    final_report = report_dir / f"{spec.run_name}_supervisor.json"
    incomplete_report = report_dir / f"{spec.run_name}_supervisor.json.incomplete"
    events_path = report_dir / f"{spec.run_name}_supervisor_events.jsonl"
    if final_report.exists() or incomplete_report.exists() or events_path.exists():
        raise FileExistsError("Existing supervisor evidence blocks a new launch")
    worker_result = args.local_runs / f".{spec.run_name}.worker_result.json"
    if worker_result.exists():
        raise FileExistsError(worker_result)
    command = [
        sys.executable, "-m", "colab.svm_official_worker",
        "--repo", str(args.repo), "--windows", str(args.windows),
        "--local-runs", str(args.local_runs), "--drive-root", str(args.drive_root),
        "--experiment", spec.experiment, "--config", spec.config,
        "--run-name", spec.run_name, "--approval", str(args.approval),
        "--technical-evidence", str(args.technical_evidence),
        "--unit-test-evidence", str(args.unit_test_evidence),
        "--authorization-confirmation", args.authorization_confirmation,
        "--worker-result", str(worker_result),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.repo), str(args.repo / "huyen_code"), environment.get("PYTHONPATH", "")])
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    peak_rss = 0
    stop_reason = None
    termination = None
    process = subprocess.Popen(command, cwd=args.repo, env=environment)
    initial = {
        "status": "RUNNING", "worker_pid": process.pid,
        "started_utc": started_wall, "identity": context.identity,
        "approval_sha256": context.approval_sha256,
        "budgets": context.budgets, "retry_count": 0,
        "resume_supported": False,
    }
    _atomic_replace_json(incomplete_report, initial)
    supervisor_error = None
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            current_rss = _tree_rss(process.pid)
            peak_rss = max(peak_rss, current_rss)
            available = psutil.virtual_memory().available
            free_disk = psutil.disk_usage(str(args.local_runs.parent)).free
            progress_path = args.local_runs / f"{spec.run_name}.incomplete" / "progress.json"
            stage = None
            if progress_path.is_file():
                try:
                    stage = json.loads(progress_path.read_text(encoding="utf-8")).get("stage")
                except ValueError:
                    stage = "MALFORMED_PROGRESS"
            sample = {
                "utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
                "worker_pid": process.pid, "tree_rss_bytes": current_rss,
                "peak_tree_rss_bytes": peak_rss, "host_available_ram_bytes": available,
                "local_disk_free_bytes": free_disk, "stage": stage,
            }
            _append_jsonl(events_path, sample)
            _atomic_replace_json(incomplete_report, {**initial, **sample})
            stop_reason = resource_stop_reason(
                elapsed, current_rss, available, free_disk, context.budgets)
            if stop_reason:
                termination = _stop_only_worker_tree(process.pid)
                break
            time.sleep(context.budgets["poll_interval_seconds"])
    except BaseException as exc:
        supervisor_error = f"{type(exc).__name__}: {exc}"
        stop_reason = "SUPERVISOR_EXCEPTION_OR_INTERRUPT"
        if process.poll() is None:
            termination = _stop_only_worker_tree(process.pid)
    returncode = process.poll()
    if returncode is None:
        returncode = process.wait(timeout=20)
    events = _event_names(args.local_runs, spec.run_name)
    test_started = "test_evaluation_started" in events
    run_validated = "run_validated" in events
    completed = None
    if stop_reason is None and returncode == 0:
        try:
            if not worker_result.is_file():
                raise FileNotFoundError("Worker completion evidence is missing")
            worker_evidence = json.loads(worker_result.read_text(encoding="utf-8"))
            if (worker_evidence.get("authorization_sha256") != context.approval_sha256
                    or worker_evidence.get("identity") != context.identity
                    or worker_evidence.get("worker_completed_without_retry") is not True):
                raise ValueError("Worker completion evidence identity mismatch")
            completed = _verify_completed(args.drive_root, spec)
            completed["worker_result_sha256"] = sha256(worker_result)
            status = "COMPLETED_VERIFIED"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            supervisor_error = f"{type(exc).__name__}: {exc}"
            stop_reason = "POST_RUN_ARTIFACT_VERIFICATION_FAILED"
            status = "INCOMPLETE_REQUIRES_MANUAL_REVIEW"
    else:
        status = "INCOMPLETE_REQUIRES_MANUAL_REVIEW"
        stop_reason = stop_reason or f"WORKER_EXIT_{returncode}"
    report = {
        **initial, "status": status,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_tree_rss_bytes": peak_rss, "worker_returncode": returncode,
        "supervisor_stop_reason": stop_reason,
        "supervisor_error": supervisor_error,
        "termination": termination,
        "test_evaluation_started": test_started,
        "run_validated": run_validated,
        "retry_performed": False, "automatic_resume_performed": False,
        "manual_review_required_before_any_future_action": status != "COMPLETED_VERIFIED",
        "do_not_retry_if_test_started_without_validation": test_started and not run_validated,
        "completed_result": completed,
        "rss_sampling_is_not_a_hard_oom_guarantee": True,
    }
    _atomic_replace_json(incomplete_report, report)
    os.replace(incomplete_report, final_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--local-runs", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--experiment", choices=("baseline_svm", "mi_rfe_svm"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--technical-evidence", type=Path, required=True)
    parser.add_argument("--unit-test-evidence", type=Path, required=True)
    parser.add_argument("--authorization-confirmation", required=True)
    args = parser.parse_args()
    result = supervise(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "COMPLETED_VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
