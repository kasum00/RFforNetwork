"""Internal single-run worker for the Stage 7.3B SVM supervisor.

Do not invoke this module directly.  It independently revalidates the approval
before delegating to the existing scientific orchestration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .experiment_launcher import RunSpec, run_one
from .svm_run_authorization import authorize_run


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
    parser.add_argument("--worker-result", type=Path, required=True)
    args = parser.parse_args()

    evidence = json.loads(args.unit_test_evidence.read_text(encoding="utf-8"))
    spec = RunSpec(args.experiment, args.config, args.run_name)
    context = authorize_run(
        args.repo, args.windows, args.local_runs, args.drive_root, spec,
        args.approval, args.technical_evidence, args.authorization_confirmation,
        unit_test_evidence=evidence)
    result = run_one(
        args.repo, args.windows, args.local_runs, args.drive_root, spec,
        unit_test_evidence=evidence,
        confirmation=f"RUN {spec.experiment} {spec.config} {spec.run_name}",
        svm_probe_path=args.technical_evidence,
        official_authorization=context)
    _atomic_json(args.worker_result, {
        "status": result["status"], "run": result,
        "authorization_sha256": context.approval_sha256,
        "identity": context.identity,
        "worker_completed_without_retry": True,
    })


if __name__ == "__main__":
    main()
