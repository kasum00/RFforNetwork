"""Read-only discovery. Verification runs original A1/B code in a fresh process."""
import json
import subprocess
import sys
from pathlib import Path
from .common import (Blocked, CONFIGS, METHODS, digest, intact, read, selection_path)


def discover(output):
    rows = []
    for method, (package, algorithm, step) in METHODS.items():
        for config in CONFIGS:
            root = Path(output) / package / "mi_rfe_svm" / config
            for directory in sorted(root.iterdir()) if root.is_dir() else []:
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                row = dict(method=method, config=config, run_id=directory.name, directory=str(directory),
                           status="BLOCKED_SELECTION_NOT_VERIFIED")
                try:
                    identity = intact(read(directory / "run_manifest.json")["identity"])
                    if (identity["algorithm"] != algorithm or identity["config"] != config
                            or identity["run_id"] != directory.name or identity["contract"]["rfe"]["step"] != step):
                        raise Blocked("Selection method/config/run/step binding differs")
                    row.update(commit=identity["git_commit"], identity=identity["sha256"],
                               completion_present=(directory / "results/SELECTION_COMPLETE.json").is_file())
                    if not row["completion_present"]:
                        row["status"] = "BLOCKED_SELECTION_INCOMPLETE"
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["reason"] = str(exc)
                rows.append(row)
    return rows


def validate_extract(value, request):
    identity = intact(value["identity"])
    package, algorithm, step = METHODS[request["method"]]
    features, k = value["features"], value["best_k"]
    names = identity["dataset"]["names"]
    if (identity["algorithm"] != algorithm or identity["config"] != request["config"]
            or identity["run_id"] != request["selection_run_id"] or identity["contract"]["rfe"]["step"] != step
            or value["verification"]["status"] != "PASS" or value["verification"]["phase_status"] != "SELECTION_COMPLETE"
            or value["verification"]["test_evaluation_calls"] != 0
            or type(k) is not int or not 1 <= k <= 100 or len(features) != k or len(set(features)) != k
            or len(names) != 380 or len(set(names)) != 380 or not set(features) <= set(names)
            or features != value["ranked"][:k] or len(value["ranked"]) != 100
            or len(set(value["ranked"])) != 100 or not set(value["ranked"]) <= set(names)
            or features != value["frozen"]["features"] or k != value["frozen"]["best_k"]
            or value["frozen"]["identity"] != identity["sha256"]
            or value["audit"] != identity["audit"] or value["audit"]["fingerprint"] != identity["dataset_fingerprint"]):
        raise Blocked("BLOCKED_SELECTION_NOT_VERIFIED: extraction/binding/order mismatch")
    return value


def verify_selection(request):
    result = subprocess.run([sys.executable, "-B", "-u", "-m", "stage9_svm_final.selection_worker"],
        input=json.dumps(request), text=True, capture_output=True, cwd=request["source"])
    for line in result.stdout.splitlines():
        if not line.startswith("__STAGE9_SELECTION__"):
            print(line, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    marked = [line[len("__STAGE9_SELECTION__"):] for line in result.stdout.splitlines()
              if line.startswith("__STAGE9_SELECTION__")]
    if result.returncode or len(marked) != 1:
        raise Blocked("BLOCKED_SELECTION_NOT_VERIFIED: original verifier failed; inspect execution log")
    return validate_extract(json.loads(marked[0]), request)
