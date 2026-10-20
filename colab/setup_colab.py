"""Read-only runtime and repository checks; no training side effects."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_VERSIONS = {
    "numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.15.3",
    "scikit-learn": "1.7.2", "joblib": "1.5.3",
}
PY313_VERSIONS = {**EXPECTED_VERSIONS, "numpy": "2.2.6"}


def selected_versions(python_version: tuple[int, int] | None = None) -> dict[str, str]:
    version = python_version or tuple(sys.version_info[:2])
    if version in ((3, 10), (3, 11), (3, 12)):
        return EXPECTED_VERSIONS.copy()
    if version == (3, 13):
        return PY313_VERSIONS.copy()
    raise RuntimeError(f"Python {version[0]}.{version[1]} is not prepared; official training blocked")
REQUIRED_REPO_FILES = (
    "huyen_code/ciciomt24_training.py",
    "huyen_code/ciciomt24_sliding_window.py",
    "huyen_code/ciciomt24_split.py",
    "huyen_code/test_ciciomt24_training.py",
    "huyen_code/test_ciciomt24_split.py",
    "huyen_code/test_ciciomt24_sliding_window.py",
    "CODE_ctu13/2_labeling.py",
    "CODE_ctu13/3_sliding_windown_and_aggregate_features.py",
    "CODE_ctu13/7_senario1_svm.py",
    "CODE_ctu13/[[CTU13]8_experiment2_svm.py",
    "CODE_ctu13/[[CTU13]9_ex2_rf.py",
    "configs/cplus_manifest/split_manifest.csv",
    "configs/cplus_manifest/validation.json",
)


def runtime_info(path: Path = Path("/content")) -> dict:
    import psutil
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(path)
    packages = {}
    for name in (*EXPECTED_VERSIONS, "psutil", "google-api-python-client"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"available": False, "name": None}
    probe = shutil.which("nvidia-smi")
    if probe:
        result = subprocess.run([probe, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0 and result.stdout.strip():
            gpu = {"available": True, "name": result.stdout.strip().splitlines()[0]}
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "python_major_minor": list(sys.version_info[:2]),
        "logical_cpus": os.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_bytes": mem.total, "ram_available_bytes": mem.available,
        "disk_total_bytes": disk.total, "disk_free_bytes": disk.free,
        "gpu": gpu, "versions": packages,
        "scientific_versions_match_stage5": all(
            packages[name] == version for name, version in EXPECTED_VERSIONS.items()),
        "scientific_versions_match_selected_profile": all(
            packages[name] == version for name, version in selected_versions().items()),
        "selected_scientific_profile": selected_versions(),
        "gpu_accelerates_this_sklearn_pipeline": False,
    }


def repo_inventory(repo: Path) -> dict:
    missing = [name for name in REQUIRED_REPO_FILES if not (repo / name).is_file()]
    return {"repo": str(repo), "missing": missing, "pass": not missing}


def clone_or_inspect(url: str, target: Path) -> dict:
    """Clone only into an absent path; never pull/reset a modified checkout."""
    if target.exists():
        if not (target / ".git").is_dir():
            raise FileExistsError(f"Existing path is not a Git checkout: {target}")
        status = subprocess.run(["git", "-C", str(target), "status", "--porcelain"],
                                capture_output=True, text=True, check=True)
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True)
        return {"action": "existing_checkout_no_pull", "head": head.stdout.strip(),
                "dirty": bool(status.stdout.strip()), "changes": status.stdout.splitlines()}
    if url != "https://github.com/kasum00/RFforNetwork.git":
        raise ValueError("Unexpected repository URL")
    subprocess.run(["git", "clone", "--", url, str(target)], check=True)
    return {"action": "cloned", **clone_or_inspect(url, target)}


def assert_supported_python() -> None:
    selected_versions()


def dependency_plan() -> dict:
    expected = selected_versions()
    current = {}
    for name in expected:
        try:
            current[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            current[name] = None
    return {"current": current, "expected": expected,
            "install_needed": any(current[n] != v for n, v in expected.items()),
            "stage5_exact_versions": current == EXPECTED_VERSIONS}


def save_environment(path: Path, info: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
