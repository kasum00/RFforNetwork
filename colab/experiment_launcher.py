"""One-config Colab adapter reusing Stage 5's scientific functions unchanged.

Only orchestration is wrapped: portable manifest location, durable completed-stage
checkpoints, and serialization of the already-fitted model before the first Test read.
No incomplete fit is resumable.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .checkpoint_manager import CheckpointSink
from .dataset_validator import EXPECTED, manifest_root_for, sha256, validate_dataset
from .setup_colab import EXPECTED_VERSIONS, repo_inventory, runtime_info


OBSERVED_FS16 = (
    {"peak": 10_730_827_776, "rows": {"train": 673_559, "val": 224_373, "test": 224_287}},
    {"peak": 10_744_430_592, "rows": {"train": 674_169, "val": 224_677, "test": 224_631}},
)
EXPERIMENTS = ("mi_rfe_rf", "baseline_svm", "mi_rfe_svm")


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    config: str
    run_name: str

    def validate(self) -> None:
        if self.experiment not in EXPERIMENTS or self.config not in EXPECTED:
            raise ValueError("Unknown experiment/config")
        if not self.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_name):
            raise ValueError("Run name must use only letters, digits, _ and -")


def svm_stage71_checks(probe: dict, spec: RunSpec,
                       training_hash: str, window_summary_hash: str) -> dict[str, bool]:
    """A bounded probe is evidence for review, never automatic full-run approval."""
    try:
        from ciciomt24_svm_probe import SCHEMA as SVM_PROBE_SCHEMA
    except ImportError:
        return {"svm_technical_probe": False,
                "svm_full_resource_gate_reviewed": False}
    runs = probe.get("runs", []) if isinstance(probe, dict) else []
    sizes = probe.get("sizes", []) if isinstance(probe, dict) else []
    complete_runs = (
        isinstance(sizes, list) and bool(sizes)
        and all(isinstance(n, int) and 4 <= n <= 8192 for n in sizes)
        and sizes == sorted(set(sizes))
        and isinstance(runs, list) and len(runs) == len(sizes)
        and all(
            isinstance(row, dict)
            and row.get("sample_rows") == n
            and row.get("status") == "PASS_TECHNICAL_ONLY"
            and row.get("worker_returncode") == 0
            and row.get("supervisor_stop_reason") is None
            and row.get("validation_csv_rows_read") == 0
            and row.get("test_csv_rows_read") == 0
            and isinstance(row.get("fits"), list) and bool(row["fits"])
            and all(isinstance(fit, dict) and isinstance(fit.get("fit"), dict)
                    and fit["fit"].get("converged") is True
                    and fit["fit"].get("fit_status") == 0
                    for fit in row["fits"])
            for n, row in zip(sizes, runs)
        )
    )
    valid = isinstance(probe, dict) and complete_runs and all((
        bool(training_hash),
        bool(window_summary_hash),
        probe.get("schema") == SVM_PROBE_SCHEMA,
        probe.get("status") == "PASS_TECHNICAL_ONLY",
        probe.get("experiment") == spec.experiment,
        probe.get("config") == spec.config,
        str(probe.get("training_code_sha256", "")).casefold() == training_hash.casefold(),
        str(probe.get("window_summary_sha256", "")).casefold() == window_summary_hash.casefold(),
        probe.get("validation_csv_rows_read") == 0,
        probe.get("test_csv_rows_read") == 0,
        probe.get("full_training_authorized") is False,
    ))
    return {"svm_technical_probe": valid,
            "svm_full_resource_gate_reviewed": False}


def _training(repo: Path):
    source = str(repo / "huyen_code")
    if source not in sys.path:
        sys.path.insert(0, source)
    import ciciomt24_training as training
    return training


def rf_empirical_required(counts: dict[str, int]) -> int:
    """Stage 5.4 unchanged: max role/total ratio, plus max(2 GiB, 20%)."""
    total = sum(counts.values())
    required = []
    for run in OBSERVED_FS16:
        old = run["rows"]
        ratio = max(*(counts[role] / old[role] for role in ("train", "val", "test")),
                    total / sum(old.values()))
        scaled = math.ceil(run["peak"] * ratio)
        required.append(scaled + max(2 * 1024**3, math.ceil(.2 * scaled)))
    return max(required)


def preflight(repo: Path, windows: Path, local_runs: Path, drive_root: Path,
              spec: RunSpec, *, unit_test_evidence: dict | None = None,
              svm_probe_path: Path | None = None) -> dict:
    """Read-only gates; no model.fit, no Test feature rows."""
    spec.validate()
    checks = {}
    inventory = repo_inventory(repo)
    checks["repository_files"] = inventory["pass"]
    if not checks["repository_files"]:
        return {"status": "FAIL", "checks": checks, "missing_repo_files": inventory["missing"]}
    training = _training(repo)
    info = runtime_info(local_runs.parent if local_runs.parent.exists() else Path("/content"))
    checks["python_supported"] = tuple(sys.version_info[:2]) in ((3, 10), (3, 11), (3, 12), (3, 13))
    checks["selected_scientific_profile"] = info["scientific_versions_match_selected_profile"]
    checks["unit_tests_current_code"] = bool(
        unit_test_evidence and unit_test_evidence.get("passed") and
        unit_test_evidence.get("training_sha256") == sha256(repo / "huyen_code" / "ciciomt24_training.py"))
    try:
        training.source_hashes(repo / "CODE_ctu13")
        training.source_audit(repo / "CODE_ctu13")
        checks["approved_original_scripts"] = True
    except (Exception, AssertionError) as exc:
        checks["approved_original_scripts"] = False
        original_error = str(exc)
    try:
        dataset = validate_dataset(windows, repo, all_headers=True)
        checks["dataset_manifest_schema"] = True
    except Exception as exc:
        checks["dataset_manifest_schema"] = False
        dataset_error = str(exc)
        dataset = None
    checks["no_existing_local_run"] = not any((local_runs / (spec.run_name + suffix)).exists()
                                             for suffix in ("", ".incomplete"))
    checks["no_existing_drive_run"] = not any((drive_root / sub / name).exists() for sub, name in (
        ("experiment_results", spec.run_name),
        ("experiment_results", spec.run_name + ".incomplete"),
        ("checkpoints", spec.run_name + ".incomplete")))
    checks["drive_mounted"] = Path("/content/drive/MyDrive").is_dir()
    checks["prior_fs16_not_rerun"] = not (spec.experiment == "mi_rfe_rf" and spec.config.endswith("FS_16"))
    if spec.experiment != "mi_rfe_rf":
        try:
            probe = json.loads(svm_probe_path.read_text(encoding="utf-8")) if svm_probe_path else {}
        except (OSError, ValueError):
            probe = {}
        # A small Train-only timing probe cannot certify full SVM resources.
        # Stage 7.2 must define and explicitly approve that separate gate.
        summary_path = windows / "window_run_summary.json"
        checks.update(svm_stage71_checks(
            probe, spec, sha256(repo / "huyen_code" / "ciciomt24_training.py"),
            sha256(summary_path) if summary_path.is_file() else ""))
    if dataset is not None:
        manifest_root = manifest_root_for(windows, repo)
        store = training.WindowStore(windows, split_root=manifest_root)
        counts = EXPECTED[spec.config]
        available = info["ram_available_bytes"]
        if spec.experiment == "mi_rfe_rf":
            required_ram = rf_empirical_required(counts)
            guard = {"policy": "stage54_empirical_scaled_FS16_peak_plus_max_2GiB_or_20pct",
                     "required_available_bytes": required_ram, "available_bytes": available,
                     "guard_is_not_proven_peak_bound": True}
        else:
            try:
                guard = training.official_memory_preflight(store, spec.config, spec.experiment)
                required_ram = math.ceil(guard["estimated_scenario_bytes"] / .85)
            except MemoryError:
                guard = {"policy": "source_official_memory_preflight",
                         "status": "FAIL", "available_bytes": available}
                required_ram = available + 1
        checks["ram_admission"] = available >= required_ram
        raw = sum(counts.values()) * 380 * 4 + sum(counts.values())
        disk_required = max(math.ceil(raw * 1.5), 8 * 1024**3)
        checks["local_disk_admission"] = info["disk_free_bytes"] >= disk_required
    else:
        guard, disk_required = None, None
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "runtime": info, "dataset": dataset, "ram_guard": guard,
            "local_disk_required_bytes": disk_required,
            "original_error": locals().get("original_error"),
            "dataset_error": locals().get("dataset_error"),
            "test_feature_rows_read": 0, "model_fit_calls": 0}


def run_technical_svm_probe(repo: Path, windows: Path, spec: RunSpec, output: Path) -> dict:
    """Stage 7.1 Train-only probe; never an official model or full-run gate."""
    spec.validate()
    if spec.experiment not in ("baseline_svm", "mi_rfe_svm"):
        raise ValueError("SVM probe is only meaningful for SVM")
    if output.exists():
        raise FileExistsError(output)
    _training(repo)  # Adds the existing scientific implementation to sys.path.
    from ciciomt24_svm_probe import run_probe
    return run_probe(repo, windows, manifest_root_for(windows, repo),
                     spec.config, spec.experiment, output)


def run_one(repo: Path, windows: Path, local_runs: Path, drive_root: Path,
            spec: RunSpec, *, unit_test_evidence: dict,
            confirmation: str, svm_probe_path: Path | None = None,
            official_authorization=None) -> dict:
    """Called only by an explicit notebook Full-train action and exact phrase."""
    expected_phrase = f"RUN {spec.experiment} {spec.config} {spec.run_name}"
    if confirmation != expected_phrase:
        raise PermissionError(f"Full training requires typing exactly: {expected_phrase}")
    if spec.experiment in ("baseline_svm", "mi_rfe_svm"):
        from .svm_run_authorization import validate_context_for_training
        authorized = validate_context_for_training(
            official_authorization, spec.config, spec.experiment,
            local_runs / spec.run_name)
        audit = authorized.audit
        if audit.get("status") != "PASS_STAGE73B_AUTHORIZED":
            raise RuntimeError("Stage 7.3B SVM authorization audit did not pass")
    else:
        authorized = None
        audit = preflight(repo, windows, local_runs, drive_root, spec,
                          unit_test_evidence=unit_test_evidence, svm_probe_path=svm_probe_path)
        if audit["status"] != "PASS":
            raise RuntimeError(f"Resource/dataset preflight FAIL: {audit['checks']}")
    from .drive_manager import require_writable_result_root
    require_writable_result_root(drive_root)  # Actual write probe before training.
    training = _training(repo)
    local_runs.mkdir(parents=True, exist_ok=True)
    out = local_runs / spec.run_name
    staged = out.with_name(out.name + ".incomplete")
    identity = {"experiment": spec.experiment, "config": spec.config,
                "run_name": spec.run_name, "seed": training.SEED,
                "dataset_fingerprint_sha256": audit["dataset"]["dataset_fingerprint_sha256"],
                "dataset_fingerprint_scope": audit["dataset"]["fingerprint_scope"],
                "training_code_sha256": sha256(repo / "huyen_code" / "ciciomt24_training.py"),
                "adapter_code_sha256": sha256(Path(__file__)),
                "original_script_hashes": training.source_hashes(repo / "CODE_ctu13"),
                "library_versions": audit["runtime"]["versions"]}
    if authorized is not None:
        identity.update(
            stage73b_authorization_sha256=authorized.approval_sha256,
            git_commit=authorized.identity["git_commit"],
            manifest_sha256=authorized.identity["manifest_sha256"],
            validation_sha256=authorized.identity["validation_sha256"],
            window_summary_sha256=authorized.identity["window_summary_sha256"],
            dataset_fingerprint_scope_sha256=authorized.identity[
                "dataset_fingerprint_scope_sha256"],
            environment_profile_sha256=authorized.identity[
                "environment_profile_sha256"],
        )
    from .setup_colab import save_environment
    save_environment(drive_root / "environment" / (spec.run_name + ".json"),
                     {**identity, "runtime": audit["runtime"],
                      "preflight_checks": audit["checks"]})
    sink = CheckpointSink(drive_root, spec.run_name, staged, identity)
    original_store = training.WindowStore
    original_progress = training.RunProgress
    original_fit = training.fit_final_classifier
    manifest_root = manifest_root_for(windows, repo)

    def portable_store(root):
        return original_store(root, split_root=manifest_root)

    class DurableProgress(original_progress):
        def __call__(self, event, **details):
            if getattr(self, "logging_error", None) and event != "run_failed":
                raise OSError(f"Progress logging failed: {self.logging_error}")
            super().__call__(event, **details)
            if event != "run_failed":
                sink.record(event, **details)

    def fit_and_checkpoint(*args, **kwargs):
        import joblib
        fitted = original_fit(*args, **kwargs)
        joblib.dump(fitted["model"], staged / "model.joblib")
        joblib.dump(fitted["scaler"], staged / "final_scaler.joblib")
        return fitted

    training.WindowStore = portable_store
    training.RunProgress = DurableProgress
    training.fit_final_classifier = fit_and_checkpoint
    try:
        result = training.run_official(windows, repo / "CODE_ctu13", out,
                                       spec.config, spec.experiment,
                                       confirm_full_training=True,
                                       official_authorization=authorized)
    finally:
        training.WindowStore = original_store
        training.RunProgress = original_progress
        training.fit_final_classifier = original_fit
    sink.local_run = out
    published = sink.publish(spec.run_name, EXPECTED[spec.config])
    return {"status": result["status"], "local_run": str(out),
            "drive_result": str(published), "best_k": len(result.get("features", []))
            if spec.experiment == "baseline_svm" else result.get("selection_measurements", {}).get("best_k"),
            "test_evaluation_calls": result["test_evaluation_calls"]}
