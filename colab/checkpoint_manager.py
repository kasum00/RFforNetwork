"""Best-effort durable Drive snapshots; checkpoints are not resumable models."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verified_copy(source: Path, target: Path) -> None:
    if target.exists():
        if _hash(source) == _hash(target):
            return
        raise FileExistsError(f"Existing checkpoint differs: {target}")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Partial checkpoint requires manual inspection: {temporary}")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if _hash(source) != _hash(temporary):
        raise IOError(f"Checkpoint copy SHA256 mismatch: {target}")
    os.replace(temporary, target)


class CheckpointSink:
    """Each completed event is persisted before the next fit begins."""

    def __init__(self, drive_root: Path, run_name: str, local_run: Path, identity: dict):
        if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
            raise ValueError("Run name must contain only letters, digits, _ and -")
        self.local_run = local_run
        self.identity = identity
        self.drive_root = drive_root
        self.dir = drive_root / "checkpoints" / (run_name + ".incomplete")
        if self.dir.exists() or (drive_root / "experiment_results" / run_name).exists():
            raise FileExistsError("An existing completed/incomplete run will never be overwritten")
        self.dir.mkdir(parents=True)
        self.completed_k = []
        self.records = []
        self._write_state("initialized")

    def _write_state(self, stage: str) -> None:
        _atomic_json(self.dir / "checkpoint.json", {
            **self.identity, "stage": stage, "completed_k": self.completed_k,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "resume_supported": False, "checkpoint_is_official_result": False,
            "durability": "copy+SHA256+staging rename on mounted Drive; Drive FUSE atomicity is not proven",
        })

    def record(self, event: str, **details) -> None:
        if event == "k_completed":
            k = int(details["k"])
            if k != len(self.completed_k) + 1:
                raise ValueError("Non-contiguous k completion cannot be checkpointed")
            path = self.dir / f"k_{k:03d}.json"
            _atomic_json(path, {**self.identity, "k": k, "result": details["result"],
                                "best_k_so_far": details["best_k"],
                                "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            self.completed_k.append(k)
        frozen_names = (("selection_frozen.json",) if self.identity["experiment"] == "baseline_svm"
                        else ("selection_frozen.json", "k_search_results.json"))
        for stage, names in {
            "mi_completed": ("mi_feature_scores.csv", "mi_top100.json"),
            "rfe_completed": ("rfe_feature_ranking.csv",),
            "selection_frozen": frozen_names,
            "final_refit_completed": ("model.joblib", "final_scaler.joblib"),
            "run_validated": ("result.json", "features.json", "preflight_audit.json",
                              "events.jsonl", "progress.json"),
        }.items():
            if event == stage:
                required = names + (("authorization.json", "hyperparameters.json",
                                     "feature_schema.json")
                                    if self.identity["experiment"] in ("baseline_svm", "mi_rfe_svm")
                                    else ())
                for name in required:
                    source = self.local_run / name
                    if not source.is_file():
                        raise FileNotFoundError(f"Completed stage lacks required artifact: {source}")
                    _verified_copy(source, self.dir / name)
        if event in {"mi_completed", "rfe_completed", "k_completed", "selection_frozen",
                     "final_refit_completed", "test_evaluation_completed", "run_validated"}:
            self._write_state(event)

    def publish(self, run_name: str, expected: dict) -> Path:
        result = json.loads((self.local_run / "result.json").read_text(encoding="utf-8"))
        if result.get("status") not in ("PASS", "COMPLETED_WITH_CONVERGENCE_WARNING"):
            raise ValueError("No official completed result to publish")
        if result.get("rows_by_split") != expected or result.get("test_evaluation_calls") != 1:
            raise ValueError("Result counts or Test evaluation calls mismatch")
        if result.get("experiment") != self.identity["experiment"] or result.get("config") != self.identity["config"]:
            raise ValueError("Experiment identity mismatch")
        if result.get("experiment") != "baseline_svm" and self.completed_k != list(range(1, 101)):
            raise ValueError("K grid incomplete")
        destination = self.drive_root / "experiment_results" / run_name
        staged = destination.with_name(destination.name + ".incomplete")
        if destination.exists() or staged.exists():
            raise FileExistsError("Result destination already exists")
        staged.mkdir(parents=True)
        names = ("preflight_audit.json", "events.jsonl", "progress.json",
                 "selection_frozen.json", "features.json", "model.joblib",
                 "final_scaler.joblib", "result.json")
        if result["experiment"] in ("baseline_svm", "mi_rfe_svm"):
            names += ("authorization.json", "hyperparameters.json",
                      "feature_schema.json")
        if result["experiment"] != "baseline_svm":
            names += ("mi_feature_scores.csv", "mi_top100.json",
                      "rfe_feature_ranking.csv", "k_search_results.json")
        manifest = {}
        for name in names:
            _verified_copy(self.local_run / name, staged / name)
            manifest[name] = {"bytes": (staged / name).stat().st_size,
                              "sha256": _hash(staged / name)}
        _atomic_json(staged / "artifact_manifest.json", {
            **self.identity, "files": manifest,
            "matrix_cache_saved": False,
            "window_csvs_remain_in_approved_dataset": True,
        })
        os.replace(staged, destination)
        return destination


def publish_completed_checkpoint(drive_root: Path, run_name: str, expected: dict) -> Path:
    """Recover only a fully validated result, never resume a partial model fit."""
    source = drive_root / "checkpoints" / (run_name + ".incomplete")
    state = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if state.get("stage") != "run_validated" or result.get("status") not in (
            "PASS", "COMPLETED_WITH_CONVERGENCE_WARNING"):
        raise ValueError("Checkpoint is not a completed official run")
    if (result.get("rows_by_split") != expected or result.get("test_evaluation_calls") != 1
            or result.get("experiment") != state.get("experiment")
            or result.get("config") != state.get("config")):
        raise ValueError("Checkpoint result identity/counts/Test calls mismatch")
    if result["experiment"] != "baseline_svm" and state.get("completed_k") != list(range(1, 101)):
        raise ValueError("Checkpoint k grid incomplete")
    events = [json.loads(line) for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    names = [item.get("event") for item in events]
    if names.count("test_evaluation_started") != 1 or names.count("test_evaluation_completed") != 1:
        raise ValueError("Test evaluation event count is not exactly one")
    if not (names.index("selection_frozen") < names.index("final_refit_completed")
            < names.index("test_evaluation_started") < names.index("test_evaluation_completed")
            < names.index("run_validated")):
        raise ValueError("Selection/refit/Test event order invalid")
    files = ["preflight_audit.json", "events.jsonl", "progress.json", "selection_frozen.json",
             "features.json", "model.joblib", "final_scaler.joblib", "result.json"]
    if result["experiment"] in ("baseline_svm", "mi_rfe_svm"):
        files += ["authorization.json", "hyperparameters.json", "feature_schema.json"]
    if result["experiment"] != "baseline_svm":
        files += ["mi_feature_scores.csv", "mi_top100.json", "rfe_feature_ranking.csv",
                  "k_search_results.json"]
    destination = drive_root / "experiment_results" / run_name
    staged = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staged.exists():
        raise FileExistsError("Result already exists or incomplete publication needs inspection")
    staged.mkdir(parents=True)
    manifest = {}
    for name in files:
        _verified_copy(source / name, staged / name)
        manifest[name] = {"bytes": (staged / name).stat().st_size,
                          "sha256": _hash(staged / name)}
    _atomic_json(staged / "artifact_manifest.json", {
        "recovered_publication_from_completed_checkpoint": True,
        "no_training_or_test_evaluation_performed": True,
        "identity": {k: state.get(k) for k in ("experiment", "config", "run_name",
                                               "dataset_fingerprint_sha256", "training_code_sha256")},
        "files": manifest, "matrix_cache_saved": False})
    os.replace(staged, destination)
    return destination
