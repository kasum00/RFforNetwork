"""Crash-safe XGBoost checkpoint and resume support for Stage 8.

This module is intentionally independent from the dataset loader.  It never
loads Train, Validation, or Test data by itself.  The caller supplies the
already-approved matrices and an ``XGBClassifier`` configured by the Stage 8
runner.

The manifest is the publication point of a checkpoint: a model is first saved
to a temporary UBJ file, loaded back and verified, atomically renamed, and only
then is its strict-JSON manifest atomically published.  A manifest without a
matching, checksum-valid model is never accepted for resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEMA = "stage8.xgboost.checkpoint.v1"
PROTOCOL_STATE_SCHEMA = "stage8.xgboost.protocol-state.v1"
ALLOWED_STAGES = ("validation_fit", "final_refit")
TEST_NOT_STARTED = "NOT_STARTED"
TEST_EVALUATION_REQUESTED = "TEST_EVALUATION_REQUESTED"
TEST_COMPLETED = "TEST_COMPLETED"
TEST_STATE_UNKNOWN = "TEST_STATE_UNKNOWN"


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures that must block training."""


class CheckpointCompatibilityError(CheckpointError):
    """The checkpoint identity differs from the requested official run."""


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint or protocol-state file is missing or corrupt."""


class AmbiguousTestStateError(CheckpointError):
    """Test may already have been read/evaluated, so re-evaluation is blocked."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_non_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite float at {location}: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{location}[{index}]")


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    _reject_non_finite(payload)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    """Validate/serialize before opening the destination, then atomically publish."""

    encoded = _strict_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite existing JSON: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"refusing to overwrite existing JSON: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {token}")


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise CheckpointIntegrityError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError(f"JSON root must be an object: {path}")
    _reject_non_finite(payload)
    return payload


def _safe_identifier(value: str, field: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def _normalise_json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible identity value without coercion."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        _reject_non_finite(value)
        return value
    raise TypeError(f"unsupported checkpoint identity value: {type(value).__name__}")


@dataclass(frozen=True)
class CheckpointContext:
    checkpoint_root: Path
    run_name: str
    run_id: str
    experiment: str
    config: str
    backend: str
    xgboost_version: str
    git_commit: str
    source_hashes: Mapping[str, str]
    dataset_identity: Mapping[str, Any]
    hyperparameters: Mapping[str, Any]
    feature_names: Sequence[str]
    selected_features: Sequence[str]
    checkpoint_interval_rounds: int = 10
    total_rounds: int = 200

    def __post_init__(self) -> None:
        _safe_identifier(self.run_name, "run_name")
        _safe_identifier(self.run_id, "run_id")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.git_commit or "") is None:
            raise ValueError("git_commit must be an exact 40-hex commit")
        if self.experiment not in {"baseline_xgb", "mi_rfe_xgb"}:
            raise ValueError(f"unsupported experiment: {self.experiment}")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.checkpoint_interval_rounds <= 0:
            raise ValueError("checkpoint_interval_rounds must be positive")
        if self.total_rounds != 200:
            raise ValueError("official Stage 8 total_rounds must remain exactly 200")
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if not self.selected_features:
            raise ValueError("selected_features must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names contain duplicates")
        if len(set(self.selected_features)) != len(self.selected_features):
            raise ValueError("selected_features contain duplicates")
        missing = set(self.selected_features) - set(self.feature_names)
        if missing:
            raise ValueError(f"selected features missing from feature order: {sorted(missing)[:5]}")
        _strict_json_bytes(self.compatibility_identity())

    @property
    def run_root(self) -> Path:
        return Path(self.checkpoint_root).resolve() / self.run_id

    def stage_root(self, stage: str) -> Path:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"unsupported checkpoint stage: {stage}")
        return self.run_root / stage

    def compatibility_identity(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "config": self.config,
            "backend": self.backend,
            "xgboost_version": self.xgboost_version,
            "git_commit": self.git_commit,
            "source_hashes": _normalise_json_value(self.source_hashes),
            "dataset_identity": _normalise_json_value(self.dataset_identity),
            "hyperparameters": _normalise_json_value(self.hyperparameters),
            "feature_names": list(self.feature_names),
            "selected_features": list(self.selected_features),
            "checkpoint_interval_rounds": self.checkpoint_interval_rounds,
            "total_rounds": self.total_rounds,
        }


@dataclass(frozen=True)
class VerifiedCheckpoint:
    manifest_path: Path
    model_path: Path
    manifest: Mapping[str, Any]
    completed_rounds: int


class CheckpointManager:
    def __init__(self, context: CheckpointContext) -> None:
        self.context = context

    def _manifest_path(self, stage: str, completed_rounds: int) -> Path:
        return self.context.stage_root(stage) / f"round_{completed_rounds:06d}.manifest.json"

    def _model_path(self, stage: str, completed_rounds: int) -> Path:
        return self.context.stage_root(stage) / f"round_{completed_rounds:06d}.ubj"

    @staticmethod
    def _booster_rounds(booster: Any) -> int:
        rounds = int(booster.num_boosted_rounds())
        if rounds < 0:
            raise CheckpointIntegrityError(f"invalid boosted round count: {rounds}")
        return rounds

    def publish(self, *, stage: str, model: Any, completed_rounds: int) -> VerifiedCheckpoint:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"unsupported checkpoint stage: {stage}")
        if not 0 < completed_rounds <= self.context.total_rounds:
            raise ValueError(f"invalid completed_rounds: {completed_rounds}")

        stage_root = self.context.stage_root(stage)
        stage_root.mkdir(parents=True, exist_ok=True)
        model_path = self._model_path(stage, completed_rounds)
        manifest_path = self._manifest_path(stage, completed_rounds)
        if model_path.exists() or manifest_path.exists():
            raise FileExistsError(
                f"refusing to overwrite checkpoint round {completed_rounds} for {stage}"
            )

        temporary_model = stage_root / f".round_{completed_rounds:06d}.{uuid.uuid4().hex}.tmp.ubj"
        booster = model.get_booster()
        actual_rounds = self._booster_rounds(booster)
        if actual_rounds != completed_rounds:
            raise CheckpointIntegrityError(
                f"model has {actual_rounds} rounds, expected {completed_rounds}"
            )

        try:
            # Save through the sklearn wrapper so resume restores classifier
            # metadata as well as the underlying Booster.
            model.save_model(str(temporary_model))
            # Windows rejects fsync() on a read-only descriptor with EBADF.
            # Open the already-written model read/write solely so its bytes
            # can be durably flushed before the atomic rename; this does not
            # modify the checkpoint payload.
            with temporary_model.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())

            import xgboost as xgb

            verified_booster = xgb.Booster()
            verified_booster.load_model(str(temporary_model))
            verified_rounds = self._booster_rounds(verified_booster)
            if verified_rounds != completed_rounds:
                raise CheckpointIntegrityError(
                    f"reloaded model has {verified_rounds} rounds, expected {completed_rounds}"
                )
            checksum = _sha256_file(temporary_model)
            os.replace(temporary_model, model_path)

            manifest = {
                "schema": CHECKPOINT_SCHEMA,
                **self.context.compatibility_identity(),
                "stage": stage,
                "completed_rounds": completed_rounds,
                "remaining_rounds": self.context.total_rounds - completed_rounds,
                "total_rounds": self.context.total_rounds,
                "status": "STAGE_COMPLETE"
                if completed_rounds == self.context.total_rounds
                else "IN_PROGRESS",
                "model_file": model_path.name,
                "checkpoint_sha256": checksum,
                "published_at_utc": _now_utc(),
            }
            _atomic_write_json(manifest_path, manifest, exclusive=True)
        finally:
            if temporary_model.exists():
                temporary_model.unlink()

        self._prune(stage, keep=2)
        return self.verify_manifest(manifest_path, stage=stage)

    def _prune(self, stage: str, *, keep: int) -> None:
        manifests = sorted(self.context.stage_root(stage).glob("round_*.manifest.json"))
        for manifest_path in manifests[:-keep]:
            manifest = _read_strict_json(manifest_path)
            model_name = manifest.get("model_file")
            if isinstance(model_name, str):
                model_path = manifest_path.parent / model_name
                if model_path.exists():
                    model_path.unlink()
            manifest_path.unlink()

    def verify_manifest(self, manifest_path: Path, *, stage: str) -> VerifiedCheckpoint:
        manifest_path = Path(manifest_path).resolve()
        expected_parent = self.context.stage_root(stage).resolve()
        if manifest_path.parent != expected_parent:
            raise CheckpointIntegrityError("checkpoint manifest escaped its approved stage root")
        manifest = _read_strict_json(manifest_path)
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointCompatibilityError("checkpoint schema mismatch")
        if manifest.get("stage") != stage:
            raise CheckpointCompatibilityError("checkpoint stage mismatch")

        expected = self.context.compatibility_identity()
        mismatches = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if mismatches:
            raise CheckpointCompatibilityError(
                "checkpoint identity mismatch: " + ", ".join(sorted(mismatches))
            )

        completed_rounds = manifest.get("completed_rounds")
        if not isinstance(completed_rounds, int) or isinstance(completed_rounds, bool):
            raise CheckpointIntegrityError("completed_rounds must be an integer")
        if not 0 < completed_rounds <= self.context.total_rounds:
            raise CheckpointIntegrityError("completed_rounds is outside the official range")
        if manifest.get("remaining_rounds") != self.context.total_rounds - completed_rounds:
            raise CheckpointIntegrityError("remaining_rounds is inconsistent")
        expected_status = (
            "STAGE_COMPLETE"
            if completed_rounds == self.context.total_rounds
            else "IN_PROGRESS"
        )
        if manifest.get("status") != expected_status:
            raise CheckpointIntegrityError("checkpoint status is inconsistent")

        model_name = manifest.get("model_file")
        if not isinstance(model_name, str) or Path(model_name).name != model_name:
            raise CheckpointIntegrityError("unsafe or missing model_file")
        model_path = (manifest_path.parent / model_name).resolve()
        if model_path.parent != expected_parent or not model_path.is_file():
            raise CheckpointIntegrityError("checkpoint model is missing or outside its stage root")
        checksum = manifest.get("checkpoint_sha256")
        if not isinstance(checksum, str) or _sha256_file(model_path) != checksum:
            raise CheckpointIntegrityError("checkpoint model checksum mismatch")

        import xgboost as xgb

        booster = xgb.Booster()
        try:
            booster.load_model(str(model_path))
        except Exception as exc:
            raise CheckpointIntegrityError(f"checkpoint model cannot be loaded: {exc}") from exc
        if self._booster_rounds(booster) != completed_rounds:
            raise CheckpointIntegrityError("checkpoint model round count mismatch")
        return VerifiedCheckpoint(
            manifest_path=manifest_path,
            model_path=model_path,
            manifest=manifest,
            completed_rounds=completed_rounds,
        )

    def latest(self, *, stage: str) -> VerifiedCheckpoint:
        stage_root = self.context.stage_root(stage)
        if not stage_root.is_dir():
            raise CheckpointIntegrityError(f"no checkpoint directory for stage {stage}")
        manifests = sorted(stage_root.glob("round_*.manifest.json"), reverse=True)
        published_models = set()
        for manifest_path in manifests:
            manifest = _read_strict_json(manifest_path)
            model_name = manifest.get("model_file")
            if isinstance(model_name, str):
                published_models.add(model_name)
        orphan_models = [
            model_path
            for model_path in stage_root.glob("round_*.ubj")
            if model_path.name not in published_models
        ]
        if orphan_models:
            raise CheckpointIntegrityError(
                "unpublished/orphan checkpoint model has no manifest: "
                + ", ".join(sorted(item.name for item in orphan_models))
            )
        if not manifests:
            raise CheckpointIntegrityError(f"no published checkpoint for stage {stage}")
        # Never silently fall back from a corrupt newest checkpoint to an older one.
        return self.verify_manifest(manifests[0], stage=stage)

    def fit_stage(
        self,
        *,
        model: Any,
        X: Any,
        y: Any,
        stage: str,
        resume: bool,
        fit_kwargs: Mapping[str, Any] | None = None,
        progress: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"unsupported checkpoint stage: {stage}")
        fit_kwargs = dict(fit_kwargs or {})
        if "xgb_model" in fit_kwargs:
            raise ValueError("xgb_model is managed exclusively by CheckpointManager")
        if self.context.experiment != "baseline_xgb":
            raise CheckpointCompatibilityError(
                "checkpointed official execution is currently approved only for baseline_xgb"
            )

        manager_stage_root = self.context.stage_root(stage)
        resume_from: VerifiedCheckpoint | None = None
        completed = 0
        booster = None
        if resume:
            resume_from = self.latest(stage=stage)
            model.load_model(str(resume_from.model_path))
            booster = model.get_booster()
            completed = resume_from.completed_rounds
        elif manager_stage_root.exists() and any(manager_stage_root.iterdir()):
            raise CheckpointIntegrityError(
                f"checkpoint data already exists for {stage}; set RESUME=True or use a new run ID"
            )

        started = time.perf_counter()
        published: VerifiedCheckpoint | None = resume_from
        fit_calls = 0
        while completed < self.context.total_rounds:
            chunk = min(
                self.context.checkpoint_interval_rounds,
                self.context.total_rounds - completed,
            )
            model.set_params(n_estimators=chunk)
            model.fit(X, y, xgb_model=booster, **fit_kwargs)
            fit_calls += 1
            booster = model.get_booster()
            completed = int(booster.num_boosted_rounds())
            if completed > self.context.total_rounds:
                raise CheckpointIntegrityError(
                    f"continued model exceeded 200 rounds: {completed}"
                )
            published = self.publish(
                stage=stage,
                model=model,
                completed_rounds=completed,
            )
            if progress is not None:
                progress(
                    "xgboost_checkpoint_published",
                    resume_supported=True,
                    resume_requested=bool(resume),
                    resume_accepted=bool(resume_from is not None),
                    checkpoint_root=str(self.context.run_root),
                    checkpoint_interval_rounds=self.context.checkpoint_interval_rounds,
                    checkpoint_verified=True,
                    resume_from_stage=stage if resume_from is not None else None,
                    resume_from_round=(
                        resume_from.completed_rounds if resume_from is not None else 0
                    ),
                    completed_rounds=completed,
                    remaining_rounds=self.context.total_rounds - completed,
                    total_rounds=self.context.total_rounds,
                    checkpoint_sha256=published.manifest["checkpoint_sha256"],
                )

        if published is None or published.completed_rounds != self.context.total_rounds:
            raise CheckpointIntegrityError("official XGBoost stage did not finish exactly 200 rounds")
        model.set_params(n_estimators=self.context.total_rounds)
        elapsed = time.perf_counter() - started
        return model, {
            "resume_supported": True,
            "resume_requested": bool(resume),
            "resume_accepted": bool(resume_from is not None),
            "checkpoint_root": str(self.context.run_root),
            "checkpoint_interval_rounds": self.context.checkpoint_interval_rounds,
            "checkpoint_verified": True,
            "resume_from_stage": stage if resume_from is not None else None,
            "resume_from_round": resume_from.completed_rounds if resume_from else 0,
            "completed_rounds": completed,
            "remaining_rounds": self.context.total_rounds - completed,
            "total_rounds": self.context.total_rounds,
            "checkpoint_sha256": published.manifest["checkpoint_sha256"],
            "fit_calls_this_process": fit_calls,
            "elapsed_seconds_this_process": elapsed,
        }


class ProtocolStateStore:
    """Durable scientific-protocol state, including the one-shot Test guard."""

    def __init__(self, context: CheckpointContext) -> None:
        self.context = context
        self.path = context.run_root / "protocol_state.json"

    def create_fresh(self) -> dict[str, Any]:
        if self.path.exists() or self.context.run_root.exists() and any(
            self.context.run_root.iterdir()
        ):
            raise FileExistsError(
                "run/checkpoint state already exists; a fresh run cannot overwrite it"
            )
        state = {
            "schema": PROTOCOL_STATE_SCHEMA,
            **self.context.compatibility_identity(),
            "official_status": "INCOMPLETE",
            "validation_fit_status": "NOT_STARTED",
            "final_refit_status": "NOT_STARTED",
            "test_state": TEST_NOT_STARTED,
            "test_evaluation_calls": 0,
            "created_at_utc": _now_utc(),
            "updated_at_utc": _now_utc(),
        }
        _atomic_write_json(self.path, state, exclusive=True)
        return state

    def load_for_resume(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise CheckpointIntegrityError("resume requested but protocol_state.json is missing")
        state = _read_strict_json(self.path)
        if state.get("schema") != PROTOCOL_STATE_SCHEMA:
            raise CheckpointCompatibilityError("protocol-state schema mismatch")
        expected = self.context.compatibility_identity()
        mismatches = [key for key, value in expected.items() if state.get(key) != value]
        if mismatches:
            raise CheckpointCompatibilityError(
                "protocol-state identity mismatch: " + ", ".join(sorted(mismatches))
            )
        if state.get("official_status") == "COMPLETED":
            raise CheckpointCompatibilityError("completed official run cannot be resumed or overwritten")
        if state.get("test_state") == TEST_EVALUATION_REQUESTED:
            state["test_state"] = TEST_STATE_UNKNOWN
            state["updated_at_utc"] = _now_utc()
            _atomic_write_json(self.path, state, exclusive=False)
            raise AmbiguousTestStateError(
                "Test had been requested before interruption; state is now "
                "TEST_STATE_UNKNOWN and Test re-evaluation is blocked"
            )
        if state.get("test_state") == TEST_STATE_UNKNOWN:
            raise AmbiguousTestStateError(
                "Test state is ambiguous; Test re-read/re-evaluation is blocked"
            )
        if state.get("test_state") == TEST_COMPLETED:
            raise AmbiguousTestStateError("Test was already evaluated; a second evaluation is blocked")
        return state

    def update(self, **changes: Any) -> dict[str, Any]:
        state = _read_strict_json(self.path)
        if state.get("official_status") == "COMPLETED":
            raise CheckpointCompatibilityError("completed official state is immutable")
        state.update(_normalise_json_value(changes))
        state["updated_at_utc"] = _now_utc()
        _atomic_write_json(self.path, state, exclusive=False)
        return state

    def mark_stage(self, stage: str, status: str) -> dict[str, Any]:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"unsupported stage: {stage}")
        if status not in {"IN_PROGRESS", "STAGE_COMPLETE"}:
            raise ValueError(f"unsupported stage status: {status}")
        return self.update(**{f"{stage}_status": status})

    def request_test_once(self) -> dict[str, Any]:
        state = _read_strict_json(self.path)
        if state.get("final_refit_status") != "STAGE_COMPLETE":
            raise CheckpointCompatibilityError("Test is blocked until final_refit is complete")
        if state.get("test_state") != TEST_NOT_STARTED:
            raise AmbiguousTestStateError(
                f"Test cannot be requested from state {state.get('test_state')!r}"
            )
        # Persist this state before the caller is allowed to read Test rows.
        return self.update(test_state=TEST_EVALUATION_REQUESTED)

    def complete_test_once(self, *, metrics_sha256: str) -> dict[str, Any]:
        state = _read_strict_json(self.path)
        if state.get("test_state") != TEST_EVALUATION_REQUESTED:
            raise AmbiguousTestStateError("Test completion has no durable request record")
        if state.get("test_evaluation_calls") != 0:
            raise AmbiguousTestStateError("Test evaluation call counter is not zero")
        return self.update(
            test_state=TEST_COMPLETED,
            test_evaluation_calls=1,
            test_metrics_sha256=metrics_sha256,
            official_status="COMPLETED",
        )

