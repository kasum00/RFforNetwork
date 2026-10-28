"""Small, dataset-free tests for Stage 8 CUDA checkpoint/resume controls.

These tests use a fake Booster/model.  They do not read Train/Validation/Test
CSV files and do not perform real model training.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from stage8_xgboost.src import checkpoint_manager as checkpoint


class FakeBooster:
    def __init__(self, rounds: int = 0) -> None:
        self.rounds = rounds

    def num_boosted_rounds(self) -> int:
        return self.rounds

    def save_model(self, path: str) -> None:
        Path(path).write_text(json.dumps({"rounds": self.rounds}), encoding="utf-8")

    def load_model(self, path: str) -> None:
        self.rounds = int(json.loads(Path(path).read_text(encoding="utf-8"))["rounds"])


class FakeClassifier:
    def __init__(self, *, fail_on_fit_call: int | None = None) -> None:
        self.n_estimators = 200
        self.booster = FakeBooster()
        self.fit_calls = 0
        self.fail_on_fit_call = fail_on_fit_call

    def set_params(self, **params):
        self.n_estimators = int(params["n_estimators"])
        return self

    def fit(self, X, y, *, xgb_model=None, **kwargs):
        del X, y, kwargs
        self.fit_calls += 1
        if self.fail_on_fit_call == self.fit_calls:
            raise RuntimeError("simulated interruption")
        prior = 0 if xgb_model is None else int(xgb_model.num_boosted_rounds())
        self.booster = FakeBooster(prior + self.n_estimators)
        return self

    def get_booster(self) -> FakeBooster:
        return self.booster

    def save_model(self, path: str) -> None:
        self.booster.save_model(path)

    def load_model(self, path: str) -> None:
        self.booster = FakeBooster()
        self.booster.load_model(path)


def make_context(root: Path, **changes) -> checkpoint.CheckpointContext:
    values = {
        "checkpoint_root": root,
        "run_name": "stage8_baseline_run01",
        "run_id": "stage8_baseline_run01",
        "experiment": "baseline_xgb",
        "config": "FW_512_FS_16",
        "backend": "cpu",
        "xgboost_version": "3.4.1",
        "git_commit": "a" * 40,
        "source_hashes": {
            "stage8_sha256": "A" * 64,
            "training_core_sha256": "B" * 64,
            "checkpoint_module_sha256": "C" * 64,
            "original_ctu13_sha256": {"one.py": "D" * 64},
        },
        "dataset_identity": {
            "window_summary_sha256": "E" * 64,
            "window_snapshot_size_mtime_sha256": "F" * 64,
        },
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_child_weight": 3,
            "tree_method": "hist",
            "device": "cpu",
        },
        "feature_names": ("f0", "f1", "f2"),
        "selected_features": ("f0", "f1", "f2"),
        "checkpoint_interval_rounds": 10,
        "total_rounds": 200,
    }
    values.update(changes)
    return checkpoint.CheckpointContext(**values)


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_xgboost = types.SimpleNamespace(Booster=FakeBooster)

    def test_fresh_fit_publishes_every_interval_and_keeps_last_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            context = make_context(Path(temp))
            manager = checkpoint.CheckpointManager(context)
            model, audit = manager.fit_stage(
                model=FakeClassifier(),
                X=object(),
                y=object(),
                stage="validation_fit",
                resume=False,
            )
            self.assertEqual(model.get_booster().num_boosted_rounds(), 200)
            self.assertEqual(audit["completed_rounds"], 200)
            self.assertEqual(audit["remaining_rounds"], 0)
            self.assertEqual(audit["total_rounds"], 200)
            manifests = sorted(context.stage_root("validation_fit").glob("*.manifest.json"))
            models = sorted(context.stage_root("validation_fit").glob("*.ubj"))
            self.assertEqual([item.name for item in manifests], [
                "round_000190.manifest.json",
                "round_000200.manifest.json",
            ])
            self.assertEqual(len(models), 2)

    def test_interrupted_fit_resumes_from_latest_verified_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            context = make_context(Path(temp))
            manager = checkpoint.CheckpointManager(context)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                manager.fit_stage(
                    model=FakeClassifier(fail_on_fit_call=3),
                    X=object(),
                    y=object(),
                    stage="validation_fit",
                    resume=False,
                )
            resumed, audit = manager.fit_stage(
                model=FakeClassifier(),
                X=object(),
                y=object(),
                stage="validation_fit",
                resume=True,
            )
            self.assertTrue(audit["resume_accepted"])
            self.assertEqual(audit["resume_from_round"], 20)
            self.assertEqual(audit["completed_rounds"], 200)
            self.assertEqual(audit["fit_calls_this_process"], 18)
            self.assertEqual(resumed.get_booster().num_boosted_rounds(), 200)

    def test_corrupt_latest_checkpoint_fails_closed_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            context = make_context(Path(temp))
            manager = checkpoint.CheckpointManager(context)
            manager.fit_stage(
                model=FakeClassifier(),
                X=object(),
                y=object(),
                stage="validation_fit",
                resume=False,
            )
            latest = context.stage_root("validation_fit") / "round_000200.ubj"
            latest.write_bytes(b"corrupt")
            with self.assertRaisesRegex(
                checkpoint.CheckpointIntegrityError, "checksum mismatch"
            ):
                manager.latest(stage="validation_fit")

    def test_resume_without_checkpoint_and_missing_manifest_metadata_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            context = make_context(Path(temp))
            manager = checkpoint.CheckpointManager(context)
            with self.assertRaises(checkpoint.CheckpointIntegrityError):
                manager.fit_stage(
                    model=FakeClassifier(),
                    X=object(),
                    y=object(),
                    stage="validation_fit",
                    resume=True,
                )
            manager.fit_stage(
                model=FakeClassifier(),
                X=object(),
                y=object(),
                stage="validation_fit",
                resume=False,
            )
            manifest_path = (
                context.stage_root("validation_fit")
                / "round_000200.manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("backend")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(checkpoint.CheckpointCompatibilityError):
                manager.latest(stage="validation_fit")

    def test_identity_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            base = make_context(Path(temp))
            checkpoint.CheckpointManager(base).fit_stage(
                model=FakeClassifier(),
                X=object(),
                y=object(),
                stage="validation_fit",
                resume=False,
            )
            variants = {
                "backend": replace(base, backend="cuda"),
                "config": replace(base, config="FW_256_FS_16"),
                "xgboost_version": replace(base, xgboost_version="3.4.0"),
                "source_hash": replace(
                    base, source_hashes={**base.source_hashes, "stage8_sha256": "0" * 64}
                ),
                "dataset": replace(
                    base,
                    dataset_identity={
                        **base.dataset_identity,
                        "window_summary_sha256": "0" * 64,
                    },
                ),
                "hyperparameters": replace(
                    base, hyperparameters={**base.hyperparameters, "max_depth": 9}
                ),
                "feature_order": replace(base, feature_names=("f2", "f1", "f0")),
            }
            for name, context in variants.items():
                with self.subTest(name=name), self.assertRaises(
                    checkpoint.CheckpointCompatibilityError
                ):
                    checkpoint.CheckpointManager(context).latest(
                        stage="validation_fit"
                    )

    def test_validation_and_final_refit_are_separate_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"xgboost": self.fake_xgboost}
        ):
            context = make_context(Path(temp))
            manager = checkpoint.CheckpointManager(context)
            for stage in ("validation_fit", "final_refit"):
                manager.fit_stage(
                    model=FakeClassifier(),
                    X=object(),
                    y=object(),
                    stage=stage,
                    resume=False,
                )
            self.assertNotEqual(
                manager.latest(stage="validation_fit").model_path,
                manager.latest(stage="final_refit").model_path,
            )
            self.assertEqual(
                manager.latest(stage="final_refit").completed_rounds, 200
            )


class ProtocolStateTests(unittest.TestCase):
    def test_ambiguous_test_state_blocks_resume_and_second_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            context = make_context(Path(temp))
            store = checkpoint.ProtocolStateStore(context)
            store.create_fresh()
            store.mark_stage("validation_fit", "STAGE_COMPLETE")
            store.mark_stage("final_refit", "STAGE_COMPLETE")
            store.request_test_once()
            with self.assertRaises(checkpoint.AmbiguousTestStateError):
                store.load_for_resume()
            with self.assertRaises(checkpoint.AmbiguousTestStateError):
                store.request_test_once()

    def test_completed_run_is_immutable_and_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            context = make_context(Path(temp))
            store = checkpoint.ProtocolStateStore(context)
            store.create_fresh()
            store.mark_stage("validation_fit", "STAGE_COMPLETE")
            store.mark_stage("final_refit", "STAGE_COMPLETE")
            store.request_test_once()
            state = store.complete_test_once(metrics_sha256="A" * 64)
            self.assertEqual(state["test_evaluation_calls"], 1)
            with self.assertRaises(checkpoint.CheckpointCompatibilityError):
                store.load_for_resume()
            with self.assertRaises(checkpoint.CheckpointCompatibilityError):
                store.update(extra="forbidden")

    def test_nonfinite_json_does_not_damage_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            checkpoint._atomic_write_json(path, {"ok": 1}, exclusive=True)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                checkpoint._atomic_write_json(
                    path, {"bad": math.nan}, exclusive=False
                )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": 1})



if __name__ == "__main__":
    unittest.main()
