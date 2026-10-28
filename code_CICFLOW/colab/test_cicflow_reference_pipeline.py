"""Synthetic regression suite; no real dataset, old run or training job is opened.

Only the user runs this suite. Fixture estimators are deliberately small and
never alter the production scientific parameter dictionaries.
"""
from __future__ import annotations

import json
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from colab import cicflow_reference_pipeline as pipeline
from stage5_rfe.rfe_selection import manual_rfe_step1


class ReferenceAlgorithmTests(unittest.TestCase):
    def test_full_380_feature_mi_matches_one_reference_call_exactly(self):
        rng = np.random.RandomState(7)
        X = rng.normal(size=(28, 380)).astype(np.float32)
        X[:, :3] = 0  # Ties / zero variance exercise pandas ordering and MI noise.
        y = np.tile([0, 1], 14)
        names = [f"feature_{i:03d}" for i in range(380)]
        expected = pd.Series(mutual_info_classif(StandardScaler().fit_transform(X), y, random_state=42),
                             index=names).sort_values(ascending=False)
        with patch("sklearn.feature_selection.mutual_info_classif", wraps=mutual_info_classif) as call:
            actual = pipeline.reference_mi(X, y, names)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs, {"random_state": 42})
        self.assertEqual(call.call_args.args[0].shape, (28, 380))
        pd.testing.assert_series_equal(actual, expected, check_exact=True)

    def test_preprocessing_preserves_rows_and_order(self):
        X, y, counts = pipeline.preprocess([[np.nan, 9], [np.inf, -np.inf], [3, 4]], [0, 1, 0], both_classes=True)
        np.testing.assert_array_equal(X, [[0, 9], [0, 0], [3, 4]])
        np.testing.assert_array_equal(y, [0, 1, 0])
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(counts, {"nan_replaced": 1, "inf_replaced": 2})
        with self.assertRaises(pipeline.ContractError):
            pipeline.preprocess([[1e300]], [0])

    def test_smallest_k_wins_exact_accuracy_tie(self):
        self.assertEqual(pipeline.best_k([{"k": 1, "accuracy": .8}, {"k": 2, "accuracy": .9},
                                        {"k": 3, "accuracy": .9}]), (2, .9))
        with self.assertRaises(pipeline.ContractError):
            pipeline.best_k([{"k": 2, "accuracy": .9}])

    def test_rf_classifier_depth20_is_separate_from_depth10_rfe(self):
        self.assertEqual(pipeline.make_model("rf").get_params()["max_depth"], 20)
        self.assertEqual(pipeline.make_model("rf").get_params()["n_estimators"], 300)
        self.assertEqual(pipeline.RFE_PARAMS["rf"]["max_depth"], 10)
        self.assertEqual(pipeline.RFE_PARAMS["svm"]["max_depth"], 15)
        self.assertEqual(pipeline.make_model("svm").get_params()["max_iter"], 20000)
        self.assertEqual(pipeline.make_model("svm").get_params()["C"], 10.0)

    def test_rfe_ties_and_resumed_elimination_match_sklearn(self):
        rng = np.random.RandomState(1)
        y = np.tile([0, 1], 12)
        for X in (rng.normal(size=(24, 6)), np.zeros((24, 6))):
            estimator = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42, n_jobs=1)
            expected = RFE(estimator, n_features_to_select=1, step=1).fit(X, y)
            recorded = []

            class StopAfterThree(Exception):
                pass

            def checkpoint(state):
                recorded.append(state)
                if state["completed_steps"] == 3:
                    raise StopAfterThree()

            with self.assertRaises(StopAfterThree):
                manual_rfe_step1(X, y, estimator, checkpoint_callback=checkpoint)
            support, ranking = pipeline.validate_rfe_states(recorded, count=6)
            actual, supported, count = manual_rfe_step1(X, y, estimator, initial_support=support,
                initial_ranking=ranking, completed_steps=3)
            np.testing.assert_array_equal(actual, expected.ranking_)
            np.testing.assert_array_equal(supported, expected.support_)
            self.assertEqual(count, 5)

    def test_final_scaler_matches_reference_c_order_train_validation_refit(self):
        import joblib
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.RandomState(11)
        train = np.asarray(rng.normal(size=(22, 5)), dtype=np.float32, order="C")
        val = np.asarray(rng.normal(size=(10, 5)), dtype=np.float32, order="C")
        names = [f"f{i}" for i in range(5)]
        frozen = {"features": ["f4", "f0", "f2"]}
        expected = np.empty((32, 3), dtype=np.float32, order="C")
        expected[:22] = train[:, [4, 0, 2]]
        expected[22:] = val[:, [4, 0, 2]]
        expected_scaler = StandardScaler().fit(expected)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "make_model", return_value=DecisionTreeClassifier(max_depth=2, random_state=42)):
            pipeline.final_refit(Path(tmp), {"sha256": "fixture"}, frozen,
                train, np.tile([0, 1], 11), val, np.tile([0, 1], 5), names, "rf", unittest.mock.Mock())
            actual = joblib.load(Path(tmp) / "final_scaler.joblib")
            np.testing.assert_array_equal(actual.mean_, expected_scaler.mean_)
            np.testing.assert_array_equal(actual.var_, expected_scaler.var_)
            np.testing.assert_array_equal(actual.scale_, expected_scaler.scale_)


class GuardAndCheckpointTests(unittest.TestCase):
    def test_source_mi_and_rfe_names_are_not_reused_as_legacy_run(self):
        self.assertNotEqual(pipeline.FORMAT, "cicflow-colab-v2.1")
        with self.assertRaises(pipeline.ContractError):
            pipeline.RunSpec("mi_rfe_rf", "FW256_FS8", "RUN_MI_OLD").validate()
        with self.assertRaises(pipeline.ContractError):
            pipeline.RunSpec("mi_rfe_xgboost", "FW256_FS8", "unused").validate()

    def test_selection_never_calls_test_reader(self):
        reader = unittest.mock.Mock()
        with self.assertRaises(pipeline.ContractError):
            pipeline.read_values({}, "test", selection=True, reader=reader)
        reader.assert_not_called()
        with self.assertRaises(pipeline.ContractError):
            pipeline.read_values({}, "test", reader=reader)
        reader.assert_not_called()

    def test_requested_train_order_is_used(self):
        import pyarrow as pa
        names = ["second", "first"]
        table = pa.table({"second": [3., 4.], "first": [1., 2.], "Window_Label": [0, 1]})
        audit = {"binding": {"names": ["first", "second"], "splits": {"train": {"path": "TRAIN_ONLY", "rows": 2}}}}
        reader = unittest.mock.Mock(return_value=table)
        X, _, _ = pipeline.read_values(audit, "train", names=names, selection=True, reader=reader)
        reader.assert_called_once_with("TRAIN_ONLY", columns=["second", "first", "Window_Label"])
        np.testing.assert_array_equal(X, [[3, 1], [4, 2]])

    def test_chain_rejects_changed_identity_hash_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            binding = {"identity": "one", "mi_state": "bound"}
            previous = pipeline.save_chain(directory, "rfe", 1, binding, {"value": 1}, None)
            pipeline.save_chain(directory, "rfe", 2, binding, {"value": 2}, previous)
            rows, _ = pipeline.load_chain(directory, "rfe", binding)
            self.assertEqual(len(rows), 2)
            with self.assertRaises(pipeline.ContractError):
                pipeline.load_chain(directory, "rfe", {"identity": "two"})
            path = directory / "rfe_002.json"
            changed = pipeline.read_json(path)
            changed["state"]["value"] = 99
            path.write_text(json.dumps(changed))
            with self.assertRaises(pipeline.ContractError):
                pipeline.load_chain(directory, "rfe", binding)

    def test_exclusive_writer_and_partial_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            pipeline.write_once(path, {"old": True})
            with self.assertRaises(FileExistsError):
                pipeline.write_once(path, {"old": False})
            self.assertEqual(pipeline.read_json(path), {"old": True})
            (Path(tmp) / "other.json.partial").write_text("partial")
            with self.assertRaises(FileExistsError):
                pipeline.write_once(Path(tmp) / "other.json", {})

    def test_resource_failure_does_not_reduce_parameters(self):
        before = pipeline.contract("rf")
        result = pipeline.resource_gate({"train": 326157, "val": 104147}, 1, 1, "rf")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(pipeline.contract("rf"), before)

    def test_test_intent_requires_final_refit_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            identity = {"sha256": "fixture"}
            with self.assertRaises(FileNotFoundError):
                pipeline.start_test_once(directory, identity)
            frozen = {"identity": "fixture", "features": ["feature"]}
            pipeline.write_once(directory / "selection_frozen.json", frozen)
            for name in ("model.joblib", "final_scaler.joblib"):
                (directory / name).write_bytes(b"synthetic model bytes")
            final = {"binding": {"identity": "fixture", "frozen": pipeline.digest(frozen)},
                     "files": {name: pipeline.artifact_info(directory / name)
                               for name in ("model.joblib", "final_scaler.joblib")}}
            pipeline.write_once(directory / "final_state.json", final)
            pipeline.start_test_once(directory, identity)
            with self.assertRaises(FileExistsError):
                pipeline.start_test_once(directory, identity)

    def test_dry_run_never_reaches_fit(self):
        with patch.object(pipeline, "preflight") as gate, patch.object(pipeline, "make_model") as fit:
            with self.assertRaises(pipeline.ContractError):
                pipeline.execute_run({"mode": "dry_run"})
            gate.assert_not_called()
            fit.assert_not_called()

    def test_redaction(self):
        value = pipeline.protect("token=hidden password=hidden Authorization: Bearer private")
        self.assertNotIn("hidden", value)
        self.assertNotIn("private", value)

    def test_k_resume_keeps_completed_k_and_exact_smallest_k_tie_rule(self):
        from sklearn.dummy import DummyClassifier
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            top = [f"f{i}" for i in range(100)]
            identity, rfe = {"sha256": "fixture"}, {"names": list(reversed(top))}
            binding = {"identity": "fixture", "rfe_state": pipeline.digest(rfe)}
            pipeline.save_chain(directory / "steps", "k", 1, binding,
                                {"k": 1, "accuracy": .5, "fit": {}}, None)
            X, y = np.zeros((6, 100)), np.array([0, 1, 0, 1, 0, 1])
            progress = unittest.mock.Mock()
            with patch.object(pipeline, "make_model", side_effect=lambda branch: DummyClassifier(strategy="most_frequent")) as factory:
                result = pipeline.search_k(directory, identity, rfe, X, y, X, y, top, rfe["names"], "rf", progress)
                self.assertEqual(factory.call_count, 99)
                self.assertEqual(result["best_k"], 1)
                self.assertEqual(len(result["results"]), 100)
                again = pipeline.search_k(directory, identity, rfe, X, y, X, y, top, rfe["names"], "rf", progress)
                self.assertEqual(again, result)
                self.assertEqual(factory.call_count, 99)

    def test_logged_supervisor_preserves_stderr_append_and_native_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            first = unittest.mock.Mock(stdin=io.StringIO(), stdout=io.StringIO("first output\n"),
                                       stderr=io.StringIO("fixture stderr\n"))
            first.wait.return_value = 7
            second = unittest.mock.Mock(stdin=io.StringIO(),
                stdout=io.StringIO(pipeline.MARKER + '{"status":"PASS"}\n'), stderr=io.StringIO())
            second.wait.return_value = 0
            with patch.object(pipeline.subprocess, "Popen", side_effect=[first, second]):
                failed = pipeline.launch("tests", {"source": str(source)})
                passed = pipeline.launch("tests", {"source": str(source)})
            self.assertEqual(failed["exit_code"], 7)
            self.assertEqual(failed["program_status"], "FAIL")
            self.assertEqual(passed["exit_code"], 0)
            self.assertEqual(passed["program_status"], "PASS")
            contents = (source / "logs/ALL_EXECUTION_RESULTS.log").read_text()
            self.assertIn("first output", contents)
            self.assertIn("fixture stderr", contents)
            self.assertEqual(contents.count('"START"'), 2)

    def test_locked_log_prevents_silent_unlogged_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path.open

            def guarded(path, *args, **kwargs):
                if path.name == "ALL_EXECUTION_RESULTS.log":
                    raise PermissionError("fixture locked log")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", guarded), patch.object(pipeline.subprocess, "Popen") as process:
                with self.assertRaises(OSError):
                    pipeline.launch("tests", {"source": tmp})
                process.assert_not_called()


class PublicationRegressionTests(unittest.TestCase):
    def test_synthetic_full_lifecycle_publishes_reference_layout_and_resume_does_not_repeat_test(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from sklearn.tree import DecisionTreeClassifier
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Path(__file__).resolve().parents[1]
            dataset = root / "CICFLOWMETER_FEATURE_CSV"
            folder = dataset / "FW256_FS8"
            folder.mkdir(parents=True)
            names = [f"f{i:03d}" for i in range(380)]
            counts = {"train": 12, "val": 8, "test": 8}
            rng = np.random.RandomState(42)
            for role, count in counts.items():
                matrix = rng.normal(size=(count, 380))
                data = {name: matrix[:, index] for index, name in enumerate(names)}
                data["Window_Label"] = np.tile([0, 1], count // 2)
                pq.write_table(pa.table(data), folder / f"{role}_windows.parquet")
            spec = pipeline.RunSpec("mi_rfe_rf", "FW256_FS8", "mi_rfe_rf_FW256_FS8_fixture")
            fixture_contract = {**pipeline.RFE_PARAMS, "rf": dict(n_estimators=2, max_depth=2, random_state=42, n_jobs=1)}
            deps = pipeline.dependencies()
            request = {"source": str(source), "dataset": str(dataset), "output": str(root),
                       "spec": asdict(spec), "mode": "full_train", "git_commit": "fixture",
                       "confirmation": "RUN PROVISIONAL " + spec.run_name,
                       "unit_tests": {"status": "PASS", "tests_run": 1,
                                      "source_sha256": pipeline.source_snapshot(source)["sha256"], "versions": deps}}
            real_reader = pipeline.read_values
            with patch.dict(pipeline.EXPECTED, {"FW256_FS8": counts}), \
                 patch.object(pipeline, "RFE_PARAMS", fixture_contract), \
                 patch("stage5_rfe.rfe_selection.branch_contract", return_value={"parameters": fixture_contract["rf"]}), \
                 patch.object(pipeline, "resource_gate", return_value={"status": "PASS", "reasons": [], "fixture_only": True}), \
                 patch.object(pipeline, "reference_mi", return_value=pd.Series(np.arange(380, 0, -1), index=names)), \
                 patch.object(pipeline, "make_model", side_effect=lambda branch: DecisionTreeClassifier(max_depth=2, random_state=42)), \
                 patch.object(pipeline, "read_values", wraps=real_reader) as reader:
                result = pipeline.execute_run(request)
                self.assertEqual(result["status"], "PASS")
                checkpoint, destination = pipeline.locations(root, spec)
                self.assertEqual(destination, root / "experiment_results" / spec.run_name)
                for name in pipeline.REQUIRED_ARTIFACTS:
                    self.assertTrue((destination / name).is_file(), name)
                test_calls = [call for call in reader.call_args_list if call.args[1] == "test"]
                self.assertEqual(len(test_calls), 1)
                original_hash = pipeline.sha256(destination / "artifact_manifest.json")
                with self.assertRaises(pipeline.ContractError):
                    pipeline.execute_run(request)  # full mode must not overwrite.
                with self.assertRaises(pipeline.ContractError):
                    pipeline.execute_run({**request, "mode": "resume", "git_commit": "changed"})
                resumed = pipeline.execute_run({**request, "mode": "resume"})
                self.assertEqual(resumed["status"], "PASS")
                self.assertEqual(pipeline.sha256(destination / "artifact_manifest.json"), original_hash)
                self.assertEqual(len([c for c in reader.call_args_list if c.args[1] == "test"]), 1)
                with patch.object(pipeline, "read_values", side_effect=AssertionError("verify must not read values")):
                    self.assertEqual(pipeline.verify_directory(destination)["status"], "PASS")
                (destination / "features.json").write_text("[]")
                with self.assertRaises(pipeline.ContractError):
                    pipeline.verify_directory(destination)


if __name__ == "__main__":
    unittest.main(verbosity=2)
