"""Stage 5.1: synthetic/unit checks; never fit a full dataset or read Test data."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import ciciomt24_training as t


def data(role="train", n=6, offset=0):
    X = (np.arange(n * 380, dtype=np.float32).reshape(n, 380) / 380 + offset)
    return t.PreparedSplit(role, X, np.arange(n, dtype=np.int8) % 2)


def frame(n=6):
    result = pd.DataFrame(data(n=n).X, columns=t.FEATURE_NAMES)
    result["Label"] = np.arange(n) % 2
    result["Window_Index"] = np.arange(n)
    result["Num_Flows"] = 512
    result["Source_CSV_ID"] = 71
    result["Scenario"] = "not a feature"
    return result


def window_fixture(directory):
    segment = t.sw.Segment(1, "train/source.csv", "Example", "train", 264,
        "2024-01-01 00:00:00", "2024-01-01 00:00:01", 0, 263,
        "train/source.csv", "Attack", "SCENARIO_PROXY")
    result = pd.DataFrame({name: [float(i), float(i + 1)]
                           for i, name in enumerate(t.FEATURE_NAMES)})
    metadata = {
        "Window_ID": [t.sw._window_id(segment, 256, 8, i) for i in range(2)],
        "Split": ["train"] * 2, "Scenario": ["Example"] * 2,
        "Source_CSV": [segment.source_csv] * 2, "Source_CSV_ID": ["sid001"] * 2,
        "Original_Row_Index_Start": ["row:0", "row:8"],
        "Original_Row_Index_End": ["row:255", "row:263"],
        "Window_Start_Timestamp": [segment.first_timestamp] * 2,
        "Window_End_Timestamp": [segment.last_timestamp] * 2,
        "Window_Index": [0, 1], "Num_Flows": [256, 256],
        "Proxy_Label": ["Attack"] * 2, "Label_Origin": ["SCENARIO_PROXY"] * 2,
        "Label": [1, 1],
    }
    result = pd.concat([pd.DataFrame(metadata), result], axis=1)
    file = t.WindowFile(Path(directory) / "fixture.csv", "train", "Example", 1, 2, 1,
                        segment, 256, 8)
    return result, file


class FeaturePreparationTests(unittest.TestCase):
    def test_exact_380_whitelist_and_names_even_with_numeric_metadata(self):
        f = frame()
        got = t.prepare_xy(f[f.columns[::-1]], "train")
        self.assertEqual(got.X.shape, (6, 380))
        self.assertEqual(got.feature_names, t.FEATURE_NAMES)
        self.assertEqual(got.X.dtype, np.float32)
        np.testing.assert_array_equal(got.X, data().X)

    def test_nan_inf_zero_compatible_with_source(self):
        f = frame()
        f.loc[0, t.FEATURE_NAMES[0]] = np.nan
        f.loc[1, t.FEATURE_NAMES[0]] = np.inf
        f.loc[2, t.FEATURE_NAMES[0]] = -np.inf
        got = t.prepare_xy(f, "train")
        np.testing.assert_array_equal(got.X[:3, 0], [0, 0, 0])
        self.assertEqual(got.cleaning, {"nan_replaced": 1, "inf_replaced": 2})
        self.assertTrue(t.finite_matrix(got.X))

    def test_reject_missing_duplicate_and_text_feature(self):
        f = frame()
        cases = [f.drop(columns=[t.FEATURE_NAMES[0]]),
                 pd.concat([f, f.iloc[:, :1]], axis=1),
                 f.assign(**{t.FEATURE_NAMES[0]: "text"})]
        for invalid in cases:
            with self.subTest(columns=len(invalid.columns)), self.assertRaises(ValueError):
                t.prepare_xy(invalid, "train")

    def test_reject_nonbinary_or_missing_labels(self):
        for value in (2, -1, np.nan, "Attack"):
            f = frame()
            f["Label"] = f["Label"].astype(object)
            f.loc[0, "Label"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                t.prepare_xy(f, "train")

    def test_float32_overflow_and_bad_role_rejected(self):
        f = frame()
        f[t.FEATURE_NAMES[0]] = 1e300
        with self.assertRaises(ValueError):
            t.prepare_xy(f, "train")
        with self.assertRaises(ValueError):
            t.prepare_xy(frame(), "embargo")

    def test_metrics_include_minority_sensitive_scores(self):
        got = t.metrics([0, 1, 1, 1], [1, 1, 1, 1])
        self.assertEqual(got["accuracy"], .75)
        self.assertEqual(got["balanced_accuracy"], .5)
        self.assertLess(got["macro_f1"], .5)
        self.assertEqual(got["confusion_matrix_labels_0_benign_1_attack"], [[0, 1], [0, 3]])
        self.assertEqual(got["label_origin"], "SCENARIO_PROXY")


class SourceParityAndLeakageTests(unittest.TestCase):
    def test_reproduce_feature_names_bug_without_importing_original_modules(self):
        f, _ = window_fixture("unused")
        for name, expected_names in (("[[CTU13]8_experiment2_svm.py", 388),
                                     ("[[CTU13]9_ex2_rf.py", 380)):
            source = ast.parse((t.DEFAULT_ORIGINAL / name).read_text(encoding="utf-8-sig"))
            nodes = [n for n in source.body if
                     (isinstance(n, ast.FunctionDef) and n.name == "prepare_xy") or
                     (isinstance(n, ast.Assign) and any(isinstance(target, ast.Name) and
                       target.id == "DROP_COLUMNS" for target in n.targets))]
            namespace = {"np": np, "pd": pd, "LABEL_COL": "Label"}
            # Only the audited preparation function/constants; no module mkdir/main.
            exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
            X, y, names = namespace["prepare_xy"](f)
            self.assertEqual(X.shape, (2, 380))
            self.assertEqual(len(names), expected_names)
        self.assertEqual(len(t.prepare_xy(f, "train").feature_names), 380)

    def test_original_hashes_and_all_parameter_dictionaries_match(self):
        self.assertEqual(len(t.source_hashes()), 5)
        self.assertEqual(len(t.source_audit()), 3)
        self.assertEqual(t.SVM_RFE_PARAMS["max_depth"], 15)
        self.assertEqual(t.RF_RFE_PARAMS["max_depth"], 10)
        self.assertNotIn("min_samples_split", t.SVM_RFE_PARAMS)
        self.assertTrue(t.make_model("svm", baseline=True).probability)
        self.assertIsNone(t.make_model("svm", baseline=True).class_weight)
        self.assertFalse(t.make_model("svm").probability)
        self.assertEqual(t.make_model("svm").class_weight, "balanced")
        self.assertEqual(t.make_model("rf").min_samples_split, 3)

    def test_mi_train_only_scaler_and_top100_alignment(self):
        train = data()
        with patch.object(t, "mutual_info_classif", return_value=np.arange(380)) as mi:
            result = t.select_mi(train)
        np.testing.assert_allclose(result["scaler"].mean_, train.X.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(mi.call_args.args[0].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_array_equal(mi.call_args.args[1], train.y)
        self.assertEqual(mi.call_args.kwargs, {"random_state": 42})
        self.assertEqual(result["names"], list(t.FEATURE_NAMES[::-1][:100]))
        self.assertEqual(result["indices"], list(range(379, 279, -1)))

    def test_selection_rejects_validation_as_train_and_test_as_validation(self):
        with self.assertRaises(ValueError):
            t.select_mi(data("val"))
        with self.assertRaises(ValueError):
            t.select_features(data(), data("test"), "svm")
        with self.assertRaises(ValueError):
            t.select_features(data("test"), data("val"), "rf")

    def test_top100_scaler_fit_train_not_validation(self):
        train, val = data(), data("val", offset=1000)
        ids = list(range(100))
        X, V, scaler = t.scale_top100(train, val, {"indices": ids})
        np.testing.assert_allclose(scaler.mean_, train.X[:, ids].mean(axis=0), atol=1e-6)
        self.assertGreater(V.mean(), 100)
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-6)

    def test_rfe_step1_and_branch_specific_forest(self):
        captured = []
        def fake_fit(selector, X, y):
            captured.append(selector)
            selector.ranking_ = np.arange(1, 101)
            return selector, {"seconds": 0}
        with patch.object(t, "timed_fit", side_effect=fake_fit):
            for kind in ("svm", "rf"):
                got = t.rank_rfe(data().X[:, :100], data().y, list(t.FEATURE_NAMES[:100]), kind)
                self.assertEqual(got["forest_fits"], 100)
                self.assertTrue(got["complete_ranking"])
        self.assertEqual([r.estimator.max_depth for r in captured], [15, 10])
        self.assertTrue(all(r.step == 1 and r.n_features_to_select == 1 for r in captured))

    def test_accuracy_tie_keeps_smallest_k_even_if_f1_differs(self):
        # Both accuracies=.5; k=2 has a higher Attack F1 but may not win.
        models = [SimpleNamespace(predict=lambda X: np.array([0, 0, 0, 0])),
                  SimpleNamespace(predict=lambda X: np.array([1, 1, 1, 1]))]
        with patch.object(t, "make_model", side_effect=models), \
             patch.object(t, "timed_fit", side_effect=lambda m, X, y: (m, {"converged": True})):
            got = t.search_k(np.zeros((4, 100)), [0, 0, 1, 1], np.zeros((4, 100)),
                             [0, 0, 1, 1], list(t.FEATURE_NAMES[:100]),
                             list(t.FEATURE_NAMES[:100]), "svm", (1, 2))
        self.assertEqual(got["best_k"], 1)
        self.assertGreater(got["results"][1]["f1_attack"], got["results"][0]["f1_attack"])

    def test_nonconvergence_recorded_not_hidden_or_accuracy_rule_changed(self):
        model = SimpleNamespace(predict=lambda X: np.array([0, 1]))
        with patch.object(t, "make_model", return_value=model), \
             patch.object(t, "timed_fit", return_value=(model, {"converged": False})):
            got = t.search_k(np.zeros((2, 100)), [0, 1], np.zeros((2, 100)), [0, 1],
                             list(t.FEATURE_NAMES[:100]), list(t.FEATURE_NAMES[:100]), "svm", (1,))
        self.assertEqual(got["best_k"], 1)
        self.assertEqual(got["nonconverged_k"], [1])
        self.assertFalse(got["selection_converged"])

    def test_refit_combines_train_val_only_and_frozen_features(self):
        train, val, test = data(n=4), data("val", n=2, offset=10), data("test", n=2, offset=10000)
        model = SimpleNamespace(predict=lambda X: np.ones(len(X), dtype=np.int8))
        fits = []
        def fit(m, X, y):
            fits.append((X.copy(), y.copy()))
            return m, {"converged": True}
        with patch.object(t, "make_model", return_value=model), \
             patch.object(t, "timed_fit", side_effect=fit), \
             patch.object(t, "select_mi", side_effect=AssertionError("MI refit forbidden")), \
             patch.object(t, "rank_rfe", side_effect=AssertionError("RFE refit forbidden")):
            result = t.refit_final(train, val, test, "svm", [t.FEATURE_NAMES[5], t.FEATURE_NAMES[1]])
        expected = np.concatenate([train.X, val.X])[:, [5, 1]]
        np.testing.assert_allclose(result["scaler"].mean_, expected.mean(axis=0), atol=1e-6)
        self.assertEqual(fits[0][0].shape, (6, 2))
        np.testing.assert_array_equal(fits[0][1], np.concatenate([train.y, val.y]))
        self.assertEqual(result["fit_rows"], 6)
        self.assertFalse(result["mi_rfe_refit"])

    def test_baseline_final_uses_all380_train_val_features(self):
        model = SimpleNamespace(predict=lambda X: np.ones(len(X), dtype=np.int8))
        with patch.object(t, "make_model", return_value=model) as factory, \
             patch.object(t, "timed_fit", side_effect=lambda m, X, y: (m, {})) as fit:
            result = t.refit_final(data(), data("val"), data("test"), "svm", baseline=True)
        self.assertEqual(fit.call_args.args[1].shape, (12, 380))
        self.assertEqual(len(result["features"]), 380)
        factory.assert_called_once_with("svm", baseline=True)

    def test_official_training_requires_explicit_flag_before_any_io(self):
        with patch.object(t, "WindowStore", side_effect=AssertionError("No I/O permitted")):
            with self.assertRaises(PermissionError):
                t.run_official(Path("missing"), Path("missing"), Path("missing"), t.CONFIGS[0], "baseline_svm")

    def test_insufficient_memory_stops_without_resampling(self):
        plan = {"windows": {"train": 100, "val": 20, "test": 20},
                "raw_float32_all_splits_bytes": 140 * 380 * 4,
                "mi_selection_array_scenario_bytes": 100000}
        with patch.object(t, "resource_plan", return_value={"cfg": plan}), \
             patch.object(t.psutil, "virtual_memory", return_value=SimpleNamespace(available=1000)):
            with self.assertRaises(MemoryError):
                t.official_memory_preflight(None, "cfg", "baseline_svm")


class SamplingAndLoaderTests(unittest.TestCase):
    def test_test_data_read_forbidden_and_benchmark_caps(self):
        store = t.WindowStore.__new__(t.WindowStore)
        for role, rows in (("test", 10), ("train", 8193), ("val", 2049)):
            with self.subTest(role=role), self.assertRaises(ValueError):
                store.load_sample("unused", role, rows)

    def test_proportional_quota_determinism_zero_and_capacity(self):
        self.assertEqual(t.proportional_quotas([1, 1, 1], 2), [1, 1, 0])
        self.assertEqual(t.proportional_quotas([0, 0], 0), [0, 0])
        self.assertEqual(sum(t.proportional_quotas([20, 2, 40], 17)), 17)
        with self.assertRaises(ValueError):
            t.proportional_quotas([1], 2)

    def test_nested_sample_preserves_sources_and_reproducible_ids(self):
        sample = data(n=20)
        sample.y = (np.arange(20, dtype=np.int8) // 2) % 2
        lineage = [{"Source_CSV_ID": i // 2} for i in range(20)]
        one, ids = t.smaller_sample(sample, 10, lineage, return_indices=True)
        two, ids2 = t.smaller_sample(sample, 10, lineage, return_indices=True)
        self.assertEqual(len({lineage[int(i)]["Source_CSV_ID"] for i in ids}), 10)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_array_equal(one.X, two.X)

    def test_validated_segment_metadata_and_chunk_offset(self):
        f, file = window_fixture("unused")
        t.WindowStore.validate_chunk(f, file, 0)
        t.WindowStore.validate_chunk(f.iloc[1:], file, 1)
        with self.assertRaises(ValueError):
            t.WindowStore.validate_chunk(f.iloc[1:], file, 0)

    def test_reject_duplicate_window_wrong_split_source_or_provenance(self):
        f, file = window_fixture("unused")
        for col, value in (("Window_ID", "duplicate"), ("Split", "test"),
                           ("Source_CSV_ID", "sid002"), ("Label", 0),
                           ("Label_Origin", "GROUND_TRUTH"),
                           ("Original_Row_Index_End", "row:999")):
            bad = f.copy()
            bad.loc[0, col] = value
            with self.subTest(column=col), self.assertRaises(ValueError):
                t.WindowStore.validate_chunk(bad, file, 0)

    def test_streaming_full_loader_synthetic_only_retains_every_row(self):
        # Two synthetic rows only: this does not invoke the official full runner.
        with tempfile.TemporaryDirectory() as directory:
            f, file = window_fixture(directory)
            f.loc[0, t.FEATURE_NAMES[0]] = np.nan
            f.to_csv(file.path, index=False)
            store = t.WindowStore.__new__(t.WindowStore)
            store.files = {"cfg": [file]}
            loaded = store.load_full("cfg", "train", Path(directory))
            try:
                self.assertEqual(loaded.X.shape, (2, 380))
                self.assertEqual(loaded.X[0, 0], 0)
                np.testing.assert_array_equal(loaded.y, [1, 1])
                self.assertEqual(loaded.X[1, 379], 380)
            finally:
                loaded.X._mmap.close()
                loaded.y._mmap.close()

    def test_header_requires_exact_canonical_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            f, file = window_fixture(directory)
            f.to_csv(file.path, index=False)
            t.WindowStore.validate_header(file.path)
            f.drop(columns=[t.FEATURE_NAMES[0]]).to_csv(file.path, index=False)
            with self.assertRaises(ValueError):
                t.WindowStore.validate_header(file.path)


class Stage52OrchestrationTests(unittest.TestCase):
    def test_cli_dispatches_exactly_one_experiment_and_config(self):
        args = ["ciciomt24_training.py", "train", "--config", "FW_512_FS_16",
                "--experiment", "mi_rfe_rf", "--output", "synthetic-unit-output",
                "--confirm-full-training"]
        with patch.object(sys, "argv", args), patch.object(t, "run_official") as official, \
             patch.object(t, "run_benchmark", side_effect=AssertionError("Wrong route")):
            t.main()
        official.assert_called_once()
        self.assertEqual(official.call_args.args[-2:], ("FW_512_FS_16", "mi_rfe_rf"))
        self.assertTrue(official.call_args.kwargs["confirm_full_training"])

    def test_ram_rejection_leaves_no_run_directory(self):
        store = SimpleNamespace(snapshot=lambda: {})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "rejected"
            with patch.object(t, "WindowStore", return_value=store), \
                 patch.object(t, "official_memory_preflight", side_effect=MemoryError("synthetic")):
                with self.assertRaises(MemoryError):
                    t.run_official(Path(directory), t.DEFAULT_ORIGINAL, out, "FW_512_FS_16",
                                   "mi_rfe_rf", confirm_full_training=True)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name + ".incomplete").exists())

    def test_test_read_only_after_frozen_features_and_final_fit(self):
        # Entire official orchestration on six SYNTHETIC rows; all ML fits mocked.
        # This does not read any real Train/Val/Test CSV or train a model.
        order, cfg = [], "FW_512_FS_16"
        with tempfile.TemporaryDirectory(dir=Path(t.__file__).parent, prefix="stage52_unit_") as directory:
            out = Path(directory) / "synthetic-official"
            f, file = window_fixture(directory)
            f.to_csv(file.path, index=False)
            def load(config, role, cache, **kwargs):
                self.assertEqual(config, cfg)
                if role == "test":
                    self.assertIn("final_fit", order)
                order.append("load_" + role)
                return data(role, n=2)
            store = SimpleNamespace(files={cfg: [file]}, snapshot=lambda: {"fixture": 1},
                                    validate_header=t.WindowStore.validate_header, load_full=load)
            selected = [t.FEATURE_NAMES[0]]
            selection = {"features": selected, "nonconverged_k": [], "best_k": 1,
                         "best_validation_accuracy": .5,
                         "mi": {"measurement": {"seconds": 0}},
                         "ranking": {"measurement": {"seconds": 0}},
                         "results": [{"k": k} for k in range(1, 101)]}
            fitted = {"model": "unit-test-only-not-a-model", "scaler": "unit-test-only",
                      "features": selected, "fit": {"seconds": 0, "warnings": []},
                      "fit_rows": 4, "fit_role": "train+val", "mi_rfe_refit": False}
            def fit(*args, **kwargs):
                frozen = json.loads((out.with_name(out.name + ".incomplete") /
                                     "selection_frozen.json").read_text())
                self.assertEqual(frozen["features"], selected)
                self.assertNotIn("load_test", order)
                order.append("final_fit")
                return fitted
            def evaluate(got, test):
                order.append("evaluate_once")
                self.assertEqual(test.role, "test")
                return {**got, "metrics": t.metrics(test.y, [0, 1]),
                        "test_rows_evaluated": 2, "test_evaluation_calls": 1,
                        "predict_seconds": 0}
            plan = {cfg: {"windows": {"train": 2, "val": 2, "test": 2},
                          "raw_float32_all_splits_bytes": 6 * 380 * 4}}
            with patch.object(t, "WindowStore", return_value=store), \
                 patch.object(t, "official_memory_preflight", return_value={"synthetic": True}), \
                 patch.object(t, "resource_plan", return_value=plan), \
                 patch.object(t, "select_features", return_value=selection) as select, \
                 patch.object(t, "fit_final_classifier", side_effect=fit) as fit_mock, \
                 patch.object(t, "evaluate_final", side_effect=evaluate) as evaluate_mock:
                result = t.run_official(Path(directory), t.DEFAULT_ORIGINAL, out, cfg,
                                        "mi_rfe_rf", confirm_full_training=True)
            self.assertEqual(order, ["load_train", "load_val", "final_fit", "load_test", "evaluate_once"])
            self.assertEqual(select.call_args.args[2], "rf")
            self.assertEqual(fit_mock.call_args.args[2], "rf")
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["sampled_training"])
            self.assertTrue((out / "result.json").is_file())
            self.assertFalse(out.with_name(out.name + ".incomplete").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
