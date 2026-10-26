"""Stage 8 unit tests. These tests use mocks/synthetic arrays only."""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import ciciomt24_training as core
    import ciciomt24_xgboost as stage8
except ModuleNotFoundError:
    from huyen_code import ciciomt24_training as core
    from huyen_code import ciciomt24_xgboost as stage8


class FakeBooster:
    def save_config(self):
        return json.dumps({"learner": {"generic_param": {"nthread": "-1"}}})

    def num_boosted_rounds(self):
        return 200


class FakeXGBClassifier:
    def __init__(self, **params):
        self.params = dict(params)
        self.fit_calls = 0

    def get_params(self, deep=False):
        return dict(self.params)

    def get_xgb_params(self):
        return dict(self.params)

    def fit(self, X, y):
        self.fit_calls += 1
        self.fit_shape = tuple(X.shape)
        self.fit_labels = np.asarray(y).copy()
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int8)

    def get_booster(self):
        return FakeBooster()

    def save_model(self, path):
        Path(path).write_bytes(b"mock-xgboost-model")


def fake_xgboost():
    return SimpleNamespace(
        __version__=stage8.SUPPORTED_XGBOOST_VERSION,
        build_info=lambda: {"USE_CUDA": True, "MOCK": True},
        XGBClassifier=FakeXGBClassifier,
    )


def split(role: str, rows: int = 8) -> core.PreparedSplit:
    X = np.arange(rows * 380, dtype=np.float32).reshape(rows, 380)
    y = np.asarray(([0, 1] * ((rows + 1) // 2))[:rows], dtype=np.int8)
    return core.PreparedSplit(role, X, y)


class HyperparameterAuditTests(unittest.TestCase):
    def test_rf_final_classifier_is_exact_300_20_3(self):
        self.assertEqual(core.RF_PARAMS["n_estimators"], 300)
        self.assertEqual(core.RF_PARAMS["max_depth"], 20)
        self.assertEqual(core.RF_PARAMS["min_samples_split"], 3)

    def test_svm_classifiers_use_required_c_and_max_iter(self):
        for params in (core.BASELINE_SVM_PARAMS, core.COMPACT_SVM_PARAMS):
            self.assertEqual(params["C"], 10.0)
            self.assertEqual(params["max_iter"], 20000)

    def test_baseline_and_compact_svm_keep_source_differences(self):
        self.assertTrue(core.BASELINE_SVM_PARAMS["probability"])
        self.assertNotIn("class_weight", core.BASELINE_SVM_PARAMS)
        self.assertFalse(core.COMPACT_SVM_PARAMS["probability"])
        self.assertEqual(core.COMPACT_SVM_PARAMS["class_weight"], "balanced")

    def test_rf_rfe_is_not_rf_final_classifier(self):
        self.assertEqual(core.RF_RFE_PARAMS["n_estimators"], 100)
        self.assertEqual(core.RF_RFE_PARAMS["max_depth"], 10)
        self.assertEqual(core.RF_RFE_PARAMS["min_samples_split"], 2)
        self.assertNotEqual(core.RF_RFE_PARAMS, core.RF_PARAMS)

    def test_svm_rfe_is_separate_from_rf_rfe_and_rf_final(self):
        self.assertEqual(core.SVM_RFE_PARAMS["n_estimators"], 100)
        self.assertEqual(core.SVM_RFE_PARAMS["max_depth"], 15)
        self.assertNotIn("min_samples_split", core.SVM_RFE_PARAMS)
        self.assertNotEqual(core.SVM_RFE_PARAMS, core.RF_RFE_PARAMS)
        self.assertNotEqual(core.SVM_RFE_PARAMS, core.RF_PARAMS)

    @mock.patch.object(stage8, "load_xgboost", side_effect=fake_xgboost)
    def test_xgboost_final_classifier_is_exact_200_10_3(self, _load):
        model = stage8.make_xgb("cpu")
        params = model.get_params()
        self.assertEqual(params["n_estimators"], 200)
        self.assertEqual(params["max_depth"], 10)
        self.assertEqual(params["min_child_weight"], 3)
        self.assertEqual(params["objective"], "binary:logistic")
        self.assertEqual(params["device"], "cpu")
        self.assertNotIn("scale_pos_weight", params)

    def test_old_rf_svm_experiment_domain_is_unchanged(self):
        source = inspect.getsource(core.run_official)
        self.assertIn('("baseline_svm", "mi_rfe_svm", "mi_rfe_rf")', source)
        self.assertNotIn("baseline_xgb", source)
        self.assertNotIn("mi_rfe_xgb", source)


class RFEPolicyTests(unittest.TestCase):
    def test_mi_rfe_xgb_is_blocked_without_approval(self):
        policy = stage8.load_rfe_policy(None, "mi_rfe_xgb")
        self.assertFalse(policy["approved"])
        self.assertTrue(policy["status"].startswith("BLOCKED"))

    def test_baseline_xgb_does_not_require_rfe_policy(self):
        policy = stage8.load_rfe_policy(None, "baseline_xgb")
        self.assertTrue(policy["approved"])
        self.assertEqual(policy["status"], "NOT_APPLICABLE")

    def test_exact_approved_policy_reuses_rf_branch_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approved.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": stage8.RFE_POLICY_SCHEMA,
                        "status": "APPROVED",
                        "approved": True,
                        "policy_id": stage8.RFE_POLICY_ID,
                        "rf_rfe_parameters": core.RF_RFE_PARAMS,
                    }
                ),
                encoding="utf-8",
            )
            policy = stage8.load_rfe_policy(path, "mi_rfe_xgb")
        self.assertTrue(policy["approved"])
        self.assertEqual(policy["rf_rfe_parameters"], core.RF_RFE_PARAMS)
        self.assertNotEqual(policy["rf_rfe_parameters"], core.SVM_RFE_PARAMS)
        self.assertNotEqual(policy["rf_rfe_parameters"], core.RF_PARAMS)

    def test_policy_rejects_substituted_xgboost_rfe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": stage8.RFE_POLICY_SCHEMA,
                        "status": "APPROVED",
                        "approved": True,
                        "policy_id": stage8.RFE_POLICY_ID,
                        "rf_rfe_parameters": {"estimator": "XGBClassifier"},
                    }
                ),
                encoding="utf-8",
            )
            policy = stage8.load_rfe_policy(path, "mi_rfe_xgb")
        self.assertFalse(policy["approved"])


class ScientificProtocolTests(unittest.TestCase):
    @mock.patch.object(stage8, "make_xgb")
    @mock.patch.object(core, "timed_fit")
    def test_k_search_uses_smallest_k_on_accuracy_tie(self, timed_fit, make_xgb):
        model = FakeXGBClassifier(**stage8.XGB_FIXED_POLICY, device="cpu")
        make_xgb.return_value = model
        timed_fit.side_effect = lambda estimator, X, y: (
            estimator.fit(X, y),
            {"seconds": 0.0, "peak_rss_bytes": 0, "warnings": []},
        )
        X_train = np.zeros((4, 100), dtype=np.float32)
        X_val = np.zeros((2, 100), dtype=np.float32)
        names = [f"f{i}" for i in range(100)]
        result = stage8.search_k_xgb(
            X_train,
            np.asarray([0, 1, 0, 1], dtype=np.int8),
            X_val,
            np.asarray([0, 1], dtype=np.int8),
            names,
            names,
            "cpu",
        )
        self.assertEqual(len(result["results"]), 100)
        self.assertEqual(result["best_k"], 1)
        self.assertEqual(result["features"], ["f0"])

    @mock.patch.object(stage8, "make_xgb")
    @mock.patch.object(core, "timed_fit")
    def test_final_refit_uses_train_plus_validation_only(self, timed_fit, make_xgb):
        model = FakeXGBClassifier(**stage8.XGB_FIXED_POLICY, device="cpu")
        make_xgb.return_value = model
        timed_fit.side_effect = lambda estimator, X, y: (
            estimator.fit(X, y),
            {"seconds": 0.0, "peak_rss_bytes": 0, "warnings": []},
        )
        fitted = stage8.fit_final_xgb(
            split("train", 6),
            split("val", 4),
            list(core.FEATURE_NAMES[:3]),
            "cpu",
        )
        self.assertEqual(fitted["fit_rows"], 10)
        self.assertEqual(fitted["fit_role"], "train+val")
        self.assertEqual(fitted["model"].fit_shape, (10, 3))
        self.assertFalse(fitted["mi_rfe_refit"])

    def test_test_guard_allows_exactly_one_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = stage8.TestOnceGuard(root)
            guard.claim("A" * 64)
            fitted = {
                "model": FakeXGBClassifier(),
                "scaler": mock.Mock(
                    transform=lambda X: X,
                ),
                "features": list(core.FEATURE_NAMES[:2]),
            }
            result = guard.evaluate(fitted, split("test", 4))
            self.assertEqual(result["test_evaluation_calls"], 1)
            with self.assertRaises(RuntimeError):
                guard.evaluate(fitted, split("test", 4))

    def test_preflight_source_contains_no_full_loader_call(self):
        source = inspect.getsource(stage8.preflight)
        self.assertNotIn(".load_full(", source)
        self.assertIn('"test_feature_rows_read": 0', source)
        self.assertIn('"model_fit_calls": 0', source)

    def test_worker_freezes_selection_and_refits_before_test_claim(self):
        source = inspect.getsource(stage8.run_worker)
        frozen = source.index('"selection_frozen.json"')
        refit = source.index("fit_final_xgb(")
        claim = source.index("test_guard.claim(")
        test_load = source.index('"test", cache_root')
        evaluate = source.index("test_guard.evaluate(")
        self.assertLess(frozen, refit)
        self.assertLess(refit, claim)
        self.assertLess(claim, test_load)
        self.assertLess(test_load, evaluate)

    def test_resource_fail_occurs_before_worker_spawn(self):
        source = inspect.getsource(stage8.supervise)
        status_check = source.index('audit["status"] != "PASS"')
        spawn = source.index("subprocess.Popen(")
        self.assertLess(status_check, spawn)

    def test_existing_run_is_never_overwritten(self):
        source = inspect.getsource(stage8.preflight)
        self.assertIn('"output_is_fresh"', source)
        self.assertIn('output.name + ".incomplete"', source)
        worker_source = inspect.getsource(stage8.run_worker)
        self.assertIn("staged.mkdir(parents=True)", worker_source)

    def test_no_sampling_or_class_balancing_in_official_xgb_path(self):
        source = inspect.getsource(stage8.run_worker)
        self.assertNotIn("smaller_sample", source)
        self.assertNotIn("load_sample", source)
        self.assertNotIn("scale_pos_weight", inspect.getsource(stage8.make_xgb))
        self.assertIn('"sampled_training": False', source)

    def test_baseline_and_compact_xgb_paths_are_explicitly_separate(self):
        source = inspect.getsource(stage8.run_worker)
        self.assertIn('experiment == "baseline_xgb"', source)
        self.assertIn("select_mi_rfe_xgb(", source)
        self.assertEqual(stage8.EXPERIMENTS, ("baseline_xgb", "mi_rfe_xgb"))


class NotebookContractTests(unittest.TestCase):
    def test_notebook_has_fourteen_ordered_code_cells_and_defaults_dry(self):
        repo = Path(stage8.__file__).resolve().parents[1]
        notebook = json.loads(
            (repo / "notebooks" / "STAGE8_XGBOOST_COLAB.ipynb").read_text(
                encoding="utf-8"
            )
        )
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 14)
        for number, cell in enumerate(cells, 1):
            self.assertEqual(cell["metadata"]["id"], f"stage8-cell-{number}")
            compile("".join(cell["source"]), f"stage8-cell-{number}", "exec")
        source = "\n".join("".join(cell["source"]) for cell in cells)
        self.assertIn("MODE = 'dry_run'", source)
        self.assertIn("I_ACCEPT_RESOURCE_BUDGET = False", source)
        self.assertIn("EXPERIMENT = 'baseline_xgb'", source)
        self.assertIn("BACKEND = 'cpu'", source)
        self.assertIn("input('Type the exact confirmation now: ')", source)
        self.assertIn("SESSION_APPROVAL_TOKEN", source)
        self.assertNotIn(r"D:\COMPACT-MI-RFE", source)

    def test_stage8_config_has_required_xgboost_values(self):
        repo = Path(stage8.__file__).resolve().parents[1]
        config = json.loads(
            (repo / "configs" / "stage8_xgboost.json").read_text(encoding="utf-8")
        )
        final = config["final_classifier"]
        self.assertEqual(final["n_estimators"], 200)
        self.assertEqual(final["max_depth"], 10)
        self.assertEqual(final["min_child_weight"], 3)
        self.assertEqual(final["scale_pos_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
