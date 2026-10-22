"""Static/scientific contract tests; no estimator fit and no Test CSV read."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from huyen_code import ciciomt24_training as training


class ScientificIntegrityTests(unittest.TestCase):
    def test_baseline_and_compact_svm_contracts(self):
        baseline = training.official_hyperparameters("baseline_svm")
        compact = training.official_hyperparameters("mi_rfe_svm")
        self.assertEqual(baseline["feature_count"], 380)
        self.assertEqual(baseline["classifier"]["C"], 10.0)
        self.assertEqual(baseline["classifier"]["kernel"], "rbf")
        self.assertTrue(baseline["classifier"]["probability"])
        self.assertNotIn("class_weight", baseline["classifier"])
        self.assertEqual(compact["mi_top_k"], 100)
        self.assertEqual(compact["rfe_step"], 1)
        self.assertEqual(compact["rfe_estimator"]["n_estimators"], 100)
        self.assertEqual(compact["rfe_estimator"]["max_depth"], 15)
        self.assertFalse(compact["classifier"]["probability"])
        self.assertEqual(compact["classifier"]["class_weight"], "balanced")
        self.assertEqual(compact["k_grid"], list(range(1, 101)))

    def test_train_only_selection_and_single_late_test_site(self):
        select_source = inspect.getsource(training.select_mi)
        run_source = inspect.getsource(training.run_official)
        self.assertIn('train.role != "train"', select_source)
        self.assertLess(run_source.index("fitted = fit_final_classifier("),
                        run_source.index('store.load_full(config, "test"'))
        self.assertEqual(run_source.count("evaluate_final(fitted, test)"), 1)
        self.assertIn("validation_accuracy_strict_greater_smallest_k_tie", run_source)
        refit_source = inspect.getsource(training.fit_final_classifier)
        self.assertNotIn("select_mi(", refit_source)
        self.assertNotIn("rank_rfe(", refit_source)

    def test_validation_accuracy_tie_keeps_smallest_k_without_real_fit(self):
        class PredictZero:
            def predict(self, X):
                return np.zeros(len(X), dtype=np.int8)
        X = np.zeros((4, 100), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        names = [f"f{i}" for i in range(100)]
        with patch.object(training, "make_model", return_value=PredictZero()), \
             patch.object(training, "timed_fit",
                          return_value=(PredictZero(), {"converged": True})), \
             patch.object(training, "metrics", return_value={"accuracy": .5}):
            result = training.search_k(X, y, X, y, names, names, "svm",
                                       k_values=(1, 2, 3))
        self.assertEqual(result["best_k"], 1)

    def test_metadata_exclusion_and_proxy_warning_are_persisted(self):
        source = inspect.getsource(training.run_official)
        self.assertIn("OUTPUT_METADATA_COLUMNS", source)
        self.assertIn('label_origin="SCENARIO_PROXY"', source)
        self.assertIn('capture_session_independence="NOT_ESTABLISHED"', source)


if __name__ == "__main__":
    unittest.main()
