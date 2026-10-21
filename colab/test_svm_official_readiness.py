"""Stage 7.3A mock/synthetic tests. No full SVM, Test CSV, or Drive access."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from colab import svm_official_readiness as ready
from colab.experiment_launcher import RunSpec


def identity(experiment="baseline_svm", config="FW_512_FS_16", run_name="svm_review_01"):
    return {field: value for field, value in zip(ready.IDENTITY_FIELDS, (
        experiment, config, run_name, "a" * 40, "b" * 64, "c" * 64,
        "d" * 64, "e" * 64, "f" * 64, "g" * 64,
    ))}


class IdentityAndGateTests(unittest.TestCase):
    def test_cross_run_and_missing_hash_approval_rejected(self):
        actual = identity()
        good = {"decision": "APPROVED_STAGE73B", "identity": actual.copy(),
                "max_elapsed_seconds": 3600, "max_tree_rss_bytes": 12 * 1024**3}
        self.assertTrue(ready.approval_matches(good, actual))
        for field in ("experiment", "config", "run_name", "git_commit",
                      "training_code_sha256", "dataset_fingerprint_sha256"):
            altered = {**good, "identity": {**actual, field: "different"}}
            with self.subTest(field=field):
                self.assertFalse(ready.approval_matches(altered, actual))
        self.assertFalse(ready.approval_matches({**good, "max_tree_rss_bytes": 0}, actual))
        self.assertFalse(ready.approval_matches({**good, "max_elapsed_seconds": True}, actual))
        self.assertFalse(ready.approval_matches({**good, "decision": "PASS_TECHNICAL_ONLY"}, actual))
        self.assertFalse(ready.identity_matches({k: v for k, v in actual.items()
                                                 if k != "window_summary_sha256"}, actual))

    def test_crlf_only_is_normalized_other_edit_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "source.py"
            p.write_bytes(b"x=1\r\ny=2\r\n")
            first = ready.normalized_source_sha256(p)
            p.write_bytes(b"x=1\ny=2\n")
            self.assertEqual(ready.normalized_source_sha256(p), first)
            p.write_bytes(b"x=1\ny=3\n")
            self.assertNotEqual(ready.normalized_source_sha256(p), first)

    def test_relative_root_and_both_windows_stems_required(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            self.assertFalse(any(ready.source_contract(repo).values()))
            code = repo / "huyen_code"
            code.mkdir()
            (code / "ciciomt24_training.py").write_text(
                "ROOT = Path(__file__).resolve().parents[1]\n"
                "BASELINE_SVM_PARAMS = {}\nCOMPACT_SVM_PARAMS = {}\n", encoding="utf-8")
            sliding = code / "ciciomt24_sliding_window.py"
            sliding.write_text("from pathlib import Path, PureWindowsPath\n" +
                               "PureWindowsPath(segment.output_relative_path).stem\n" * 2,
                               encoding="utf-8")
            self.assertTrue(all(ready.source_contract(repo).values()))
            sliding.write_text("from pathlib import Path, PureWindowsPath\n" +
                               "PureWindowsPath(segment.output_relative_path).stem\n",
                               encoding="utf-8")
            self.assertFalse(ready.source_contract(repo)["two_windows_stem_sites"])

    def test_review_always_blocks_even_if_mock_inputs_pass(self):
        actual = identity()
        approval = {"decision": "APPROVED_STAGE73B", "identity": actual,
                    "max_elapsed_seconds": 3600, "max_tree_rss_bytes": 12 * 1024**3}
        base = {"status": "PASS", "checks": {"ram_admission": True,
                "local_disk_admission": True, "svm_technical_probe": True},
                "dataset": {"dataset_fingerprint_sha256": actual["dataset_fingerprint_sha256"]},
                "runtime": {}, "ram_guard": {"status": "PASS"}}
        paths = ["training_code_sha256", "sliding_code_sha256"]
        hash_values = iter(actual[name] for name in paths)
        other_hashes = iter(actual[name] for name in (
            "manifest_sha256", "validation_sha256", "window_summary_sha256"))
        with patch.object(ready, "preflight", return_value=base), \
             patch.object(ready, "source_contract", return_value={"portable": True}), \
             patch.object(ready, "git_checkout", return_value={"commit": actual["git_commit"], "clean": True}), \
             patch("colab.dataset_validator.manifest_root_for", return_value=Path("/manifest")), \
             patch.object(ready, "normalized_source_sha256", side_effect=hash_values), \
             patch.object(ready, "sha256", side_effect=other_hashes), \
             patch("colab.experiment_launcher.run_one", side_effect=AssertionError("training forbidden")):
            result = ready.review(Path("/repo"), Path("/windows"), Path("/runs"),
                                  Path("/drive"), RunSpec("baseline_svm", "FW_512_FS_16", "svm_review_01"),
                                  expected_identity=actual, approval=approval)
        self.assertEqual(result["status"], "BLOCKED_STAGE73A")
        self.assertFalse(result["full_training_authorized"])
        self.assertFalse(result["checks"]["stage73b_implementation_and_authorization"])
        self.assertTrue(result["checks"]["per_run_budget_approval_bound"])
        self.assertEqual(result["test_feature_rows_read"], 0)
        self.assertEqual(result["model_fit_calls"], 0)

    def test_resource_failure_cannot_be_hidden_by_probe(self):
        base = {"status": "FAIL", "checks": {"ram_admission": False,
                "local_disk_admission": True, "svm_technical_probe": True}, "dataset": None}
        with patch.object(ready, "preflight", return_value=base), \
             patch.object(ready, "source_contract", return_value={"portable": True}), \
             patch.object(ready, "git_checkout", return_value={"commit": None, "clean": False}), \
             patch("colab.dataset_validator.manifest_root_for", side_effect=FileNotFoundError):
            result = ready.review(Path("/repo"), Path("/windows"), Path("/runs"),
                                  Path("/drive"), RunSpec("mi_rfe_svm", "FW_256_FS_8", "review02"))
        self.assertFalse(result["checks"]["ram_admission"])
        self.assertFalse(result["checks"]["source_and_dataset_identity_approved"])
        self.assertEqual(result["status"], "BLOCKED_STAGE73A")


class ScientificContractTests(unittest.TestCase):
    def test_distinct_svm_params_and_canonical_features(self):
        from huyen_code import ciciomt24_training as training
        baseline = training.make_model("svm", baseline=True).get_params()
        compact = training.make_model("svm").get_params()
        for params in (baseline, compact):
            self.assertEqual((params["kernel"], params["C"], params["gamma"],
                              params["max_iter"], params["cache_size"]),
                             ("rbf", 10.0, "scale", 20000, 10000))
        self.assertTrue(baseline["probability"])
        self.assertIsNone(baseline["class_weight"])
        self.assertFalse(compact["probability"])
        self.assertEqual(compact["class_weight"], "balanced")
        self.assertEqual(len(training.FEATURE_NAMES), 380)
        self.assertEqual(training.SVM_RFE_PARAMS["n_estimators"], 100)
        self.assertEqual(training.SVM_RFE_PARAMS["max_depth"], 15)

    def test_train_only_selection_and_smallest_k_tie_without_fit(self):
        from huyen_code import ciciomt24_training as training
        val = training.PreparedSplit("val", np.zeros((4, 380), dtype=np.float32),
                                     np.array([0, 1, 0, 1], dtype=np.int8))
        with patch.object(training, "select_mi", side_effect=AssertionError("MI must not start")):
            with self.assertRaises(ValueError):
                training.select_features(val, val, "svm")

        class PredictZero:
            def predict(self, X):
                return np.zeros(len(X), dtype=np.int8)

        top = [f"f{i}" for i in range(100)]
        X = np.zeros((4, 100), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        with patch.object(training, "timed_fit", return_value=(PredictZero(), {"converged": True})), \
             patch.object(training, "make_model", return_value=PredictZero()), \
             patch.object(training, "metrics", return_value={"accuracy": 0.5}):
            result = training.search_k(X, y, X, y, top, top, "svm", k_values=(1, 2, 3))
        self.assertEqual(result["best_k"], 1)
        self.assertEqual(len(result["results"]), 3)

    def test_official_test_read_is_after_refit_and_only_one_evaluation_site(self):
        import inspect
        from huyen_code import ciciomt24_training as training
        source = inspect.getsource(training.run_official)
        self.assertLess(source.index("fitted = fit_final_classifier("),
                        source.index('store.load_full(config, "test"'))
        self.assertEqual(source.count("evaluate_final(fitted, test)"), 1)
        self.assertIn("test_evaluation_calls", source)
        self.assertIn('if out.exists() or out.with_name(out.name + ".incomplete").exists()', source)
        refit = inspect.getsource(training.fit_final_classifier)
        self.assertNotIn("select_mi(", refit)
        self.assertNotIn("rank_rfe(", refit)

    def test_incomplete_checkpoint_cannot_re_evaluate_test(self):
        from colab.checkpoint_manager import publish_completed_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoints" / "run01.incomplete"
            checkpoint.mkdir(parents=True)
            (checkpoint / "checkpoint.json").write_text('{"stage":"test_evaluation_completed"}')
            (checkpoint / "result.json").write_text('{"status":"INCOMPLETE_DO_NOT_USE"}')
            with self.assertRaises(ValueError), patch("colab.experiment_launcher.run_one",
                                                     side_effect=AssertionError("must not retrain")):
                publish_completed_checkpoint(root, "run01", {})


if __name__ == "__main__":
    unittest.main()
