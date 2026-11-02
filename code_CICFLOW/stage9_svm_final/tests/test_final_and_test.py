import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from stage9_svm_final.common import Blocked, confirmation, digest, info, once, paths, read
from stage9_svm_final.data import load
from stage9_svm_final.metrics import compute
from stage9_svm_final.pipeline import (claim_state, execute, fit_final, test_once as evaluate_once,
                                       verify_final, verify_test)
from stage9_svm_final.store import bundle, copy_once, lease
from .fixtures import fixture, final_bundle


class FinalAndTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request, self.plan, self.matrices = fixture(self.root)
        self.identity = self.plan["identity"]
        self.local, self.remote = paths(self.request)

    def authorize_test(self):
        return {**self.request, "enable_test_evaluation": True,
                "test_confirmation": confirmation(self.request, "TEST_ONCE")}

    def test_train_and_validation_keep_selected_column_order(self):
        for role in ("train", "val"):
            X, y = load(self.request, self.identity, role)
            np.testing.assert_array_equal(X, self.matrices[role][0])
            np.testing.assert_array_equal(y, self.matrices[role][1])
            self.assertEqual(X.dtype, np.float32)

    def test_final_scaler_train_plus_val_and_no_test_open(self):
        import joblib
        import pyarrow.parquet as pq
        original = pq.read_table
        observed = []
        def reader(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test_windows.parquet")
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)
        with patch("pyarrow.parquet.read_table", side_effect=reader):
            final_bundle(self.request, self.plan)
        scaler = joblib.load(self.remote / "final_model/final_scaler.joblib")
        expected = np.concatenate([self.matrices[r][0] for r in ("train", "val")])
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0, dtype=np.float64))
        self.assertEqual(int(scaler.n_samples_seen_), 12)
        self.assertEqual(observed, ["train_windows.parquet", "val_windows.parquet"])
        self.assertEqual(read(self.remote / "final_model/final_state.json")["fit_rows"], 12)
        self.assertFalse((self.remote / "TEST_CLAIM.json").exists())

    def test_final_fit_requires_exact_authorization(self):
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("data before authorization")):
            with self.assertRaises(Blocked):
                fit_final(self.request, self.plan)

    def test_completed_final_resume_does_not_refit(self):
        final_bundle(self.request, self.plan)
        before = info(self.remote / "final_model/model.joblib")
        request = {**self.request, "enable_final_fit": True, "final_confirmation": confirmation(self.request, "FINAL_FIT")}
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("unnecessary refit")):
            self.assertEqual(fit_final(request, self.plan)["status"], "PASS")
        self.assertEqual(info(self.remote / "final_model/model.joblib"), before)

    def test_resume_valid_local_fit_after_publication_failure_without_refitting(self):
        with patch("stage9_svm_final.pipeline.publish_bundle", side_effect=OSError("upload interrupted")):
            with self.assertRaises(OSError):
                final_bundle(self.request, self.plan)
        saved = list(self.local.glob("fit_*/final_checkpoint.json"))
        self.assertEqual(len(saved), 1)
        before = info(saved[0].parent / "model.joblib")
        request = {**self.request, "enable_final_fit": True, "final_confirmation": confirmation(self.request, "FINAL_FIT")}
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("valid fit should be reused")):
            self.assertEqual(fit_final(request, self.plan)["status"], "PASS")
        self.assertEqual(before, info(self.remote / "final_model/model.joblib"))
        self.assertFalse((self.remote / "TEST_CLAIM.json").exists())

    def test_partial_remote_without_local_bundle_is_preserved_and_blocked(self):
        once(self.remote / "identity_manifest.json", self.identity)
        target = self.remote / "final_model/model.joblib"
        target.parent.mkdir()
        target.write_bytes(b"partial preserved")
        request = {**self.request, "enable_final_fit": True, "final_confirmation": confirmation(self.request, "FINAL_FIT")}
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("unsafe refit over partial bundle")):
            with self.assertRaisesRegex(Blocked, "not committed"):
                fit_final(request, self.plan)
        self.assertEqual(target.read_bytes(), b"partial preserved")

    def test_interrupted_final_fit_restarts_final_only_when_no_test_claim(self):
        from mi_rfe_svm_colab.common import core
        from stage9_svm_final.telemetry import Heartbeat
        module = core(self.request["source"])
        request = {**self.request, "enable_final_fit": True, "final_confirmation": confirmation(self.request, "FINAL_FIT")}
        with patch.object(module, "timed_fit", side_effect=KeyboardInterrupt), patch.object(Heartbeat, "emit"):
            with self.assertRaises(KeyboardInterrupt):
                fit_final(request, self.plan)
        self.assertFalse(list(self.local.glob("fit_*/final_checkpoint.json")))
        self.assertFalse((self.remote / "TEST_CLAIM.json").exists())
        with patch.object(module, "saved_mi", side_effect=AssertionError("MI must not rerun")), \
             patch.object(module, "saved_rfe", side_effect=AssertionError("RFE must not rerun")):
            self.assertEqual(final_bundle(self.request, self.plan)["status"], "PASS")

    def test_no_test_io_before_claim(self):
        with patch("stage9_svm_final.data.sha", side_effect=AssertionError("Test hash before claim")), \
             patch("pyarrow.parquet.read_table", side_effect=AssertionError("Test read before claim")):
            with self.assertRaisesRegex(Blocked, "claim"):
                load(self.request, self.identity, "test")

    def test_test_confirmation_required_before_any_data(self):
        final_bundle(self.request, self.plan)
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("unsafe Test access")):
            with self.assertRaisesRegex(Blocked, "confirmation"):
                evaluate_once(self.request, self.plan)
        self.assertFalse((self.remote / "TEST_CLAIM.json").exists())

    def test_claim_is_durable_before_test_hash_footer_and_load(self):
        from stage9_svm_final.data import sha as actual_sha
        final_bundle(self.request, self.plan)
        def check_hash(path):
            if Path(path).name == "test_windows.parquet":
                claim = read(self.remote / "TEST_CLAIM.json")
                self.assertEqual(claim["identity"], self.identity["sha256"])
                self.assertEqual(claim["evaluation_number"], 1)
            return actual_sha(path)
        with patch("stage9_svm_final.data.sha", side_effect=check_hash), \
             patch("sklearn.svm.SVC.predict", return_value=np.array([0, 0, 1, 1])) as predict:
            outcome = evaluate_once(self.authorize_test(), self.plan)
        self.assertEqual(outcome["status"], "PASS")
        predict.assert_called_once()
        self.assertEqual(outcome["result"]["metrics"]["accuracy"], 0.5)

    def test_test_completed_returns_verified_artifacts_without_second_predict(self):
        final_bundle(self.request, self.plan)
        with patch("sklearn.svm.SVC.predict", return_value=np.array([0, 0, 1, 1])):
            evaluate_once(self.authorize_test(), self.plan)
        before = info(self.remote / "test_metrics.json")
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("second Test load")), \
             patch("sklearn.svm.SVC.predict", side_effect=AssertionError("second Test predict")):
            self.assertEqual(evaluate_once(self.request, self.plan)["status"], "PASS")
            self.assertEqual(verify_test(self.remote, self.identity)["status"], "PASS")
        self.assertEqual(before, info(self.remote / "test_metrics.json"))

    def test_interruption_after_claim_is_never_retried(self):
        final_bundle(self.request, self.plan)
        with patch("stage9_svm_final.pipeline.load", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                evaluate_once(self.authorize_test(), self.plan)
        self.assertEqual(claim_state(self.remote), "BLOCKED_AMBIGUOUS_TEST_CLAIM")
        with patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("retry forbidden")):
            with self.assertRaisesRegex(Blocked, "retry forbidden"):
                evaluate_once(self.authorize_test(), self.plan)

    def test_claim_write_failure_prevents_test_read(self):
        final_bundle(self.request, self.plan)
        from stage9_svm_final.pipeline import once as actual_once
        def fail_claim(path, value):
            if Path(path).name == "TEST_CLAIM.json":
                raise OSError("synthetic disk failure")
            return actual_once(path, value)
        with patch("stage9_svm_final.pipeline.once", side_effect=fail_claim), \
             patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("Test without claim")):
            with self.assertRaises(OSError):
                evaluate_once(self.authorize_test(), self.plan)

    def test_incomplete_test_publication_is_blocked_without_predict_retry(self):
        final_bundle(self.request, self.plan)
        with patch("sklearn.svm.SVC.predict", return_value=np.array([0, 0, 1, 1])), \
             patch("stage9_svm_final.pipeline.publish_bundle", side_effect=OSError("publication interrupted")):
            with self.assertRaises(OSError):
                evaluate_once(self.authorize_test(), self.plan)
        with patch("sklearn.svm.SVC.predict", side_effect=AssertionError("must never retry")):
            with self.assertRaises(Blocked):
                evaluate_once(self.authorize_test(), self.plan)
        self.assertTrue(list(self.local.glob("test_*/test_checkpoint.json")))

    def test_corrupt_model_sha_blocks_before_test_claim(self):
        final_bundle(self.request, self.plan)
        with (self.remote / "final_model/model.joblib").open("ab") as f:
            f.write(b"corrupted")
        with self.assertRaisesRegex(Blocked, "SHA"):
            evaluate_once(self.authorize_test(), self.plan)
        self.assertFalse((self.remote / "TEST_CLAIM.json").exists())

    def test_test_checksum_failure_after_claim_cannot_be_retried(self):
        final_bundle(self.request, self.plan)
        with Path(self.identity["dataset"]["binding"]["splits"]["test"]["path"]).open("ab") as f:
            f.write(b"tamper")
        with self.assertRaisesRegex(Blocked, "file changed"):
            evaluate_once(self.authorize_test(), self.plan)
        self.assertTrue((self.remote / "TEST_CLAIM.json").exists())

    def test_metric_recomputation_and_sha_tampering(self):
        final_bundle(self.request, self.plan)
        with patch("sklearn.svm.SVC.predict", return_value=np.array([0, 0, 1, 1])):
            evaluate_once(self.authorize_test(), self.plan)
        (self.remote / "test_metrics.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(Blocked):
            verify_test(self.remote, self.identity)

    def test_method_config_identity_mismatch_cannot_reuse_final(self):
        final_bundle(self.request, self.plan)
        other = copy.deepcopy(self.identity)
        other["method"] = "B"
        other["sha256"] = digest({k: v for k, v in other.items() if k != "sha256"})
        with self.assertRaises(Blocked):
            verify_final(self.remote, other)

    def test_copy_and_json_never_overwrite(self):
        a, b = self.root / "a", self.root / "b"
        a.write_bytes(b"old")
        b.write_bytes(b"protected")
        with self.assertRaises(Blocked):
            copy_once(a, b)
        self.assertEqual(b.read_bytes(), b"protected")
        once(self.root / "record.json", {"old": True})
        with self.assertRaises(FileExistsError):
            once(self.root / "record.json", {"old": False})

    def test_unreleased_lease_cannot_auto_takeover(self):
        final_bundle(self.request, self.plan)
        from stage9_svm_final.common import utc
        once(self.remote / "leases/LEASE_000002.json", dict(identity=self.identity["sha256"], token="old", started=utc()))
        with self.assertRaisesRegex(Blocked, "Unreleased"):
            with lease(self.remote, self.identity):
                self.fail("Unreleased lease admitted worker")

    def test_preflight_only_does_not_fit_or_read_test(self):
        with patch("stage9_svm_final.pipeline.review", return_value=self.plan), \
             patch("stage9_svm_final.pipeline.fit_final", side_effect=AssertionError("fit in preflight")), \
             patch("stage9_svm_final.pipeline.load", side_effect=AssertionError("values in preflight")):
            self.assertEqual(execute({**self.request, "mode": "preflight"})["status"], "PASS")

    def test_unknown_selection_training_modes_are_rejected(self):
        for mode in ("mi", "rfe", "run", "k-search", "final-and-test"):
            with patch("stage9_svm_final.pipeline.review", side_effect=AssertionError("unsafe mode reached gate")):
                with self.assertRaises(Blocked):
                    execute({"mode": mode})

    def test_explicit_positive_label_and_macro_are_independent(self):
        y, p = [3, 3, 3, 7], [3, 3, 7, 7]
        value = compute(y, p, dict(benign=3, attack=7))
        self.assertEqual(value["positive_label"], 7)
        self.assertAlmostEqual(value["f1_attack"], 2/3)
        self.assertAlmostEqual(value["macro_f1"], (0.8+2/3)/2)
        self.assertEqual(value["confusion_matrix"], [[2, 1], [0, 1]])
