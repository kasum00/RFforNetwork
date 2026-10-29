import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from mi_rfe_svm_colab.common import Blocked, SVM_PARAMS, RFE_PARAMS, atomic, digest, info, read
from mi_rfe_svm_colab.dataset import cache_arrays, raw_split, svc_prefix
from mi_rfe_svm_colab.local_progress import KProgress, choose
from mi_rfe_svm_colab.runner import search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"sha256": "identity"}
        self.ranked = [f"f{i}" for i in range(100)]
        atomic(self.root / "rfe_state.json", {"fixture": True})
        self.X = np.arange(800, dtype=np.float32).reshape(8, 100)
        self.y = np.array([0, 1]*4)
        self.fits, self.predicts = [], []

    def run_fake(self, boundary=None, stop=lambda: False, fail_at=None):
        owner = self
        class FakeSVC:
            def predict(self, X):
                owner.predicts.append(X.shape[1])
                return owner.y.copy()
        def fit(model, X, y):
            self.assertTrue(X.flags.c_contiguous)
            self.assertEqual(X.dtype, np.float64)
            self.fits.append(X.shape[1])
            if X.shape[1] == fail_at:
                raise KeyboardInterrupt
            return model, {"seconds": 1.0, "converged": True, "warnings": []}
        return search(self.root, self.identity, self.ranked, self.X, self.y, self.X, self.y,
            Mock(), model_factory=FakeSVC, fit_fn=fit, metrics_fn=lambda y, p: {"accuracy": 0.5},
            boundary=boundary or Mock(), stop_requested=stop)

    def test_complete_100_once_and_resume_does_not_refit(self):
        rows = self.run_fake()
        self.assertEqual(self.fits, list(range(1, 101)))
        self.assertEqual(self.predicts, self.fits)
        self.assertEqual(choose(rows, self.ranked, complete=True)["k"], 1)
        self.assertEqual(self.run_fake(), rows)
        self.assertEqual(len(self.fits), 100)

    def test_resume_after_interrupted_fit_repeats_only_uncommitted_k(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(fail_at=3)
        self.assertEqual(self.fits, [1, 2, 3])
        self.assertEqual(self.predicts, [1, 2])
        self.run_fake()
        self.assertEqual(self.fits, [1, 2, 3] + list(range(3, 101)))
        self.assertEqual(self.predicts, list(range(1, 101)))

    def test_graceful_stop_forces_boundary_after_completed_k(self):
        boundary = Mock()
        rows = self.run_fake(boundary=boundary, stop=lambda: len(self.fits) >= 3)
        self.assertIsNone(rows)
        self.assertEqual(self.fits, [1, 2, 3])
        self.assertTrue(boundary.call_args.kwargs["force"])

    def test_local_progress_hash_and_identity_mismatch(self):
        history = KProgress(self.root, self.identity, self.ranked, "rfe")
        history.append({"k": 1, "features": self.ranked[:1], "accuracy": 0.5})
        self.assertEqual(len(KProgress(self.root, self.identity, self.ranked, "rfe").rows), 1)
        with self.assertRaises(Blocked):
            KProgress(self.root, {"sha256": "another"}, self.ranked, "rfe")
        record = read(self.root / "k/k_001.json")
        record["row"]["accuracy"] = 0.9
        atomic(self.root / "k/k_001.json", record)
        with self.assertRaises(Blocked):
            KProgress(self.root, self.identity, self.ranked, "rfe")

    def test_gap_and_incomplete_selection_blocked(self):
        with self.assertRaises(Blocked):
            choose([{"k": 2, "features": self.ranked[:2], "accuracy": 1}], self.ranked)
        with self.assertRaises(Blocked):
            choose([{"k": 1, "features": self.ranked[:1], "accuracy": 1}], self.ranked, complete=True)

    def test_no_test_before_claim(self):
        with patch("mi_rfe_svm_colab.dataset.core") as module:
            with self.assertRaises(Blocked):
                raw_split({}, {}, self.root, self.identity, "test")
            with self.assertRaises(Blocked):
                raw_split({}, {}, self.root, self.identity, "test", test_claim={})
            module.assert_not_called()

    def test_cache_builder_once_and_checksum_verified(self):
        build = Mock(return_value=({"X": self.X, "y": self.y}, {}))
        cache_arrays(self.root / "cache", {"id": 1}, build)
        cache_arrays(self.root / "cache", {"id": 1}, build)
        self.assertEqual(build.call_count, 1)
        with (self.root / "cache/X.npy").open("ab") as f:
            f.write(b"bad")
        with self.assertRaises(Blocked):
            cache_arrays(self.root / "cache", {"id": 1}, build)

    def test_prefix_exact_values_all_k_after_reference_float32_scaling(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.validation import check_array
        raw = np.random.default_rng(42).normal(size=(16, 380)).astype(np.float32)
        top = list(range(100, 200))
        scaled = StandardScaler().fit_transform(raw[:, top])
        order = list(reversed(range(100)))
        ranked = np.ascontiguousarray(scaled[:, order])
        for k in range(1, 101):
            reference_boundary = check_array(scaled[:, order[:k]], dtype=np.float64, order="C")
            actual = svc_prefix(ranked, k)
            np.testing.assert_array_equal(actual.view(np.uint64), reference_boundary.view(np.uint64))
            self.assertEqual(actual.var(), reference_boundary.var())

    def test_locked_parameters(self):
        from sklearn.svm import SVC
        self.assertEqual(SVM_PARAMS, dict(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced",
            shrinking=True, probability=False, cache_size=10000, max_iter=20000, random_state=42))
        self.assertEqual(RFE_PARAMS, dict(n_estimators=100, max_depth=15,
            class_weight="balanced", n_jobs=-1, random_state=42))
        self.assertEqual(SVC(**SVM_PARAMS).get_params()["cache_size"], 10000)
