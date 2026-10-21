"""Stage 7.1 unit tests. Do not execute automatically or fit real estimators."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ciciomt24_training as training
import ciciomt24_svm_probe as probe
from colab.experiment_launcher import RunSpec, svm_stage71_checks


class ProbeConfigurationTests(unittest.TestCase):
    def test_limits_cap_sample_time_rss_and_seed(self):
        probe.ProbeLimits().validate()
        invalid = (
            probe.ProbeLimits(sizes=(2000, 2000)),
            probe.ProbeLimits(sizes=(5000, 2000)),
            probe.ProbeLimits(sizes=(2000, 20000)),
            probe.ProbeLimits(seed=-1),
            probe.ProbeLimits(timeout_per_size_sec=1201),
            probe.ProbeLimits(max_rss_gib=17),
        )
        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                limits.validate()

    def test_seeded_stratified_internal_split_is_disjoint(self):
        labels = np.array([0] * 12 + [1] * 88, dtype=np.int8)
        first = probe.stratified_train_indices(labels, 40, seed=42)
        second = probe.stratified_train_indices(labels, 40, seed=42)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        fit, hold, selected = first
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(fit) + len(hold), 40)
        self.assertEqual(np.intersect1d(fit, hold).size, 0)
        self.assertEqual(set(labels[fit]), {0, 1})
        self.assertEqual(set(labels[hold]), {0, 1})

    def test_minority_floor_and_invalid_pool(self):
        labels = np.array([0] * 2 + [1] * 98, dtype=np.int8)
        fit, hold, _ = probe.stratified_train_indices(labels, 10, seed=42)
        self.assertIn(0, labels[fit])
        self.assertIn(0, labels[hold])
        with self.assertRaises(ValueError):
            probe.stratified_train_indices(np.array([0] + [1] * 99), 10, seed=42)

    def test_resource_guard_reasons(self):
        limits = probe.ProbeLimits()
        self.assertEqual(probe.admission_reason(9 * 1024**3, limits),
                         "INSUFFICIENT_AVAILABLE_RAM")
        self.assertIsNone(probe.admission_reason(11 * 1024**3, limits))
        self.assertEqual(probe.guard_reason(601, 0, 20 * 1024**3, limits), "TIMEOUT")
        self.assertEqual(probe.guard_reason(1, 9 * 1024**3, 20 * 1024**3, limits),
                         "RSS_CAP_EXCEEDED")
        self.assertEqual(probe.guard_reason(1, 0, 512 * 1024**2, limits),
                         "HOST_AVAILABLE_RAM_LOW")

    def test_valid_probe_cannot_unlock_full_svm(self):
        spec = RunSpec("baseline_svm", "FW_512_FS_16", "stage71_unit")
        artifact = {"schema": probe.SCHEMA, "status": "PASS_TECHNICAL_ONLY",
                    "experiment": spec.experiment, "config": spec.config,
                    "training_code_sha256": "training-hash",
                    "window_summary_sha256": "window-hash",
                    "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                    "full_training_authorized": False, "sizes": [2000],
                    "runs": [{"sample_rows": 2000, "status": "PASS_TECHNICAL_ONLY",
                              "worker_returncode": 0, "supervisor_stop_reason": None,
                              "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                              "fits": [{"fit": {"converged": True, "fit_status": 0}}]}]}
        checks = svm_stage71_checks(artifact, spec, "training-hash", "window-hash")
        self.assertTrue(checks["svm_technical_probe"])
        self.assertFalse(checks["svm_full_resource_gate_reviewed"])
        artifact["test_csv_rows_read"] = 1
        self.assertFalse(svm_stage71_checks(artifact, spec, "training-hash", "window-hash")
                         ["svm_technical_probe"])
        artifact["test_csv_rows_read"] = 0
        artifact["runs"] = []
        self.assertFalse(svm_stage71_checks(artifact, spec, "training-hash", "window-hash")
                         ["svm_technical_probe"])
        artifact["runs"] = [{"sample_rows": 2000, "status": "PASS_TECHNICAL_ONLY",
                             "worker_returncode": 0, "supervisor_stop_reason": None,
                             "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                             "fits": [{"fit": {"converged": True, "fit_status": 0}}]}]
        artifact.pop("schema")  # An old Stage 6 probe cannot unlock Stage 7.1.
        self.assertFalse(svm_stage71_checks(artifact, spec, "training-hash", "window-hash")
                         ["svm_technical_probe"])


class TrainOnlyProbeTests(unittest.TestCase):
    @staticmethod
    def _pool():
        X = np.arange(24 * 380, dtype=np.float32).reshape(24, 380)
        y = np.array([0] * 8 + [1] * 16, dtype=np.int8)
        return training.PreparedSplit("train", X, y)

    def _fake_store(self):
        calls = []
        pool = self._pool()
        lineage = [{"Window_ID": f"train-{i}", "Source_CSV_ID": i % 3}
                   for i in range(24)]

        class Store:
            def __init__(self, root, split_root):
                self.root, self.split_root = root, split_root

            def load_sample(self, config, role, rows):
                calls.append((config, role, rows))
                if role != "train":
                    raise AssertionError("Probe attempted to read Validation/Test")
                return pool, {"sample_method": "unit Train pool"}, lineage

        return Store, calls

    def test_baseline_worker_never_loads_validation_or_test(self):
        Store, calls = self._fake_store()
        fake_model = SimpleNamespace(predict=lambda X: np.ones(len(X), dtype=np.int8))
        fit_measure = {"seconds": .1, "peak_rss_bytes": 123,
                       "support_vectors": 4, "converged": True, "fit_status": 0}
        with patch.object(training, "WindowStore", Store), \
             patch.object(training, "make_model", return_value=fake_model) as make, \
             patch.object(training, "timed_fit", return_value=(fake_model, fit_measure)):
            result = probe.run_worker(Path("unused"), Path("manifest"),
                                      "FW_512_FS_16", "baseline_svm", 12, 24, 42)
        self.assertEqual(calls, [("FW_512_FS_16", "train", 24)])
        make.assert_called_once_with("svm", baseline=True)
        self.assertEqual(result["status"], "PASS_TECHNICAL_ONLY")
        self.assertEqual(result["validation_csv_rows_read"], 0)
        self.assertEqual(result["test_csv_rows_read"], 0)
        self.assertIsNone(result["official_best_k"])
        self.assertEqual(result["fits"][0]["fit"]["support_vectors"], 4)

    def test_compact_worker_uses_only_internal_train_holdout(self):
        Store, calls = self._fake_store()
        names = list(training.FEATURE_NAMES[:100])
        fake_model = SimpleNamespace(predict=lambda X: np.ones(len(X), dtype=np.int8))
        fit_measure = {"seconds": .1, "support_vectors": 3,
                       "converged": True, "fit_status": 0}
        fake_mi = {"names": names, "measurement": {"seconds": .01}}

        def scale(fit, hold, mi):
            self.assertEqual((fit.role, hold.role), ("train", "val"))
            return fit.X[:, :100], hold.X[:, :100], None

        with patch.object(training, "WindowStore", Store), \
             patch.object(training, "select_mi", return_value=fake_mi), \
             patch.object(training, "scale_top100", side_effect=scale), \
             patch.object(training, "rank_rfe", return_value={
                 "names": names, "measurement": {"seconds": .02}, "forest_fits": 4}), \
             patch.object(training, "make_model", return_value=fake_model), \
             patch.object(training, "timed_fit", return_value=(fake_model, fit_measure)):
            result = probe.run_worker(Path("unused"), Path("manifest"),
                                      "FW_512_FS_16", "mi_rfe_svm", 12, 24, 42)
        self.assertEqual(calls, [("FW_512_FS_16", "train", 24)])
        self.assertEqual([row["features"] for row in result["fits"]], [1, 10, 100])
        self.assertFalse(result["partial_rfe"]["complete_ranking"])
        self.assertEqual(result["validation_csv_rows_read"], 0)
        self.assertEqual(result["test_csv_rows_read"], 0)

    def test_admission_fail_creates_incomplete_report_without_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            out = base / "probe.json"
            with patch.object(training, "sha256", return_value="hash"), \
                 patch.object(training, "source_hashes", return_value={}), \
                 patch.object(training, "environment", return_value={}), \
                 patch.object(probe.psutil, "virtual_memory",
                              return_value=SimpleNamespace(available=9 * 1024**3,
                                                           total=20 * 1024**3)), \
                 patch.object(probe.subprocess, "Popen",
                              side_effect=AssertionError("worker should not start")):
                result = probe.run_probe(base, base, base, "FW_512_FS_16",
                                         "baseline_svm", out)
            self.assertEqual(result["status"], "FAIL_RESOURCE_ADMISSION")
            self.assertFalse(result["full_training_authorized"])
            self.assertEqual(json.loads(out.read_text())["test_csv_rows_read"], 0)
            with self.assertRaises(FileExistsError):
                probe.run_probe(base, base, base, "FW_512_FS_16", "baseline_svm", out)

    def test_supervisor_stops_only_its_worker_on_rss_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            out = base / "probe.json"
            worker = SimpleNamespace(pid=123, returncode=None)
            worker.poll = lambda: worker.returncode

            def stop_owned(target):
                self.assertIs(target, worker)
                target.returncode = -15

            with patch.object(training, "sha256", return_value="hash"), \
                 patch.object(training, "source_hashes", return_value={}), \
                 patch.object(training, "environment", return_value={}), \
                 patch.object(probe.psutil, "virtual_memory",
                              return_value=SimpleNamespace(available=20 * 1024**3)), \
                 patch.object(probe.subprocess, "Popen", return_value=worker) as spawned, \
                 patch.object(probe, "_tree_rss", return_value=9 * 1024**3), \
                 patch.object(probe, "_stop_own_worker", side_effect=stop_owned) as stopped:
                result = probe.run_probe(base, base, base, "FW_512_FS_16",
                                         "baseline_svm", out,
                                         probe.ProbeLimits(sizes=(2000,)))
            spawned.assert_called_once()
            stopped.assert_called_once_with(worker)
            self.assertEqual(result["status"], "REVIEW_REQUIRED_OR_FAILED")
            self.assertEqual(result["stop_reason"], "RSS_CAP_EXCEEDED")
            self.assertFalse(result["full_training_authorized"])


if __name__ == "__main__":
    unittest.main()
