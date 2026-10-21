"""Stage 7.2 synthetic/mocked tests. Never launches a real SVM or reads dataset."""
from __future__ import annotations

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ciciomt24_svm_probe as stage71
import ciciomt24_svm_scaling_probe as scaling
import ciciomt24_training as training
from colab.experiment_launcher import RunSpec, svm_stage71_checks


def complete_row(size=16_000, experiment="baseline_svm"):
    fit_rows = size * 4 // 5
    hold_rows = size - fit_rows
    fit = {"seconds": 1.0, "peak_rss_bytes": 10_000,
           "support_vectors": 100, "converged": True, "fit_status": 0,
           "n_iter": [10], "warnings": []}
    row = {"schema": scaling.SCHEMA, "experiment": experiment,
           "sample_rows_requested": size, "sample_rows_actual": size,
           "fit_rows": fit_rows, "internal_train_holdout_rows": hold_rows,
           "feature_count": 380, "label_origin": "SCENARIO_PROXY",
           "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
           "sampled_training": True, "scientific_result": False,
           "full_training_authorized": False, "official_best_k": None,
           "peak_process_rss_bytes": 11_000,
           "sample": {"requested_rows": size, "actual_rows": size,
                      "sample_window_ids_sha256": "a" * 64,
                      "source_count": 2, "scenario_count": 2,
                      "sample_by_proxy_label": {"benign": 40, "attack": size - 40},
                      "population_by_proxy_label": {"benign": 3182, "attack": 1_345_135}},
           "fit_by_proxy_label": {"benign": 32, "attack": fit_rows - 32},
           "holdout_by_proxy_label": {"benign": 8, "attack": hold_rows - 8}}
    ks = [380] if experiment == "baseline_svm" else [1, 10, 100]
    row["fits"] = [{"features": k, "fit": dict(fit),
                    "predict": {"rows": hold_rows, "seconds": .1}} for k in ks]
    if experiment == "mi_rfe_svm":
        measured = {"seconds": 2.0, "peak_rss_bytes": 12_000}
        row["mi"] = dict(measured)
        row["partial_rfe"] = {"measurement": dict(measured),
                              "forest_fits": 4, "complete_ranking": False}
    return row


class ScalingConfigurationTests(unittest.TestCase):
    def test_notebook_stage72_cells_parse_and_default_off(self):
        notebook = json.loads((REPO / "notebooks" / "COMPACT_MI_RFE_ColabPro.ipynb")
                              .read_text(encoding="utf-8"))
        stage = {cell.get("id"): "".join(cell.get("source", []))
                 for cell in notebook["cells"] if str(cell.get("id", "")).startswith("stage72-")}
        self.assertEqual(set(stage), {"stage72-guide", "stage72-tests", "stage72-config",
                                      "stage72-run", "stage72-review", "stage72-backup"})
        for cell in notebook["cells"]:
            if cell.get("id") in stage and cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        for assignment in ("STAGE72_RUN_TESTS = False", "STAGE72_ENABLE = False",
                           "STAGE72_RUN_REVIEW = False", "STAGE72_BACKUP_TO_DRIVE = False"):
            self.assertTrue(any(assignment in source for source in stage.values()))

    def test_stage71_defaults_unchanged_and_scaling_requires_opt_in(self):
        self.assertEqual(stage71.ProbeLimits().sizes, (2000, 5000, 8192))
        self.assertEqual(scaling.ScalingLimits().sizes, (16000, 32000, 64000))
        with self.assertRaises(PermissionError), patch.object(scaling, "_write_new") as write:
            scaling.run_scaling_probe(Path("repo"), Path("windows"), Path("manifest"),
                                      "FW_256_FS_8", "baseline_svm", Path("new.json"))
        write.assert_not_called()

    def test_cap_size_order_timeout_and_rss(self):
        for sizes in ((16000, 64001), (32000, 16000), (16000, 16000), (8192,)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                scaling.ScalingLimits(sizes=sizes).validate()
        for limits in (scaling.ScalingLimits(timeout_per_size_sec=59),
                       scaling.ScalingLimits(max_rss_gib=33),
                       scaling.ScalingLimits(seed=-1)):
            with self.assertRaises(ValueError):
                limits.validate()
        self.assertEqual(scaling.HARD_CAP, 64000)

    def test_class_ratio_floor_and_shortfall(self):
        self.assertEqual(sum(scaling.class_quotas({0: 3182, 1: 1_345_135}, 16000).values()), 16000)
        self.assertEqual(scaling.class_quotas({0: 2, 1: 100000}, 16000)[0], 2)
        with self.assertRaises(ValueError):
            scaling.class_quotas({0: 1, 1: 100000}, 16000)
        with self.assertRaises(ValueError):
            scaling.class_quotas({0: 2, 1: 100}, 16000)

    def test_seeded_nested_nonduplicate_positions(self):
        files = [SimpleNamespace(label=0, count=200),
                 SimpleNamespace(label=1, count=2000),
                 SimpleNamespace(label=1, count=2000)]
        small, _, _ = scaling.selected_positions(files, 100, 42)
        repeat, _, _ = scaling.selected_positions(files, 100, 42)
        large, _, _ = scaling.selected_positions(files, 200, 42)
        self.assertEqual(sum(map(len, small.values())), 100)
        for index in small:
            np.testing.assert_array_equal(small[index], repeat[index])
            self.assertEqual(len(np.unique(small[index])), len(small[index]))
            self.assertTrue(set(small[index]) <= set(large[index]))

    def test_timeout_rss_and_host_ram_stop(self):
        limits = scaling.ScalingLimits(max_rss_gib=4)
        self.assertEqual(scaling.admission_reason(5 * scaling.GIB, limits),
                         "INSUFFICIENT_AVAILABLE_RAM")
        self.assertIsNone(scaling.admission_reason(10 * scaling.GIB, limits))
        self.assertEqual(scaling.guard_reason(1201, 0, 20 * scaling.GIB, limits), "TIMEOUT")
        self.assertEqual(scaling.guard_reason(0, 5 * scaling.GIB, 20 * scaling.GIB, limits),
                         "RSS_CAP_EXCEEDED")
        with patch.object(scaling.psutil, "virtual_memory",
                          return_value=SimpleNamespace(total=20 * scaling.GIB)):
            self.assertEqual(scaling.guard_reason(0, 0, scaling.GIB, limits),
                             "HOST_AVAILABLE_RAM_LOW")

    def test_stage72_json_cannot_unlock_stage71_full_gate(self):
        spec = RunSpec("baseline_svm", "FW_256_FS_8", "stage72_mock")
        checks = svm_stage71_checks({"schema": scaling.SCHEMA, "status": "PASS_TECHNICAL_ONLY",
                                     "sizes": [16000], "runs": []}, spec, "a", "b")
        self.assertFalse(checks["svm_technical_probe"])
        self.assertFalse(checks["svm_full_resource_gate_reviewed"])


class ScalingEvidenceTests(unittest.TestCase):
    def test_required_measurements_and_no_test_rows(self):
        for experiment in ("baseline_svm", "mi_rfe_svm"):
            row = complete_row(experiment=experiment)
            self.assertTrue(scaling.evidence_complete(row))
            self.assertGreater(row["fits"][0]["support_vector_fraction"], 0)
            row["fits"][0]["fit"].pop("support_vectors")
            self.assertFalse(scaling.evidence_complete(row))
        row = complete_row()
        row["test_csv_rows_read"] = 1
        self.assertFalse(scaling.evidence_complete(row))
        row = complete_row(experiment="mi_rfe_svm")
        row["partial_rfe"]["complete_ranking"] = True
        self.assertFalse(scaling.evidence_complete(row))
        for key in ("sample", "fits", "mi", "partial_rfe"):
            row = complete_row(experiment="mi_rfe_svm")
            row[key] = None
            self.assertFalse(scaling.evidence_complete(row))

    def test_no_overwrite_existing_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.json"
            scaling._write_new(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                scaling._write_new(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text())["status"], "first")

    def test_loader_train_only_deterministic_exact_count_and_schema(self):
        train_b = SimpleNamespace(role="train", label=0, count=10, source_id=1,
                                  path=Path("benign.csv"), scenario="Benign")
        train_a = SimpleNamespace(role="train", label=1, count=10, source_id=2,
                                  path=Path("attack.csv"), scenario="Attack")
        val = SimpleNamespace(role="val", label=0, count=10, source_id=3,
                              path=Path("val.csv"), scenario="Benign")
        test = SimpleNamespace(role="test", label=1, count=10, source_id=4,
                               path=Path("test.csv"), scenario="Attack")
        reads = []

        class Store:
            def __init__(self, root, split_root):
                self.files = {"FW_256_FS_8": [train_b, train_a, val, test]}

            def validate_header(self, path):
                self.assert_train(path)

            @staticmethod
            def assert_train(path):
                if path.name not in ("benign.csv", "attack.csv"):
                    raise AssertionError("Validation/Test was opened")

            def validate_chunk(self, frame, file, offset):
                self.assert_train(file.path)

        def reader(path, **kwargs):
            Store.assert_train(path)
            reads.append(path.name)
            label = 0 if path.name == "benign.csv" else 1
            return iter([pd.DataFrame({"Window_ID": [f"{path.stem}-{i}" for i in range(10)],
                                       "Label": [label] * 10})])

        def prepared(frame, role):
            return training.PreparedSplit(role, np.zeros((len(frame), 380), dtype=np.float32),
                                          frame["Label"].to_numpy(dtype=np.int8))

        with patch.object(scaling, "SIZES", (8, 16, 32)), \
             patch.object(scaling.training, "WindowStore", Store), \
             patch.object(scaling.pd, "read_csv", side_effect=reader), \
             patch.object(scaling.training, "prepare_xy", side_effect=prepared):
            sample, info = scaling.load_scaling_sample(Path("unused"), Path("manifest"),
                                                        "FW_256_FS_8", 8, 42)
            repeat, info2 = scaling.load_scaling_sample(Path("unused"), Path("manifest"),
                                                         "FW_256_FS_8", 8, 42)
            with self.assertRaises(ValueError):
                scaling.load_scaling_sample(Path("unused"), Path("manifest"),
                                             "FW_256_FS_8", 32, 42)
        self.assertEqual(len(sample.y), 8)
        self.assertEqual(sample.X.shape[1], 380)
        self.assertEqual(info["sample_window_ids_sha256"], info2["sample_window_ids_sha256"])
        self.assertEqual(info["source_count"], 2)
        self.assertNotIn("val.csv", reads)
        self.assertNotIn("test.csv", reads)

    def test_source_hash_mismatch_before_loader(self):
        with patch.object(scaling.training, "sha256", return_value="actual"), \
             patch.object(scaling, "load_scaling_sample") as loader:
            with self.assertRaises(ValueError):
                scaling.run_worker(Path("windows"), Path("manifest"),
                                   "FW_256_FS_8", "baseline_svm", 16000, 42,
                                   {"training": "expected"})
        loader.assert_not_called()

    def test_feature_schema_mismatch_and_duplicate_window_ids_fail(self):
        with patch.object(scaling.training, "FEATURE_NAMES", ("only_one",)):
            with self.assertRaisesRegex(ValueError, "380"):
                scaling.load_scaling_sample(Path("unused"), Path("manifest"),
                                            "FW_256_FS_8", 16000, 42)
        files = [SimpleNamespace(role="train", label=label, count=4,
                                 source_id=label + 1, path=Path(f"{label}.csv"),
                                 scenario=str(label)) for label in (0, 1)]

        class Store:
            def __init__(self, root, split_root):
                self.files = {"FW_256_FS_8": files}

            def validate_header(self, path):
                pass

            def validate_chunk(self, frame, file, offset):
                pass

        def reader(path, **kwargs):
            return iter([pd.DataFrame({"Window_ID": ["duplicate"] * 4,
                                       "Label": [int(path.stem)] * 4})])

        def prepared(frame, role):
            return training.PreparedSplit(role, np.zeros((len(frame), 380), dtype=np.float32),
                                          frame["Label"].to_numpy(dtype=np.int8))

        with patch.object(scaling, "SIZES", (8, 16000, 32000, 64000)), \
             patch.object(scaling.training, "WindowStore", Store), \
             patch.object(scaling.pd, "read_csv", side_effect=reader), \
             patch.object(scaling.training, "prepare_xy", side_effect=prepared):
            with self.assertRaisesRegex(ValueError, "repeated Window_ID"):
                scaling.load_scaling_sample(Path("unused"), Path("manifest"),
                                            "FW_256_FS_8", 8, 42)

    def test_fail_worker_stops_before_larger_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "scaling.json"
            sizes = (16000, 32000)
            failed = {"schema": scaling.SCHEMA, "status": "WORKER_FAILED"}

            def spawn(command, cwd):
                worker_output = Path(command[command.index("--output") + 1])
                worker_output.write_text(json.dumps(failed), encoding="utf-8")
                return SimpleNamespace(poll=lambda: 1, pid=123)

            with patch.object(scaling.tempfile, "gettempdir", return_value=temp), \
                 patch.object(scaling.training, "sha256", return_value="hash"), \
                 patch.object(scaling.training, "source_hashes", return_value={}), \
                 patch.object(scaling.training, "environment", return_value={}), \
                 patch.object(scaling.psutil, "virtual_memory",
                              return_value=SimpleNamespace(available=30 * scaling.GIB,
                                                           total=32 * scaling.GIB)), \
                 patch.object(scaling.subprocess, "Popen", side_effect=spawn) as popen:
                result = scaling.run_scaling_probe(root, root, root, "FW_256_FS_8",
                                                   "baseline_svm", output,
                                                   scaling.ScalingLimits(sizes=sizes),
                                                   scaling_opt_in=True)
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(result["status"], "REVIEW_REQUIRED_OR_FAILED")
            self.assertEqual(len(result["runs"]), 1)
            self.assertFalse(result["full_training_authorized"])

    def test_review_json_only_no_fit_or_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline.json"
            compact = root / "compact.json"
            baseline.write_text(json.dumps({"schema": scaling.SCHEMA,
                                            "experiment": "baseline_svm",
                                            "config": "FW_256_FS_8", "runs": [complete_row()]}))
            compact.write_text(json.dumps({"schema": scaling.SCHEMA,
                                           "experiment": "mi_rfe_svm",
                                           "config": "FW_256_FS_8",
                                           "runs": [complete_row(experiment="mi_rfe_svm")]}))
            with patch.object(training, "WindowStore", side_effect=AssertionError("CSV read")), \
                 patch.object(training, "timed_fit", side_effect=AssertionError("model fit")):
                review = scaling.render_resource_review(baseline, compact)
            self.assertIn("REVIEW_REQUIRED", review)
            self.assertIn("NOT Test accuracy", review)


if __name__ == "__main__":
    unittest.main()
