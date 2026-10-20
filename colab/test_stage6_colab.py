"""Offline Stage 6 tests: no full fits, browser, Drive or Test feature reads."""
from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from colab.checkpoint_manager import CheckpointSink, publish_completed_checkpoint
from colab.dataset_validator import EXPECTED, inspect_archive, validate_dataset
from colab.experiment_launcher import RunSpec, preflight, rf_empirical_required, run_one
from colab.setup_colab import repo_inventory


REPO = Path(__file__).resolve().parents[1]


class NotebookAndManifestTests(unittest.TestCase):
    def test_notebook_twelve_code_cells_parse_and_default_dry(self):
        notebook = json.loads((REPO / "notebooks" / "COMPACT_MI_RFE_ColabPro.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(cells), 12)
        for cell in cells:
            ast.parse("".join(cell["source"]))
        self.assertIn('MODE = "dry_run"', "".join(cells[7]["source"]))
        self.assertIn("if MODE == 'full_train':", "".join(cells[9]["source"]))

    def test_repo_inventory(self):
        self.assertTrue(repo_inventory(REPO)["pass"])

    def test_approved_manifest_copy_is_semantically_identical(self):
        original = REPO / "CICIoMT24" / "CSV_SPLIT_CPLUS"
        portable = REPO / "configs" / "cplus_manifest"
        if not original.is_dir():
            self.skipTest("Original Windows metadata is absent from the Colab clone")
        with (original / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as left, \
             (portable / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as right:
            self.assertEqual(list(csv.DictReader(left)), list(csv.DictReader(right)))
        self.assertEqual(json.loads((original / "validation.json").read_text(encoding="utf-8")),
                         json.loads((portable / "validation.json").read_text(encoding="utf-8")))

    def test_real_dataset_metadata_and_schema_no_test_rows(self):
        root = REPO / "CICIoMT24" / "WINDOWS_CPLUS"
        if not root.is_dir():
            self.skipTest("Local Windows dataset unavailable")
        audit = validate_dataset(root, REPO)
        self.assertEqual(audit["status"], "PASS_METADATA_SCHEMA")
        self.assertEqual(audit["counts"], EXPECTED)
        self.assertEqual(audit["csv_files"], 322)
        self.assertEqual(audit["features"], 380)
        self.assertEqual(audit["test_feature_rows_read"], 0)


class SafetyTests(unittest.TestCase):
    def test_zip_slip_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../WINDOWS_CPLUS/window_run_summary.json", "{}")
            with self.assertRaises(ValueError):
                inspect_archive(path)

    def test_duplicate_archive_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("WINDOWS_CPLUS/window_run_summary.json", "{}")
                archive.writestr("WINDOWS_CPLUS/window_run_summary.json", "{}")
            with self.assertRaises(ValueError):
                inspect_archive(path)

    def test_fs8_guard_matches_stage54(self):
        self.assertEqual(rf_empirical_required(EXPECTED["FW_512_FS_8"]), 25_762_976_100)

    def test_run_spec_rejects_invalid_or_path_name(self):
        with self.assertRaises(ValueError):
            RunSpec("mi_rfe_rf", "FW_512_FS_8", "../evil").validate()
        with self.assertRaises(ValueError):
            RunSpec("xgb", "FW_512_FS_8", "run1").validate()

    def test_checkpoint_per_k_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=REPO / "colab") as temp:
            root = Path(temp)
            (root / "checkpoints").mkdir()
            (root / "experiment_results").mkdir()
            sink = CheckpointSink(root, "unit_run", root / "local", {"experiment": "mi_rfe_rf", "config": "FW_512_FS_8"})
            sink.record("k_completed", k=1, best_k=1, result={"accuracy": .9})
            self.assertEqual(json.loads((sink.dir / "checkpoint.json").read_text())["completed_k"], [1])
            with self.assertRaises(FileExistsError):
                CheckpointSink(root, "unit_run", root / "local", {"experiment": "mi_rfe_rf", "config": "FW_512_FS_8"})
            with self.assertRaises(ValueError):
                sink.record("k_completed", k=3, best_k=1, result={"accuracy": .9})

    def test_preflight_no_fit_and_no_test_feature_rows(self):
        root = REPO / "CICIoMT24" / "WINDOWS_CPLUS"
        if not root.is_dir():
            self.skipTest("Local Windows dataset unavailable")
        from colab.dataset_validator import sha256
        import sys
        sys.path.insert(0, str(REPO / "huyen_code"))
        import ciciomt24_training as training
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(training, "timed_fit", side_effect=AssertionError("preflight called model.fit")), \
                 patch.object(training, "select_mi", side_effect=AssertionError("preflight fit MI")):
                audit = preflight(REPO, root, Path(temp) / "runs", Path(temp) / "drive",
                                  RunSpec("mi_rfe_rf", "FW_512_FS_8", "dry_unit"),
                                  unit_test_evidence={"passed": True, "training_sha256": sha256(REPO / "huyen_code" / "ciciomt24_training.py")})
        self.assertEqual(audit["model_fit_calls"], 0)
        self.assertEqual(audit["test_feature_rows_read"], 0)
        self.assertEqual(audit["status"], "FAIL")  # Drive not mounted here.

    def test_adapter_persists_refit_before_test_and_restores_source_symbols(self):
        """Mock orchestration only; no estimator fit or Test CSV read."""
        import sys
        sys.path.insert(0, str(REPO / "huyen_code"))
        import ciciomt24_training as training
        original_symbols = (training.WindowStore, training.RunProgress, training.fit_final_classifier)
        with tempfile.TemporaryDirectory(dir=REPO / "colab") as temp:
            base = Path(temp)
            local, drive = base / "runs", base / "drive"
            for sub in ("experiment_results", "checkpoints", "environment"):
                (drive / sub).mkdir(parents=True, exist_ok=True)
            spec = RunSpec("baseline_svm", "FW_512_FS_8", "adapter_unit")
            fake_audit = {"status": "PASS", "checks": {"unit": True},
                          "dataset": {"dataset_fingerprint_sha256": "fakefixture",
                                      "fingerprint_scope": "unit fixture"},
                          "runtime": {"versions": {"scikit-learn": "unit"}}}

            def fake_fit(*args, **kwargs):
                return {"model": {"unit": True}, "scaler": {"unit": True}}

            def fake_official(windows, originals, output, config, experiment, *, confirm_full_training):
                self.assertTrue(confirm_full_training)
                staged = output.with_name(output.name + ".incomplete")
                staged.mkdir()
                training.write_json(staged / "preflight_audit.json", {"unit": True})
                training.write_json(staged / "selection_frozen.json", {"best_k": 380})
                with training.RunProgress(staged, config, experiment) as progress:
                    progress("selection_frozen")
                    training.fit_final_classifier(None, None, "svm", baseline=True)
                    progress("final_refit_completed")
                    self.assertTrue((drive / "checkpoints" / "adapter_unit.incomplete" / "model.joblib").is_file())
                    progress("test_evaluation_started", evaluation_number=1)
                    progress("test_evaluation_completed")
                    training.write_json(staged / "features.json", ["unit"])
                    result = {"status": "PASS", "experiment": experiment, "config": config,
                              "rows_by_split": EXPECTED[config], "test_evaluation_calls": 1,
                              "features": ["unit"]}
                    training.write_json(staged / "result.json", result)
                    progress("run_validated")
                os.replace(staged, output)
                return result

            with patch("colab.experiment_launcher.preflight", return_value=fake_audit), \
                 patch("colab.drive_manager.require_writable_result_root", return_value=drive), \
                 patch.object(training, "fit_final_classifier", fake_fit), \
                 patch.object(training, "run_official", fake_official):
                result = run_one(REPO, base / "not_used", local, drive, spec,
                                 unit_test_evidence={"passed": True},
                                 confirmation="RUN baseline_svm FW_512_FS_8 adapter_unit")
            self.assertEqual(result["status"], "PASS")
            self.assertTrue((drive / "experiment_results" / "adapter_unit" / "result.json").is_file())
            recovered = base / "recovered_drive"
            (recovered / "experiment_results").mkdir(parents=True)
            (recovered / "checkpoints").mkdir()
            shutil.copytree(drive / "checkpoints" / "adapter_unit.incomplete",
                            recovered / "checkpoints" / "adapter_unit.incomplete")
            path = publish_completed_checkpoint(recovered, "adapter_unit", EXPECTED[spec.config])
            self.assertTrue((path / "result.json").is_file())
            self.assertEqual((training.WindowStore, training.RunProgress,
                              training.fit_final_classifier), original_symbols)


if __name__ == "__main__":
    unittest.main()
