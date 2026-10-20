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
from colab.dataset_validator import (EXPECTED, _inventory_precheck, inspect_archive,
    manifest_root_for, prepare_dataset, prepare_drive_folder_dataset, prepare_mounted_folder_dataset,
    validate_dataset)
from colab.experiment_launcher import RunSpec, preflight, rf_empirical_required, run_one
from colab.setup_colab import repo_inventory, selected_versions
from colab.drive_manager import (FOLDER_MIME, SHORTCUT_MIME, discover_shared_dataset,
                                 inventory_window_files, list_children)


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
        self.assertIn("expected = selected_versions()", "".join(cells[4]["source"]))
        self.assertIn("discover_shared_dataset", "".join(cells[5]["source"]))
        self.assertIn("DATASET_READY = False", "".join(cells[5]["source"]))
        self.assertIn("DATASET_READY = DATASET_READY and", "".join(cells[6]["source"]))
        self.assertIn("if not DATASET_READY:", "".join(cells[8]["source"]))

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


class FakeRequest:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def execute(self):
        if self.error:
            raise self.error
        return self.value


class FakeDrive:
    def __init__(self, items, pages):
        self.items, self.pages = items, pages

    def files(self):
        return self

    def get(self, *, fileId, **unused):
        return FakeRequest(self.items[fileId])

    def list(self, *, q, pageToken=None, **unused):
        parent = q.split("'")[1]
        page = self.pages[(parent, pageToken)]
        return FakeRequest(page)


class FolderRegressionTests(unittest.TestCase):
    def _fake_drive(self, *, shortcut=False, paged=False):
        parent = {"id": "parent", "name": "shared", "mimeType": FOLDER_MIME}
        windows = {"id": "win", "name": "WINDOWS_CPLUS", "mimeType": FOLDER_MIME}
        audit = {"id": "audit", "name": "CSV_SPLIT_CPLUS", "mimeType": FOLDER_MIME}
        if shortcut:
            alias = {"id": "alias", "name": "WINDOWS_CPLUS", "mimeType": SHORTCUT_MIME,
                     "shortcutDetails": {"targetId": "win", "targetMimeType": FOLDER_MIME}}
        else:
            alias = windows
        if paged:
            pages = {("parent", None): {"files": [alias], "nextPageToken": "next"},
                     ("parent", "next"): {"files": [audit]}}
        else:
            pages = {("parent", None): {"files": [alias, audit]}}
        pages[("win", None)] = {"files": [
            {"id": "summary", "name": "window_run_summary.json", "mimeType": "application/json",
             "size": "2", "md5Checksum": "99914b932bd37a50b983c5e7c90ae93b"},
            {"id": "fw", "name": "FW_512_FS_8", "mimeType": FOLDER_MIME}]}
        pages[("fw", None)] = {"files": [
            {"id": "csv", "name": "example_windows.csv", "mimeType": "text/csv",
             "size": "3", "md5Checksum": "900150983cd24fb0d6963f7d28e17f72"}]}
        items = {item["id"]: item for item in (parent, windows, audit)}
        return FakeDrive(items, pages)

    def test_shared_parent_has_two_folders_and_no_zip(self):
        found = discover_shared_dataset("parent", service=self._fake_drive())
        self.assertEqual(found["mode"], "folder")
        self.assertTrue(found["audit_folder_present"])
        self.assertEqual(found["zips"], [])
        self.assertEqual(found["windows_folder"]["id"], "win")

    def test_api_only_folder_pagination_and_recursive_inventory(self):
        service = self._fake_drive(paged=True)
        found = discover_shared_dataset("parent", service=service)
        self.assertEqual(len(found["children"]), 2)
        paths = [x["relative_path"] for x in inventory_window_files(service, "win")]
        self.assertEqual(paths, ["FW_512_FS_8/example_windows.csv", "window_run_summary.json"])

    def test_google_drive_shortcut_resolves_target_id(self):
        found = discover_shared_dataset("parent", service=self._fake_drive(shortcut=True))
        self.assertEqual(found["windows_folder"]["id"], "win")

    def test_drive_permission_failure_is_explicit(self):
        class Denied(Exception):
            resp = type("Resp", (), {"status": 403})()
        service = self._fake_drive()
        service.get = lambda **unused: FakeRequest(error=Denied())
        with self.assertRaisesRegex(PermissionError, "Viewer access"):
            discover_shared_dataset("parent", service=service)

    def test_missing_fw_or_csv_and_metadata_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, "322"):
            _inventory_precheck([{"relative_path": "window_run_summary.json"}])
        one_config = [{"relative_path": "window_run_summary.json"}]
        one_config += [{"relative_path": f"FW_512_FS_8/seg_{index:03d}_windows.csv"}
                       for index in range(322)]
        with self.assertRaisesRegex(FileNotFoundError, "configuration folder"):
            _inventory_precheck(one_config)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "split_manifest"):
                manifest_root_for(Path(temp) / "WINDOWS_CPLUS", Path(temp))
            portable = Path(temp) / "configs" / "cplus_manifest"
            portable.mkdir(parents=True)
            (portable / "split_manifest.csv").write_text("fixture", encoding="utf-8")
            (portable / "validation.json").write_text("{}", encoding="utf-8")
            self.assertEqual(manifest_root_for(Path(temp) / "WINDOWS_CPLUS", Path(temp)), portable)

    def test_python_profiles_keep_stage5_and_accept_313(self):
        self.assertEqual(selected_versions((3, 12))["numpy"], "1.26.4")
        self.assertEqual(selected_versions((3, 13))["numpy"], "2.2.6")
        self.assertEqual(selected_versions((3, 13))["scikit-learn"], "1.7.2")
        with self.assertRaises(RuntimeError):
            selected_versions((3, 14))

    def test_mounted_windows_folder_staging_and_rerun_without_copy(self):
        with tempfile.TemporaryDirectory(dir=REPO / "colab") as temp:
            base = Path(temp)
            source = base / "drive" / "WINDOWS_CPLUS"
            (source / "FW_512_FS_8").mkdir(parents=True)
            (source / "window_run_summary.json").write_bytes(b"{}")
            (source / "FW_512_FS_8" / "example_windows.csv").write_bytes(b"abc")
            output = base / "content" / "CICIoMT24" / "WINDOWS_CPLUS"
            valid = {"status": "PASS_METADATA_SCHEMA", "csv_files": 322,
                     "config_directories": 4, "counts": EXPECTED, "missing_files": []}
            with patch("colab.dataset_validator._inventory_precheck"), \
                 patch("colab.dataset_validator._MIN_TRAINING_DISK_RESERVE", 0), \
                 patch("colab.dataset_validator.validate_dataset", return_value=valid):
                first = prepare_mounted_folder_dataset(REPO, source, output_root=output)
                self.assertEqual(first["action"], "prepared_verified")
                self.assertEqual((output / "FW_512_FS_8" / "example_windows.csv").read_bytes(), b"abc")
                with patch("shutil.copyfile", side_effect=AssertionError("re-download")):
                    second = prepare_dataset(REPO, output_root=output)  # Cell 6 rerun branch.
                self.assertEqual(second["action"], "reused_verified")

    def test_api_transfer_resume_verified_files_and_no_flow_csv(self):
        with tempfile.TemporaryDirectory(dir=REPO / "colab") as temp:
            base = Path(temp)
            output = base / "CICIoMT24" / "WINDOWS_CPLUS"
            entries = [{"id": "summary", "relative_path": "window_run_summary.json",
                        "size": 2, "md5Checksum": "99914b932bd37a50b983c5e7c90ae93b"},
                       {"id": "csv", "relative_path": "FW_512_FS_8/example_windows.csv",
                        "size": 3, "md5Checksum": "900150983cd24fb0d6963f7d28e17f72"}]
            payload = {"summary": b"{}", "csv": b"abc"}
            calls = []
            def fake_download(service, item, path):
                calls.append(item["id"])
                if item["id"] == "csv" and calls.count("csv") == 1:
                    raise ConnectionError("fixture interruption before this file")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload[item["id"]])
            valid = {"status": "PASS_METADATA_SCHEMA", "csv_files": 322,
                     "config_directories": 4, "counts": EXPECTED, "missing_files": []}
            with patch("colab.drive_manager.inventory_window_files", return_value=entries), \
                 patch("colab.drive_manager.download_drive_file", side_effect=fake_download), \
                 patch("colab.dataset_validator._inventory_precheck"), \
                 patch("colab.dataset_validator._MIN_TRAINING_DISK_RESERVE", 0), \
                 patch("colab.dataset_validator.validate_dataset", return_value=valid):
                from colab.dataset_validator import prepare_drive_folder_dataset as prepare
                with self.assertRaises(ConnectionError):
                    prepare(REPO, "win", object(), output_root=output)
                self.assertTrue((output.parent.with_name("CICIoMT24.incomplete") / "WINDOWS_CPLUS" /
                                 "window_run_summary.json").is_file())
                result = prepare(REPO, "win", object(), output_root=output)
                self.assertEqual(result["transferred_files"], 1)
                self.assertEqual(calls, ["summary", "csv", "csv"])
                self.assertFalse((output.parent / "CSV_SPLIT_CPLUS").exists())

    def test_unrecognized_incomplete_and_low_disk_stop(self):
        with tempfile.TemporaryDirectory(dir=REPO / "colab") as temp:
            base = Path(temp)
            source = base / "WINDOWS_CPLUS"
            source.mkdir()
            (source / "window_run_summary.json").write_bytes(b"{}")
            (source / "FW_512_FS_8").mkdir()
            (source / "FW_512_FS_8" / "example_windows.csv").write_bytes(b"abc")
            output = base / "new" / "CICIoMT24" / "WINDOWS_CPLUS"
            incomplete = output.parent.with_name("CICIoMT24.incomplete")
            incomplete.mkdir(parents=True)
            with patch("colab.dataset_validator._inventory_precheck"):
                with self.assertRaisesRegex(FileExistsError, "Unrecognized"):
                    prepare_mounted_folder_dataset(REPO, source, output_root=output)
            incomplete.rmdir()
            with patch("colab.dataset_validator._inventory_precheck"), \
                 patch("colab.dataset_validator._MIN_TRAINING_DISK_RESERVE", 8 * 1024**3), \
                 patch("colab.dataset_validator.shutil.disk_usage", return_value=type("Disk", (), {"free": 1})()):
                with self.assertRaisesRegex(OSError, "short by"):
                    prepare_mounted_folder_dataset(REPO, source, output_root=output)


if __name__ == "__main__":
    unittest.main()
