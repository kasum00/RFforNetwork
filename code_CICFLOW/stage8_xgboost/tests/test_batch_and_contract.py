"""Synthetic/mock-only admission, queue and no-Test/no-overwrite regressions."""
import json
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from stage8_xgboost.src import common, dataset_loader, drive_artifact_manager, runner, xgboost_preflight as admission

def request(root):
    return {"source": str(Path(__file__).resolve().parents[2]), "dataset": str(root / "CICFLOWMETER_FEATURE_CSV"),
            "result_root": str(root / "experiment_results"), "configs": list(common.CONFIGS),
            "experiment": "baseline_xgb", "mode": "full", "backend": "cpu", "resume": False,
            "run_tag": "run01", "git_commit": "a"*40, "budget": {
                "min_start_available_ram_bytes": 8*2**30, "min_runtime_available_ram_bytes": 4*2**30,
                "max_tree_rss_bytes": 40*2**30, "min_free_disk_bytes": 4*2**30,
                "max_elapsed_seconds": 43200, "min_gpu_free_bytes": 4*2**30}}

class BatchTests(unittest.TestCase):
    def test_exact_four_configurations_run_in_order_and_separate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = request(Path(tmp))
            base["confirmation"] = common.confirmation(base)
            seen = []
            def fake(action, item):
                seen.append((action, item["config"], str(common.output_path(item))))
                return {"program_status": "PASS", "exit_code": 0, "config": item["config"]}
            with patch.object(runner, "launch", side_effect=fake):
                result = runner.run_all(base)
            self.assertEqual([item[1] for item in seen], list(common.CONFIGS))
            self.assertEqual(len(set(item[2] for item in seen)), 4)
            self.assertTrue(all(item[0] == "run" for item in seen))
            self.assertEqual(result["status"], "PASS")

    def test_failure_does_not_silently_drop_other_configurations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = request(Path(tmp))
            base["confirmation"] = common.confirmation(base)
            def fake(action, item):
                return {"program_status": "FAIL" if item["config"] == common.CONFIGS[1] else "PASS",
                        "exit_code": 7 if item["config"] == common.CONFIGS[1] else 0, "config": item["config"]}
            with patch.object(runner, "launch", side_effect=fake) as launch:
                result = runner.run_all(base)
            self.assertEqual(launch.call_count, 4)
            self.assertEqual(result["status"], "INCOMPLETE_OR_FAILED")
            self.assertEqual(result["rows"][1]["exit_code"], 7)

    def test_preflight_batch_never_launches_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "mode": "preflight"}
            with patch.object(runner, "launch", side_effect=lambda action, r: {
                    "program_status": "PASS", "config": r["config"], "exit_code": 0}) as launch:
                runner.run_all(base)
            self.assertEqual([call.args[0] for call in launch.call_args_list], ["preflight"]*4)

    def test_duplicate_missing_or_extra_configs_are_blocked_before_launch(self):
        base = request(Path("/fixture"))
        for values in ([common.CONFIGS[0]]*4, list(common.CONFIGS)[:3], list(common.CONFIGS)+["other"]):
            with self.assertRaises(common.Blocked), patch.object(runner, "launch") as launch:
                runner.run_all({**base, "configs": values})
                launch.assert_not_called()

    def test_confirmation_is_bound_to_backend_budget_run_and_source_commit(self):
        base = request(Path("/fixture"))
        original = common.confirmation(base)
        for changed in ({"backend": "cuda"}, {"run_tag": "run02"}, {"git_commit": "b"*40},
                        {"budget": {**base["budget"], "max_elapsed_seconds": 3}}):
            self.assertNotEqual(original, common.confirmation({**base, **changed}))
        with patch.object(runner, "launch") as launch, self.assertRaises(common.Blocked):
            runner.run_all({**base, "confirmation": "wrong"})
        launch.assert_not_called()

    def test_verified_completed_config_is_skipped_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = request(Path(tmp))
            base["confirmation"] = common.confirmation(base)
            first = common.ensure_owned({**base, "config": common.CONFIGS[0]})
            common.write(first / "RUN_COMPLETE.json", {"fixture": True})
            before = (first / "RUN_COMPLETE.json").read_bytes()
            with patch.object(runner, "launch", side_effect=lambda action, r: {
                    "program_status": "PASS", "exit_code": 0, "config": r["config"]}) as launch, \
                 patch.object(runner.artifacts, "verify", return_value={"status": "PASS"}):
                runner.run_all(base)
            self.assertEqual([call.args[0] for call in launch.call_args_list], ["preflight", "run", "run", "run"])
            self.assertEqual((first / "RUN_COMPLETE.json").read_bytes(), before)

class ContractAndGuards(unittest.TestCase):
    def test_reference_parameters_and_device_are_exact(self):
        base = request(Path("/fixture"))
        with patch.dict("sys.modules", {"xgboost": types.SimpleNamespace(__version__="3.4.1", XGBClassifier=Mock())}):
            import xgboost
            admission.model_factory(base)
            kwargs = xgboost.XGBClassifier.call_args.kwargs
            self.assertEqual((kwargs["n_estimators"], kwargs["max_depth"], kwargs["min_child_weight"]), (200, 10, 3))
            self.assertEqual(kwargs["tree_method"], "hist")
            self.assertEqual(kwargs["device"], "cpu")
            self.assertNotIn("learning_rate", kwargs)  # preserve pinned library/source defaults
            self.assertNotIn("scale_pos_weight", kwargs)

    def test_gpu_cannot_fall_back_to_cpu(self):
        with self.assertRaises(common.Blocked):
            admission.assert_backend("cuda", "cpu")
        admission.assert_backend("cuda", "cuda:0")
        admission.assert_backend("cpu", "cpu")

    def test_test_cannot_be_read_during_validation_or_refit(self):
        with patch.object(dataset_loader, "load_core") as reader, self.assertRaises(common.Blocked):
            dataset_loader.load({}, {}, "test")
        reader.assert_not_called()

    def test_rf_or_foreign_outputs_cannot_be_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "config": common.CONFIGS[0]}
            with self.assertRaises(common.Blocked):
                common.ensure_owned({**base, "experiment": "mi_rfe_rf"})
            path = common.output_path(base)
            path.mkdir(parents=True)
            (path / "result.json").write_text("old result")
            with self.assertRaises(common.Blocked):
                common.ensure_owned(base)
            self.assertEqual((path / "result.json").read_text(), "old result")

    def test_completed_marker_cannot_be_created_on_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(drive_artifact_manager, "verify") as verify:
            with self.assertRaises(common.Blocked):
                drive_artifact_manager.complete_after_worker_exit(Path(tmp), 7)
            verify.assert_not_called()
            self.assertFalse((Path(tmp) / "RUN_COMPLETE.json").exists())

    def test_completion_marker_is_written_only_after_candidate_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common.write(root / "run_manifest.json", {"identity_sha256": "fixture"})
            common.write(root / "RUN_VALIDATED.json", {"fixture": True})
            observed = []
            def checked(directory, **kwargs):
                observed.append((root / "RUN_COMPLETE.json").exists())
                return {"status": "PASS"}
            with patch.object(drive_artifact_manager, "verify", side_effect=checked):
                result = drive_artifact_manager.complete_after_worker_exit(root, 0)
            self.assertEqual(observed, [False, True])
            self.assertEqual(result["status"], "PASS")

    def test_split_manifest_counts_and_feature_contract_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "config": common.CONFIGS[0]}
            folder = Path(base["dataset"]) / base["config"]
            folder.mkdir(parents=True)
            expected = common.EXPECTED[base["config"]]
            integrity = {"status": "PASS", "windows_by_split": expected,
                         "window_labels_by_split": {role: {"0": 1, "1": count-1} for role, count in expected.items()}}
            common.write(folder / "window_integrity.json", integrity)
            common.write(folder / "window_feature_schema.json", {"aggregate_feature_count": 380,
                         "output_label": "Window_Label", "binary_label_status": "PROVISIONAL_FOLDER"})
            core = Mock()
            core.metadata_audit.return_value = {"status": "PASS", "binding": {
                "names": [f"f{i}" for i in range(380)],
                "splits": {role: {"rows": count} for role, count in expected.items()}}}
            with patch.object(dataset_loader, "load_core", return_value=core):
                audit = dataset_loader.inspect(base)
                self.assertEqual(audit["binary_label_status"], "PROVISIONAL_FOLDER")
                core.read_values.assert_not_called()
                integrity["windows_by_split"] = {**expected, "train": 1}
                common.write(folder / "window_integrity.json", integrity, exclusive=False)
                with self.assertRaises(common.Blocked):
                    dataset_loader.inspect(base)

    def test_missing_artifacts_are_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(common.Blocked):
                drive_artifact_manager.verify(Path(tmp))

    def test_exclusive_files_and_checksum_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common.write(root / "artifact.json", {"value": 1})
            records = {"artifact.json": common.file_record(root / "artifact.json")}
            with self.assertRaises(FileExistsError):
                common.write(root / "artifact.json", {"value": 2})
            (root / "artifact.json").write_text("changed")
            with self.assertRaises(common.Blocked):
                common.check_files(root, records)

    def test_resource_fail_is_blocked_without_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "config": common.CONFIGS[0]}
            common.ensure_owned(base)
            audit = {"binding": {"splits": {"train": {"rows": 326157}}}}
            with patch("psutil.virtual_memory", return_value=types.SimpleNamespace(available=1)), \
                 patch.object(admission.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1)), \
                 patch.object(admission, "model_factory") as fit:
                result = admission.resources(base, audit)
            self.assertEqual(result["status"], "BLOCKED")
            fit.assert_not_called()

    def test_live_logging_records_stderr_and_preserves_native_failure_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "config": common.CONFIGS[0]}
            proc = Mock(stdin=io.StringIO(), stdout=io.StringIO("fixture stdout\n"), stderr=io.StringIO("fixture stderr\n"))
            proc.wait.return_value = 7
            with patch.object(runner.subprocess, "Popen", return_value=proc):
                result = runner.launch("tests", base)
            self.assertEqual(result["exit_code"], 7)
            self.assertEqual(result["program_status"], "FAIL")
            text = (common.output_path(base) / "logs/execution.log").read_text()
            self.assertIn("fixture stdout", text)
            self.assertIn("fixture stderr", text)

    def test_log_failure_prevents_child_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {**request(Path(tmp)), "config": common.CONFIGS[0]}
            real_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.name == "execution.log":
                    raise PermissionError("fixture locked log")
                return real_open(path, *args, **kwargs)
            with patch.object(Path, "open", guarded), patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(OSError, "LOG OPEN FAILED"):
                    runner.launch("tests", base)
                launch.assert_not_called()

    def test_new_notebook_defaults_to_preflight_and_has_fourteen_cells(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/CICFLOW_STAGE8_XGBOOST_COLAB.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 14)
        code = {cell["metadata"]["id"]: "".join(cell["source"]) for cell in cells}
        self.assertIn('MODE = "preflight"', code["cicflow-stage8-cell-8"])
        self.assertIn('BACKEND = "cpu"', code["cicflow-stage8-cell-8"])
        self.assertIn("runner.run_all(REQUEST)", code["cicflow-stage8-cell-9"])
        for number in (11, 12, 13, 14):
            self.assertNotIn(".predict(", code[f"cicflow-stage8-cell-{number}"])

if __name__ == "__main__":
    unittest.main()
