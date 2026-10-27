from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from colab.colab_pipeline import (
    ColabContractError,
    assert_single_dataset_root,
    backup_small_artifacts,
    build_paths,
    validate_mode,
)


class ColabPipelineUnitTests(unittest.TestCase):
    def test_default_modes_are_explicit(self):
        self.assertEqual(validate_mode("dry_run"), "dry_run")
        with self.assertRaises(ColabContractError):
            validate_mode("train_everything")

    def test_mixed_dataset_root_is_rejected(self):
        with self.assertRaises(ColabContractError):
            assert_single_dataset_root(Path("/content/CICIoMT24/WINDOWS_CPLUS"))

    def test_path_builder_keeps_stage_roots_separate(self):
        paths = build_paths(
            project_root=Path("/content/drive/MyDrive/COMPACT-MI-RFE"),
            dataset_root=Path("/content/drive/MyDrive/COMPACT-MI-RFE/CICFLOWMETER_FEATURE_CSV"),
            drive_root=Path("/content/drive/MyDrive"),
        )
        self.assertIn("stage4_mi", str(paths["stage4_runs_root"]))
        self.assertIn("stage5_rfe", str(paths["stage5_runs_root"]))

    def test_backup_copies_only_small_artifacts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            backup = root / "backup"
            source.mkdir()
            (source / "rfe_manifest.json").write_text("{}", encoding="utf-8")
            (source / "large_dataset.parquet").write_bytes(b"fixture")
            result = backup_small_artifacts(source, backup)
            self.assertTrue(result["dataset_files_copied"] is False)
            self.assertTrue((backup / source.name / "BACKUP_COMPLETE.json").exists())
            self.assertFalse((backup / source.name / "large_dataset.parquet").exists())
            with self.assertRaises(ColabContractError):
                backup_small_artifacts(source, backup)


if __name__ == "__main__":
    unittest.main()

