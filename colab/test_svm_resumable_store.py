"""Small synthetic filesystem fixtures only. Written, NOT EXECUTED by Codex."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from colab.svm_resumable_store import CheckpointStore, atomic_json, file_sha, sync_file


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.binding = {"experiment": "mi_rfe_svm", "commit": "a", "dataset": "b",
                        "config": "FW_512_FS_16", "features": ["a", "b"]}
        self.store = CheckpointStore(self.root, self.binding, resume=True)

    def test_sync_uses_writable_non_truncating_descriptor_and_flushes_first(self):
        path = MagicMock()
        handle = path.open.return_value.__enter__.return_value
        handle.fileno.return_value = 123

        def check_sync(descriptor):
            self.assertEqual(descriptor, 123)
            handle.flush.assert_called_once_with()

        with patch("colab.svm_resumable_store.os.fsync", side_effect=check_sync) as fsync:
            sync_file(path)
        path.open.assert_called_once_with("r+b")
        fsync.assert_called_once_with(123)
        handle.write.assert_not_called()
        handle.truncate.assert_not_called()
        path.open.return_value.__exit__.assert_called_once()

    def test_sync_preserves_existing_artifact_bytes_and_checksum(self):
        path = Path(self.temp.name) / "artifact.bin"
        content = b"\x00synthetic-artifact\xff\n"
        path.write_bytes(content)
        before = file_sha(path)
        sync_file(path)
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(file_sha(path), before)

    def test_sync_failure_preserves_incomplete_stage_without_committing(self):
        with patch("colab.svm_resumable_store.sync_file",
                   side_effect=OSError("synthetic sync failure")):
            with self.assertRaisesRegex(OSError, "synthetic sync failure"):
                self.store.save("mi", {"features": ["a"]}, writers={
                    "scaler.bin": lambda p: p.write_bytes(b"synthetic-scaler")})
        self.assertIsNone(self.store.load("mi"))
        self.assertFalse(self.store.directory("mi").exists())
        incomplete = list((self.root / "checkpoints").glob("mi.incomplete-*"))
        self.assertEqual(len(incomplete), 1)
        self.assertTrue((incomplete[0] / "data.json").is_file())
        self.assertEqual((incomplete[0] / "scaler.bin").read_bytes(), b"synthetic-scaler")
        self.assertFalse((incomplete[0] / "checkpoint.json").exists())

    def test_mi_save_load_with_preprocessing_artifact(self):
        self.store.save("mi", {"features": ["a", "b"]}, writers={
            "scaler.bin": lambda p: p.write_bytes(b"synthetic-scaler")})
        resumed = CheckpointStore(self.root, self.binding.copy(), resume=True)
        self.assertEqual(resumed.data("mi")["features"], ["a", "b"])

    def test_rfe_requires_exact_mi_dependency(self):
        self.store.save("rfe", {"features": ["b", "a"]}, parents={"mi": "hash"})
        with self.assertRaises(ValueError):
            self.store.load("rfe", parents={"mi": "different"})

    def test_successful_k_is_immutable_and_checks_prefix(self):
        names = [f"f{i}" for i in range(100)]
        self.store.save("k_search/k_001", {
            "k": 1, "features": names[:1], "metrics": {"accuracy": .5}}, parents={"mi": "x"})
        self.assertEqual(len(self.store.completed_k({"mi": "x"}, names)), 1)
        with self.assertRaises(FileExistsError):
            self.store.save("k_search/k_001", {})
        with self.assertRaises(ValueError):
            self.store.completed_k({"mi": "x"}, list(reversed(names)))

    def test_interrupted_write_never_looks_complete(self):
        def interrupted(path):
            path.write_bytes(b"partial")
            raise RuntimeError("disconnect")
        with self.assertRaises(RuntimeError):
            self.store.save("mi", {}, writers={"object.bin": interrupted})
        self.assertIsNone(self.store.load("mi"))
        self.assertTrue(list((self.root / "checkpoints").glob("mi.incomplete-*")))
        self.store.save("mi", {"recomputed_uncommitted_stage": True})
        self.assertIsNotNone(self.store.load("mi"))

    def test_checksum_corruption_is_rejected(self):
        self.store.save("mi", {"features": ["a"]})
        (self.store.directory("mi") / "data.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.load("mi")

    def test_commit_dataset_config_feature_mismatch_is_rejected(self):
        for key in self.binding:
            with self.subTest(key=key), self.assertRaises(PermissionError):
                CheckpointStore(self.root, {**self.binding, key: "different"}, resume=True)

    def test_missing_manifest_and_non_resume_block(self):
        with self.assertRaises(FileExistsError):
            CheckpointStore(self.root, self.binding, resume=False)
        (self.root / "run_manifest.json").unlink()
        with self.assertRaises(ValueError):
            CheckpointStore(self.root, self.binding, resume=True)

    def test_resume_first_unfinished_k_and_reject_gaps(self):
        names = [f"f{i}" for i in range(100)]
        for k in range(1, 13):
            self.store.save(f"k_search/k_{k:03d}", {
                "k": k, "features": names[:k], "metrics": {"accuracy": .9}}, parents={"rfe": "same"})
        self.assertEqual(len(self.store.completed_k({"rfe": "same"}, names)) + 1, 13)
        self.store.save("k_search/k_014", {
            "k": 14, "features": names[:14], "metrics": {"accuracy": .9}}, parents={"rfe": "same"})
        with self.assertRaises(ValueError):
            self.store.completed_k({"rfe": "same"}, names)

    def test_test_requires_final_and_can_only_be_claimed_once(self):
        with self.assertRaises(ValueError):
            self.store.claim_test_once("missing", "session01")
        self.store.save("final", {"features": ["a"]})
        reference = self.store.reference("final")
        self.store.claim_test_once(reference, "session01")
        with self.assertRaises(FileExistsError):
            self.store.claim_test_once(reference, "session02")

    def test_partial_test_claim_also_blocks_retry(self):
        self.store.save("final", {})
        path = self.root / "results" / "test_evaluation_claim.json"
        path.parent.mkdir()
        path.write_text("{partial", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.store.claim_test_once(self.store.reference("final"), "next")


if __name__ == "__main__":
    unittest.main()
