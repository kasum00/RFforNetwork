import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from mi_rfe_svm_colab.common import Blocked, FORMAT, atomic, digest, info, read
from mi_rfe_svm_colab.drive_checkpoint import DriveCheckpoint, committed, latest, restore, unpack


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local, self.remote = self.root / "local", self.root / "drive/run"
        self.local.mkdir()
        self.identity = {"format": FORMAT, "sha256": "fixture"}
        self.state = self.local / "state"
        atomic(self.state / "run_identity.json", self.identity)
        self.now = 0.0
        self.manager = DriveCheckpoint(self.local, self.remote, self.identity, clock=lambda: self.now)

    def sync(self, **kw):
        return self.manager.sync(self.state, reason="test", safe_boundary=True, **kw)

    def test_no_drive_calls_before_18000_and_trigger_exactly_at_boundary(self):
        self.now = 17999
        with patch.object(self.manager, "assert_lease", side_effect=AssertionError("Drive accessed")):
            self.assertIsNone(self.sync())
        self.now = 18000
        row = self.sync()
        self.assertEqual(row["status"], "COMMITTED")
        self.assertEqual(row["seconds_since_previous_sync"], 18000)
        self.assertFalse(self.manager.due())

    def test_long_fit_does_not_upload_until_safe_boundary(self):
        self.now = 21000
        with self.assertRaises(Blocked):
            self.manager.sync(self.state, reason="during_fit", safe_boundary=False)
        self.assertEqual(committed(self.remote), [])
        self.assertEqual(self.sync()["interval_overrun_seconds"], 3000)

    def test_forced_stop_and_append_preserve_previous_bundle(self):
        first = self.sync(force=True)
        first_info = info(self.remote / "checkpoints" / (first["id"] + ".zip"))
        atomic(self.state / "progress.json", {"k": 61})
        second = self.sync(force=True)
        self.assertEqual(second["parent"], first["id"])
        self.assertEqual(len(committed(self.remote)), 2)
        self.assertEqual(info(self.remote / "checkpoints" / (first["id"] + ".zip")), first_info)

    def test_interrupted_upload_never_committed_and_old_checkpoint_retained(self):
        first = self.sync(force=True)
        with patch("mi_rfe_svm_colab.drive_checkpoint.shutil.copyfileobj", side_effect=OSError("disconnected")):
            with self.assertRaises(OSError):
                self.sync(force=True)
        self.assertEqual(latest(self.remote, "fixture")["id"], first["id"])

    def test_corrupt_latest_not_silently_rolled_back(self):
        self.sync(force=True)
        row = self.sync(force=True)
        with (self.remote / "checkpoints" / (row["id"] + ".zip")).open("ab") as f:
            f.write(b"corruption")
        with self.assertRaises(Blocked):
            latest(self.remote, "fixture")

    def test_resume_latest_committed_and_local_ancestry(self):
        atomic(self.state / "k/k_001.json", {"test": "state"})
        row = self.sync(force=True)
        another = self.root / "another"
        another.mkdir()
        restored = restore(another, self.remote, self.identity)
        self.assertEqual(read(restored / "k/k_001.json"), {"test": "state"})
        self.assertEqual(read(another / "sync_receipt.json")["id"], row["id"])
        self.assertEqual(restore(another, self.remote, self.identity), restored)

    def test_bundle_identity_and_checksum_mismatch(self):
        row = self.sync(force=True)
        with self.assertRaises(Blocked):
            unpack(self.remote / "checkpoints" / (row["id"] + ".zip"), self.root / "wrong", "wrong", max_bytes=10**9)

    def test_old_folder_and_active_job_protection(self):
        old = self.root / "old"
        old.mkdir()
        (old / "result.json").write_text("old")
        with self.assertRaises(Blocked):
            DriveCheckpoint(self.local, old, self.identity)
        self.assertEqual((old / "result.json").read_text(), "old")
        with self.assertRaises(Blocked):
            DriveCheckpoint(self.local, self.remote, self.identity)
        old_lease = self.manager.lease["id"]
        replacement = DriveCheckpoint(self.local, self.remote, self.identity,
            takeover="PREVIOUS RUNTIME STOPPED " + old_lease)
        with self.assertRaises(Blocked):
            self.manager.assert_lease()
        replacement.release("PAUSED")

    def test_completed_run_protected(self):
        atomic(self.remote / "RUN_COMPLETE.json", {"status": "COMPLETE"})
        with self.assertRaises(Blocked):
            DriveCheckpoint(self.local, self.remote, self.identity)
        with self.assertRaises(Blocked):
            self.sync(force=True)
        self.assertEqual(committed(self.remote), [])

    def test_training_arrays_never_bundled(self):
        (self.state / "train.npy").write_bytes(b"not allowed")
        with self.assertRaises(Blocked):
            self.sync(force=True)
