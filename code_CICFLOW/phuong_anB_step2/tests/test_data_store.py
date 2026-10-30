import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from phuong_anB_step2.contract import Blocked, atomic, digest, info, paths, read, sha
from phuong_anB_step2.dataset import _inspect_bound, load_selection, validate_dataset_reference
from phuong_anB_step2.store import Store, ledger
from mi_rfe_svm_colab.preflight import resource_gate
from .fixtures import SOURCE, dataset_fixture, small_identity


class DataStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_dataset_train_val_footer_and_no_test_open_or_hash(self):
        import pyarrow.parquet as pq
        request, parent, proof, folder = dataset_fixture(self.root)
        with patch("phuong_anB_step2.dataset.evidence", return_value=(parent, proof)), \
             patch("phuong_anB_step2.dataset.sha", wraps=sha) as hashes, \
             patch("pyarrow.parquet.ParquetFile", wraps=pq.ParquetFile) as footers:
            audit = _inspect_bound(request)
        self.assertEqual(audit["binding"]["splits"]["train"]["rows"], 8)
        self.assertEqual(audit["binding"]["splits"]["val"]["rows"], 4)
        self.assertFalse(audit["binding"]["splits"]["test"]["sha256_verified"])
        test = folder / "test_windows.parquet"
        self.assertNotIn(test, [Path(c.args[0]) for c in hashes.call_args_list])
        self.assertNotIn(test, [Path(c.args[0]) for c in footers.call_args_list])

    def test_test_value_loader_is_rejected_before_any_core_or_cache(self):
        with patch("phuong_anB_step2.dataset.core") as core, \
             patch("phuong_anB_step2.dataset.cache_arrays") as cache:
            with self.assertRaises(Blocked):
                load_selection({}, {}, self.root, {}, "test")
        core.assert_not_called(); cache.assert_not_called()

    def test_corrupt_validation_is_blocked_before_any_footer(self):
        request, parent, proof, folder = dataset_fixture(self.root)
        with (folder / "val_windows.parquet").open("ab") as handle:
            handle.write(b"tampered")
        with patch("phuong_anB_step2.dataset.evidence", return_value=(parent, proof)), \
             patch("pyarrow.parquet.ParquetFile", side_effect=AssertionError("footer too early")), \
             self.assertRaisesRegex(Blocked, "SHA256 differs: val"):
            _inspect_bound(request)

    def test_frozen_dataset_reference_blocks_changed_sidecar(self):
        _, _, _, folder = dataset_fixture(self.root)
        from phuong_anB_step2.dataset import SIDECARS
        integrity, schema = read(folder / "window_integrity.json"), read(folder / "window_feature_schema.json")
        expected = {"sidecars": {n: info(folder / n) for n in SIDECARS},
                    "binding_sha256": integrity["binding_sha256"], "windows_by_split": integrity["windows_by_split"],
                    "feature_count": 380, "label": schema["output_label"],
                    "model_sha256": {r: integrity["output_sha256"][r]["model_sha256"] for r in ("train", "val", "test")}}
        validate_dataset_reference(folder, expected)
        (folder / "window_counts_by_segment.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(Blocked):
            validate_dataset_reference(folder, expected)

    def test_b_paths_never_target_a1_and_a1_run_id_is_rejected(self):
        request = {"source": str(SOURCE), "dataset": str(self.root / "CICFLOWMETER_FEATURE_CSV"),
                   "output": str(self.root / "experiment_results"), "local_root": str(self.root / "local"),
                   "config": "FW256_FS8", "run_id": "RUN_B2_01", "production": False}
        local, remote = paths(request)
        self.assertIn("phuong_anB_step2", remote.parts)
        self.assertNotIn("phuong_anA", remote.parts)
        protected = self.root / "experiment_results/phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_01"
        protected.mkdir(parents=True)
        sentinel = protected / "sentinel.json"
        sentinel.write_text("unchanged", encoding="utf-8")
        request["run_id"] = "RUN_A1_01"
        with self.assertRaises(Blocked):
            paths(request)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_new_run_guard_and_identity_mismatch_leave_old_checkpoint_unchanged(self):
        identity = small_identity()
        store = Store(self.root / "local", self.root / "drive", identity, resume=False)
        atomic(store.state / "init.json", {"identity": identity["sha256"]})
        cp = store.commit("init", {"init.json": store.state / "init.json"})
        original = info(cp["path"] / "checkpoint.json")
        for field, changed in (("source", {"sha256": "changed"}), ("dataset_fingerprint", "changed"),
                               ("algorithm", "A1"), ("rfe_step", 5), ("run_id", "RUN_B2_OTHER"),
                               ("config", "FW512_FS16"), ("contract", {"rfe": {"step": 5}})):
            foreign = {**identity, field: changed}
            foreign["sha256"] = digest({k: v for k, v in foreign.items() if k != "sha256"})
            with self.assertRaises(Blocked):
                Store(self.root / ("local_" + field), store.remote, foreign, resume=True)
        with self.assertRaises(Blocked):
            Store(self.root / "duplicate", store.remote, identity, resume=False)
        self.assertEqual(info(cp["path"] / "checkpoint.json"), original)

    def test_draft_grid_or_step1_ranking_identity_is_never_admitted(self):
        identity = small_identity()
        for field, value in (
            ("search_policy", {**identity["search_policy"], "coarse_grid": list(range(100, 1, -2))+[1]}),
            ("search_policy", {**identity["search_policy"], "fine_radius": 2}),
            ("algorithm", "MI_RFE_SVM_B_STEP2_COARSE_FINE_V1"),
            ("format", "phuong_anB_step2.selection.v1"),
            ("rfe_step", 1),
        ):
            foreign = {**identity, field: value}
            foreign["sha256"] = digest({k: v for k, v in foreign.items() if k != "sha256"})
            with self.assertRaises(Blocked):
                Store(self.root / "foreign_local", self.root / "foreign_drive", foreign, resume=False)
        self.assertFalse((self.root / "foreign_drive").exists())

    def test_committed_corruption_and_duplicate_stage_are_rejected(self):
        store = Store(self.root / "local", self.root / "drive", small_identity(), resume=False)
        atomic(store.state / "init.json", {"value": 1})
        cp = store.commit("init", {"init.json": store.state / "init.json"})
        with self.assertRaises(Blocked):
            store.commit("init", {"init.json": store.state / "init.json"})
        (cp["path"] / "init.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(Blocked):
            ledger(store.remote, store.identity)

    def test_incomplete_upload_is_ignored_and_preserved(self):
        store = Store(self.root / "local", self.root / "drive", small_identity(), resume=False)
        incomplete = store.remote / "checkpoints/.incomplete_interrupted"
        incomplete.mkdir(parents=True)
        (incomplete / "partial.json").write_text("partial")
        self.assertEqual(ledger(store.remote, store.identity), [])
        self.assertTrue(incomplete.is_dir())

    def test_resource_failure_blocks_without_changing_counts_or_budget(self):
        counts = {"train": 8, "val": 4, "test": 4}
        budget = {"min_available_ram_bytes": 2*1024**3, "max_process_rss_bytes": 40*1024**3}
        sample = {"available_ram_bytes": 0, "disk_free_bytes": 128*1024**3, "logical_cpus": 2, "rss_bytes": 1}
        result = resource_gate(counts, budget, sample=sample)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(counts, {"train": 8, "val": 4, "test": 4})
        self.assertEqual(budget["max_process_rss_bytes"], 40*1024**3)
