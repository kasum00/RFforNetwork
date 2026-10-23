"""Synthetic fixtures/mocks ONLY. Written by Codex, NOT EXECUTED by Codex."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from colab.svm_resumable_store import CheckpointStore, atomic_json, digest, file_sha
from local import checkpoint_migration as migration
from local.paths import CONFIG, ORIGIN_COMMIT, RUN_NAME, logical_path, absolute
from local.windows_preflight import (PINS, check_environment, checksum_index, resources,
                                     validate_budgets, verify_content, inventory_fingerprint)
from local.local_authorization import validate_local_context, phrase


def fixture(root: Path, completed=3):
    names = [f"feature{i}" for i in range(380)]
    binding = {"identity": {"experiment": "mi_rfe_svm", "config": CONFIG, "run_name": RUN_NAME,
                             "git_commit": ORIGIN_COMMIT}, "feature_names": names,
               "scientific_contract": {"classifier": "synthetic metadata only"},
               "seed": 42, "preprocessing": "canonical float32 Inf/NaN->0; Train-only MI and Top100 scalers",
               "rows_by_split": {"train": 673559, "val": 224373, "test": 224287}}
    store = CheckpointStore(root, binding, resume=True)
    store.save("data_prepared", {"train_rows": 673559, "val_rows": 224373, "feature_names": names,
                                "test_feature_rows_read": 0,
                                "label_counts": {"train": {"benign": 1575, "attack": 671984},
                                                 "val": {"benign": 505, "attack": 223868}}})
    store.save("mi", {"features": names[:100], "indices": list(range(100)), "fit_role": "train"},
               parents={"data": store.reference("data_prepared")}, writers={
                   "mi.joblib": lambda p: p.write_bytes(b"not-a-real-pickle"),
                   "top100_scaler.joblib": lambda p: p.write_bytes(b"not-a-real-pickle"),
                   "mi_feature_scores.csv": lambda p: p.write_text("synthetic", encoding="utf-8")})
    store.save("rfe", {"features": names[:100], "complete_ranking": True, "forest_fits": 100},
               parents={"mi": store.reference("mi")}, writers={
                   "rfe_feature_ranking.csv": lambda p: p.write_text("synthetic", encoding="utf-8")})
    parents = {"mi": store.reference("mi"), "rfe": store.reference("rfe")}
    for k in range(1, completed + 1):
        store.save(f"k_search/k_{k:03d}", {"k": k, "features": names[:k],
            "metrics": {"accuracy": .8}, "train_rows": 673559, "validation_rows": 224373}, parents=parents)
    return store, names, parents


class CheckpointMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "origin"
        self.store, self.names, self.parents = fixture(self.root)

    def test_scan_next_k_is_first_unfinished_not_hardcoded_76(self):
        scan = migration.scan_checkpoints(self.root)
        self.assertEqual((scan["completed_k"], scan["next_k"]), (3, 4))
        self.assertEqual((scan["mi"], scan["rfe"]), ("PASS_REUSE", "PASS_REUSE"))

    def test_import_preserves_all_bytes_hidden_file_and_empty_directory(self):
        (self.root / ".hidden").write_bytes(b"hidden")
        (self.root / "empty").mkdir()
        before = migration.tree_files(self.root)
        destination = Path(self.temp.name) / "imported"
        result = migration.copy_checkpoint_tree(self.root, destination)
        self.assertEqual(result["status"], "IMPORTED_BYTES_VERIFIED")
        self.assertEqual(migration.tree_files(destination), before)
        self.assertEqual(migration.tree_files(self.root), before)
        self.assertTrue((destination / "empty").is_dir())

    def test_existing_import_target_never_overwritten(self):
        with self.assertRaises(migration.MigrationBlocked):
            migration.copy_checkpoint_tree(self.root, self.root)

    def test_completed_checkpoint_corruption_blocks_not_skips(self):
        path = self.store.directory("k_search/k_003") / "data.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            migration.scan_checkpoints(self.root)

    def test_gap_blocks_instead_of_repeating_or_skipping(self):
        self.store.save("k_search/k_005", {"k": 5, "features": self.names[:5],
            "metrics": {"accuracy": .9}, "train_rows": 673559, "validation_rows": 224373}, parents=self.parents)
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            migration.scan_checkpoints(self.root)

    def test_partial_next_k_is_preserved_and_not_completed(self):
        partial = self.store.directory("k_search/k_004.incomplete-dead")
        partial.mkdir()
        (partial / "partial.bin").write_bytes(b"unfinished")
        scan = migration.scan_checkpoints(self.root)
        self.assertEqual(scan["next_k"], 4)
        self.assertIn("k_search/k_004.incomplete-dead", scan["incomplete_directories_preserved_ignored"])
        self.assertEqual((partial / "partial.bin").read_bytes(), b"unfinished")

    def test_completed_name_without_manifest_blocks(self):
        self.store.directory("k_search/k_004").mkdir()
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            migration.scan_checkpoints(self.root)

    def test_missing_mi_never_triggers_training(self):
        with patch.object(CheckpointStore, "reference", side_effect=ValueError("missing MI")):
            with self.assertRaises(ValueError):
                migration.scan_checkpoints(self.root)

    def test_feature_order_mismatch_blocks(self):
        with self.assertRaisesRegex(ValueError, "feature order"):
            migration.scan_checkpoints(self.root, expected_names=list(reversed(self.names)))

    def test_scientific_contract_mismatch_blocks(self):
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            migration.scan_checkpoints(self.root, expected_contract={"changed": True})

    def test_test_claim_without_result_blocks_before_any_loader(self):
        (self.root / "results").mkdir()
        (self.root / "results" / "test_evaluation_claim.json").write_text("{partial", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Test claimed"):
            migration.scan_checkpoints(self.root)

    def test_scanner_never_mutates_origin_manifest_or_binding(self):
        before = file_sha(self.root / "run_manifest.json")
        migration.scan_checkpoints(self.root)
        self.assertEqual(file_sha(self.root / "run_manifest.json"), before)

    def test_all100_select_smallest_k_and_reuse_final_checkpoint(self):
        from colab.svm_resumable_training import choose_best
        for k in range(4, 101):
            self.store.save(f"k_search/k_{k:03d}", {"k": k, "features": self.names[:k],
                "metrics": {"accuracy": .8}, "train_rows": 673559, "validation_rows": 224373}, parents=self.parents)
        rows = self.store.completed_k(self.parents, self.names[:100])
        best = choose_best(rows)
        self.assertEqual(best["k"], 1)
        self.store.save("selection", {"best_k": 1, "features": self.names[:1]},
                        parents={**self.parents, "k_history": digest(rows)})
        self.store.save("final", {"features": self.names[:1], "fit_role": "train+val", "fit_rows": 897932,
                                  "mi_rfe_refit": False},
                        parents={"selection": self.store.reference("selection")}, writers={
                            "final_model.joblib": lambda p: p.write_bytes(b"synthetic"),
                            "final_scaler.joblib": lambda p: p.write_bytes(b"synthetic")})
        scan = migration.scan_checkpoints(self.root)
        self.assertEqual(scan["next_stage"], "final_test")
        self.assertIsNone(scan["next_k"])


class PortabilityAndResourceTests(unittest.TestCase):
    def test_colab_logical_fingerprint_is_identical_under_windows_paths(self):
        import hashlib
        linux = [("FW_512_FS_16/train/A/a.csv", 10), ("FW_512_FS_16/val/A/b.csv", 20)]
        windows = [(name.replace("/", "\\"), size) for name, size in reversed(linux)]
        old_payload = {"manifest_sha256": "m", "validation_sha256": "v", "summary_sha256": "s",
                       "inventory_sizes": sorted(linux)}
        original = hashlib.sha256(json.dumps(old_payload, sort_keys=True).encode()).hexdigest()
        self.assertEqual(inventory_fingerprint("m", "v", "s", windows), original)
        self.assertNotEqual(inventory_fingerprint("m", "changed", "s", windows), original)

    def test_source_origin_mismatch_blocks_without_loading_scientific_code(self):
        with self.assertRaisesRegex(ValueError, "origin commit"):
            migration.verify_sources(Path("unused"), {"identity": {"git_commit": "wrong"}})

    def test_logical_relative_path_is_os_independent(self):
        self.assertEqual(logical_path(r"FW_512_FS_16\train\A\file.csv"), "FW_512_FS_16/train/A/file.csv")
        for path in ("../file", "/content/file", r"D:\file", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                logical_path(path)
        with self.assertRaises(ValueError):
            absolute("relative")

    def test_blocked_budget_template_cannot_admit(self):
        from local.windows_preflight import BUDGET_FIELDS
        with self.assertRaises(ValueError):
            validate_budgets({name: None for name in BUDGET_FIELDS})

    def test_environment_requires_pinned_packages_windows_x64(self):
        good = {"os_name": "nt", "bits": 64, "python_major_minor": [3, 13], "versions": PINS.copy()}
        check_environment(good)
        for value in ({**good, "os_name": "posix"}, {**good, "bits": 32},
                      {**good, "versions": {**PINS, "numpy": "wrong"}}):
            with self.assertRaises(ValueError):
                check_environment(value)

    def test_resource_guard_ram_and_disk_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = SimpleNamespace(checkpoint_root=Path(temp), output_root=Path(temp))
            budgets = {"max_elapsed_seconds": 36000, "max_tree_rss_bytes": 400,
                       "min_start_available_ram_bytes": 600, "min_runtime_available_ram_bytes": 100,
                       "min_free_disk_bytes": 100, "poll_interval_seconds": 10}
            disk = lambda path: SimpleNamespace(free=1000)
            self.assertEqual(resources(paths, budgets, 300, vm=SimpleNamespace(available=1000, total=2000), disk_usage=disk)["status"], "PASS")
            self.assertEqual(resources(paths, budgets, 300, vm=SimpleNamespace(available=400, total=2000), disk_usage=disk)["status"], "RESOURCE_ADMISSION_FAIL")

    def test_preflight_hashes_train_val_but_never_test_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = []
            for role in ("train", "val", "test"):
                rel = f"{CONFIG}/{role}/A/a_windows.csv"
                path = root / rel
                path.parent.mkdir(parents=True)
                path.write_bytes(role.encode())
                items.append({"area": "windows", "path": rel, "bytes": path.stat().st_size, "sha256": file_sha(path)})
            payload = {"schema": "stage73b.ssd.file-sha256.v1", "files": items}
            calls = []
            def hashed(path):
                self.assertNotIn("test", path.parts)
                calls.append(path)
                return file_sha(path)
            with patch("local.windows_preflight.file_sha", side_effect=hashed):
                verify_content(SimpleNamespace(data_root=root), payload, roles=("train", "val"))
            self.assertEqual(len(calls), 2)

    def test_duplicate_normalized_dataset_records_rejected(self):
        row = {"area": "windows", "path": r"A\B", "bytes": 1, "sha256": "a" * 64}
        with self.assertRaises(ValueError):
            checksum_index({"schema": "stage73b.ssd.file-sha256.v1", "files": [row, {**row, "path": "A/B"}]})

    def test_sealed_local_authorization_cannot_be_plain_object(self):
        with self.assertRaises(PermissionError):
            validate_local_context(object(), Path("repo"), Path("data"), Path("output"))

    def test_local_confirmation_binds_source_session_and_original_binding(self):
        audit = {"scan": {"binding": {"identity": {"run_name": RUN_NAME}}, "binding_sha256": "a" * 64},
                 "git": {"commit": "b" * 40}}
        result = phrase(audit, "session02")
        for value in (RUN_NAME, "session02", "a" * 64, "b" * 40):
            self.assertIn(value, result)

    def test_infrastructure_ast_bridge_accepts_only_three_adapters(self):
        old = ('def run_resumable(repo, windows, local_runs, authorization):\n'
               ' context = validate_authorization(authorization)\n'
               ' deadline = SessionDeadline(1, 1)\n'
               ' with DurableProgress(root, sid, 60) as p:\n'
               '  return WindowStore(windows, split_root=manifest_root_for(windows, repo))\n')
        new = ('def run_resumable(repo, windows, local_runs, authorization):\n'
               ' context = validate_authorization(authorization)\n'
               ' return _execute_resumable(repo, windows, local_runs, context)\n'
               'def run_windows_resumable(repo, windows, local_runs, authorization):\n pass\n'
               'def _execute_resumable(repo, windows, local_runs, context, *, manifest_root=None, progress_factory=None, deadline_factory=None):\n'
               ' deadline = (deadline_factory or SessionDeadline)(1, 1)\n'
               ' with (progress_factory or DurableProgress)(root, sid, 60) as p:\n'
               '  return WindowStore(windows, split_root=manifest_root or manifest_root_for(windows, repo))\n')
        migration.verify_engine_bridge(old, new)
        with self.assertRaises(ValueError):
            migration.verify_engine_bridge(old, new.replace('SessionDeadline)(1, 1)', 'SessionDeadline)(2, 1)'))
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            migration.verify_engine_bridge(old, new + '\ndef _execute_resumable():\n return None\n')
        with self.assertRaises(ValueError):
            migration.verify_engine_bridge(old, new.replace('def _execute_resumable(', '@unchecked\ndef _execute_resumable('))


if __name__ == "__main__":
    unittest.main()
