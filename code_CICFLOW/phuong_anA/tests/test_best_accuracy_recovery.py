"""Real Telemetry + search + Store, fake SVC, temporary artifacts only."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from .fixtures import SOURCE, small_identity
from .test_a1 import TiedForest
from phuong_anA.contract import ALGORITHM, Blocked, atomic, core, digest, info, read, scientific, sha
from phuong_anA.pipeline import run
from phuong_anA.recovery import inspect_run
from phuong_anA.rfe import fit_step5
from phuong_anA.store import Store
from phuong_anA.telemetry import Telemetry
from mi_rfe_svm_colab.local_progress import KProgress
from mi_rfe_svm_colab.runner import search


class BestAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = [f"f{i}" for i in range(380)]
        self.top = self.names[:100]
        source_files = {"synthetic.py": "a"*64}
        binding = {"names": self.names, "splits": {"train": {"rows": 8}, "val": {"rows": 4}, "test": {"rows": 4}}}
        audit = {"binding": binding}
        audit["fingerprint"] = digest(audit)
        self.identity = small_identity(source={"files": source_files, "sha256": digest(source_files)},
            dataset=binding, audit=audit, dataset_fingerprint=audit["fingerprint"], contract=scientific(SOURCE))
        self.store = Store(self.root / "local", self.root / "drive", self.identity, resume=False)
        state = self.store.state
        atomic(state / "run_identity.json", self.identity)
        atomic(state / "mi_top100.json", self.top)
        with (state / "mi_feature_scores.csv").open("x", encoding="utf-8") as stream:
            stream.write(",0\n" + "".join(f"{name},{380-i}\n" for i, name in enumerate(self.names)))
        atomic(state / "mi_state.json", {"identity": self.identity["sha256"], "files": {
            name: info(state / name) for name in ("mi_top100.json", "mi_feature_scores.csv")}})
        (state / "top100_scaler.joblib").write_bytes(b"opaque synthetic scaler; never deserialize")
        atomic(state / "top100_scaler_state.json", {"binding": {"identity": self.identity["sha256"],
            "mi": sha(state / "mi_state.json"), "top": self.top}, "fit_role": "train", "rows": 8,
            "files": {"top100_scaler.joblib": info(state / "top100_scaler.joblib")}})
        rfe = fit_step5(np.ones((8, 100), np.float32), np.array([0, 1]*4), self.top, TiedForest())
        with (state / "rfe_feature_ranking.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rfe["records"][0]))
            writer.writeheader(); writer.writerows(rfe["records"])
        atomic(state / "rfe_state.json", {"binding": {"identity": self.identity["sha256"],
            "mi": sha(state / "mi_state.json"), "scaler": sha(state / "top100_scaler_state.json"), "top": digest(self.top)},
            "result": rfe, "files": {"rfe_feature_ranking.csv": info(state / "rfe_feature_ranking.csv")}})
        self.ranked = rfe["names"]
        self.store.commit("rfe", {p.name: p for p in state.iterdir() if p.is_file()})
        self.request = {"config": "FW256_FS8", "run_id": "RUN_A1_UNIT",
            "runtime": {"heartbeat_seconds": 15, "session_budget_seconds": 0, "stop_margin_seconds": 0},
            "budget": {"min_available_ram_bytes": 1024**3, "max_process_rss_bytes": 40*1024**3}}
        self.progress = Telemetry(self.store, self.request)
        self.progress.check = Mock()  # Resource sampling is tested separately; do not admit a real fit here.
        self.X, self.y, self.val_y = np.ones((8, 100), np.float32), np.array([0, 1]*4), np.array([0, 1, 0, 1])

    def k_rows(self, store=None):
        store = store or self.store
        return KProgress(store.state, self.identity, self.ranked, sha(store.state / "rfe_state.json"))

    def commit_k(self, store=None):
        store = store or self.store
        for row in self.k_rows(store).rows:
            name = f"k_{row['k']:03d}"
            if not store.saved(name):
                store.commit(name, {f"k/{name}.json": store.state / "k" / (name + ".json")})

    def search_one(self, *, progress=None, state=None, boundary=None, fit_calls=None):
        progress, state = progress or self.progress, state or self.store.state
        calls = fit_calls if fit_calls is not None else []
        predicted = []
        class FakeSVC:
            def predict(model, matrix):
                predicted.append(matrix.shape[1])
                return np.array([0, 1, 1, 1])  # 3/4 Validation accuracy, measured by actual metric function.
        def fit(model, matrix, labels):
            calls.append(matrix.shape[1])
            return model, {"seconds": 0.01, "converged": True, "warnings": []}
        def stop_after_boundary(**kwargs):
            if boundary is not None:
                boundary()
            stop[0] = True
        stop = [False]
        result = search(state, self.identity, self.ranked, self.X, self.y, self.X[:4], self.val_y, progress,
            model_factory=FakeSVC, fit_fn=fit, metrics_fn=core(SOURCE).metrics,
            boundary=stop_after_boundary, stop_requested=lambda: stop[0])
        return result, calls, predicted

    def completed_event(self, k, accuracy, best_k):
        self.progress.event("k_started", k=k, completed_k=k-1)
        self.progress.event("k_completed", k=k, completed_k=k, best_k=best_k,
                            validation_accuracy=accuracy, fit_seconds=0.1)

    def test_initial_best_is_none(self):
        row = self.progress.snapshot()
        self.assertIsNone(row["best_k_so_far"])
        self.assertIsNone(row["best_validation_accuracy"])
        self.assertEqual(row["last_completed_k"], 0)
        self.assertEqual(row["durable_last_completed_k"], 0)

    def test_real_k1_event_payload_no_keyerror_and_durable_only_after_commit(self):
        def boundary():
            before = self.progress.snapshot()
            self.assertEqual(before["last_completed_k"], 1)
            self.assertEqual(before["durable_last_completed_k"], 0)
            self.assertEqual(before["best_validation_accuracy"], 0.75)
            self.commit_k()
        _, fits, predicts = self.search_one(boundary=boundary)
        self.assertEqual(fits, [1]); self.assertEqual(predicts, [1])
        row = self.progress.snapshot()
        self.assertEqual(row["best_k_so_far"], 1)
        self.assertEqual(row["best_validation_accuracy"], self.k_rows().rows[0]["accuracy"])
        self.assertEqual(row["durable_last_completed_k"], 1)

    def test_best_accuracy_updates_from_measured_validation_and_ties_keep_smaller_k(self):
        self.completed_event(1, 0.5, 1)
        self.completed_event(2, 0.75, 2)
        self.completed_event(3, 0.75, 2)
        self.completed_event(4, 0.25, 2)
        self.assertEqual(self.progress.best_k, 2)
        self.assertEqual(self.progress.best_accuracy, 0.75)
        self.assertEqual(self.progress.snapshot()["durable_last_completed_k"], 0)

    def test_missing_required_accuracy_is_not_silently_hidden_or_completed(self):
        self.progress.event("k_started", k=1, completed_k=0)
        with self.assertRaises(KeyError):
            self.progress.event("k_completed", k=1, completed_k=1, best_k=1)
        self.assertEqual(self.progress.completed, 0)
        self.assertIsNone(self.progress.best_accuracy)

    def test_wrong_best_k_or_nonfinite_accuracy_is_blocked_before_state_update(self):
        self.progress.event("k_started", k=1, completed_k=0)
        for accuracy, winner in ((float("nan"), 1), (0.75, 2)):
            with self.subTest(accuracy=accuracy), self.assertRaises(Blocked):
                self.progress.event("k_completed", k=1, completed_k=1, best_k=winner, validation_accuracy=accuracy)
        self.assertEqual(self.progress.completed, 0)
        self.assertIsNone(self.progress.best_k)

    def test_failure_before_commit_leaves_local_k_but_failed_checkpoint_is_not_durable_k(self):
        def fail_before_commit():
            raise OSError("upload never committed")
        with self.assertRaisesRegex(OSError, "upload never committed"):
            self.search_one(boundary=fail_before_commit)
        self.store.commit("failed_before_commit", {"failure.json": self.failure_record("before")})
        self.assertEqual(len(self.k_rows().rows), 1)
        self.assertEqual(self.progress.snapshot()["last_completed_k"], 1)
        self.assertEqual(self.progress.snapshot()["durable_last_completed_k"], 0)
        report = inspect_run(self.store.remote)
        self.assertFalse(report["k1_committed_and_verified"])
        self.assertEqual(report["next_fit_k_from_drive"], 1)
        self.assertEqual(report["failed_checkpoint_count"], 1)

    def failure_record(self, reason):
        path = self.store.local / (reason + "_error.json")
        atomic(path, {"error": reason, "completed": False}, exclusive=True)
        return path

    def test_failure_after_commit_and_resume_skips_k1(self):
        def fail_after_commit():
            self.commit_k()
            raise OSError("after confirmed commit")
        with self.assertRaisesRegex(OSError, "after confirmed commit"):
            self.search_one(boundary=fail_after_commit)
        self.store.commit("failed_after_commit", {"failure.json": self.failure_record("after")})
        report = inspect_run(self.store.remote)
        self.assertTrue(report["k1_committed_and_verified"])
        self.assertEqual(report["next_fit_k_from_drive"], 2)
        self.assertEqual(self.progress.snapshot()["durable_last_completed_k"], 1)
        self.store.release("FAILED")
        restored = Store(self.root / "restored", self.store.remote, self.identity, resume=True)
        progress = Telemetry(restored, self.request); progress.check = Mock()
        progress.restore_history(self.k_rows(restored).rows, self.ranked)
        _, fits, _ = self.search_one(progress=progress, state=restored.state,
                                      boundary=lambda: self.commit_k(restored))
        self.assertEqual(fits, [2])
        self.assertEqual(progress.best_k, 1)  # Exact tie on the same Validation predictions.
        self.assertEqual(progress.snapshot()["durable_last_completed_k"], 2)

    def test_legacy_error_between_local_append_and_boundary_is_read_only_recoverable_local_only(self):
        event = self.progress.event
        def legacy_error(name, **details):
            if name == "k_completed":
                self.progress.completed = details["completed_k"]
                # Reproduce the old consumer against search()'s actual payload.
                self.progress.best_k, self.progress.best_accuracy = details["best_k"], details["best_validation_accuracy"]
                return
            event(name, **details)
        self.progress.event = legacy_error
        boundary = Mock(side_effect=lambda: self.commit_k())
        with self.assertRaisesRegex(KeyError, "best_validation_accuracy"):
            self.search_one(boundary=boundary)
        boundary.assert_not_called()
        self.assertEqual(self.progress.snapshot()["last_completed_k"], 1)
        self.assertEqual(self.progress.snapshot()["durable_last_completed_k"], 0)
        self.assertIsNone(self.progress.snapshot()["best_validation_accuracy"])
        self.store.commit("failed_legacy", {"failure.json": self.failure_record("legacy")})
        before = {p.relative_to(self.root).as_posix(): info(p) for p in self.root.rglob("*") if p.is_file()}
        with patch("pathlib.Path.mkdir", side_effect=AssertionError("Audit must never create directories")), \
             patch("joblib.load", side_effect=AssertionError("Audit must never load models")):
            report = inspect_run(self.store.remote, local=self.store.local)
        after = {p.relative_to(self.root).as_posix(): info(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(report["k1_committed_and_verified"])
        self.assertEqual(report["local"]["pending_commit_k"], [1])
        self.assertEqual(report["local"]["next_fit_k_if_identity_matches"], 2)
        self.assertEqual(report["remote_writes"], 0)

    def test_source_dataset_and_algorithm_mismatches_block_resume_without_writes(self):
        for key, value in (("source", {"files": {}, "sha256": digest({})}),
                           ("dataset_fingerprint", "other"), ("config", "FW512_FS16"), ("algorithm", "OTHER")):
            changed = {**self.identity, key: value}
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.subTest(key=key), self.assertRaisesRegex(Blocked, "identity mismatch"):
                inspect_run(self.store.remote, expected_identity=changed)

    def test_corrupt_k_checkpoint_blocks_instead_of_guessing_completed_work(self):
        self.search_one(boundary=lambda: self.commit_k())
        saved = self.store.saved("k_001")
        with (saved["path"] / "k/k_001.json").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(Blocked):
            inspect_run(self.store.remote)

    def test_selection_has_no_test_or_final_refit_entrypoint(self):
        from phuong_anA.dataset import load_selection
        with patch("phuong_anA.dataset.core", side_effect=AssertionError("Test source accessed")):
            with self.assertRaises(Blocked):
                load_selection({}, {}, self.root, {}, "test")
        with patch("phuong_anA.pipeline.review", side_effect=AssertionError("Final phase reached preflight")):
            for mode in ("finalize", "test", "final_refit"):
                with self.subTest(mode=mode), self.assertRaises(Blocked):
                    run({"mode": mode})
