import tempfile
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import RFE
from phuong_anA.contract import (ALGORITHM, Blocked, atomic, core, digest, info, paths, read, scientific, sha)
from phuong_anA.dataset import inspect, load_selection
from phuong_anA.rfe import fit_step5, total_order, validate_result
from phuong_anA.store import Store, ledger
from phuong_anA.telemetry import Telemetry, Pause
from mi_rfe_svm_colab.runner import search
from mi_rfe_svm_colab.local_progress import choose, KProgress
from .fixtures import SOURCE, dataset_fixture, small_identity


class TiedForest(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        self.classes_ = np.unique(y)
        return self


class A1Tests(unittest.TestCase):
    def test_rfe_step5_target1_matches_sklearn_rank_groups_and_resume(self):
        X, y = np.ones((8, 100), np.float32), np.array([0, 1]*4)
        names, steps = [f"f{i}" for i in range(100)], []
        expected = RFE(TiedForest(), step=5, n_features_to_select=1).fit(X, y)
        result = fit_step5(X, y, names, TiedForest(), checkpoint=steps.append)
        np.testing.assert_array_equal(result["ranking"], expected.ranking_)
        self.assertEqual(len(steps), 20)
        self.assertEqual(result["forest_fits_in_completed_rfe"], 21)
        self.assertEqual(result["unique_sklearn_rank_count"], 21)
        self.assertEqual(len(set(result["names"])), 100)
        resumed = fit_step5(X, y, names, TiedForest(), steps=steps[:7])
        self.assertEqual(resumed["ranking"], result["ranking"])
        self.assertEqual(resumed["records"], result["records"])
        self.assertEqual(validate_result(resumed, names), result["names"])

    def test_importance_then_mi_position_tie_policy(self):
        names = ["a", "b", "c", "d"]
        ranks = [2, 2, 2, 1]
        steps = [{"step": 1, "active_indices": [0, 1, 2, 3], "removed_indices": [0, 1, 2],
                  "squared_importances": [0.1, 0.4, 0.4, 0.9]}]
        self.assertEqual([r["feature"] for r in total_order(names, ranks, steps)], ["d", "b", "c", "a"])

    def test_rfe_train_only_and_invalid_checkpoint_blocked(self):
        with self.assertRaises(Blocked):
            fit_step5(np.ones((4, 3)), np.array([0, 1, 0, 1]), ["a", "b", "c"], TiedForest(), fit_role="val")
        with self.assertRaises(Blocked):
            fit_step5(np.ones((4, 3)), np.array([0, 1, 0, 1]), ["a", "b", "c"], TiedForest(), steps=[{}])

    def test_unchanged_mi_svm_and_rfe_estimator_parameters(self):
        a0, a1 = core(SOURCE).contract("svm"), scientific(SOURCE)
        self.assertEqual(a1["mi"], a0["mi"])
        self.assertEqual(a1["classifier"], a0["classifier"])
        self.assertEqual(a1["rfe"]["parameters"], a0["rfe"]["parameters"])
        self.assertEqual(a1["rfe"]["parameters"]["max_depth"], 15)
        self.assertEqual(a1["rfe"]["step"], 5)
        self.assertEqual(a1["rfe"]["n_features_to_select"], 1)
        self.assertEqual(a1["k_grid"], list(range(1, 101)))
        self.assertEqual(a1["test_evaluation_calls"], 0)

    def test_k_search_all100_one_fit_predict_and_no_duplicate_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = small_identity()
            store = Store(Path(temporary) / "local", Path(temporary) / "drive", identity, resume=False)
            state = store.state
            atomic(state / "rfe_state.json", {"algorithm": ALGORITHM})
            store.commit("rfe", {"rfe_state.json": state / "rfe_state.json"})
            calls, predictions, saved = [], [], []
            class FakeSVC:
                def predict(self, X):
                    predictions.append(X.shape[1]); return np.array([0, 1]*4)
            def fit(model, X, y):
                calls.append(X.shape[1]); return model, {"seconds": 0.1, "converged": True}
            def boundary(**kw):
                history = KProgress(state, identity, names, sha(state / "rfe_state.json"))
                saved.append(history.rows[-1]["k"])
                k = history.rows[-1]["k"]
                store.commit(f"k_{k:03d}", {f"k/k_{k:03d}.json": state / "k" / f"k_{k:03d}.json"})
            names, X, y = [f"f{i}" for i in range(100)], np.ones((8, 100), np.float32), np.array([0, 1]*4)
            kwargs = dict(model_factory=FakeSVC, fit_fn=fit, metrics_fn=lambda a, b: {"accuracy": 1.0},
                          boundary=boundary)
            rows = search(state, identity, names, X, y, X, y, Mock(), **kwargs)
            self.assertEqual(calls, list(range(1, 101))); self.assertEqual(predictions, calls)
            self.assertEqual(saved, calls)
            self.assertEqual(choose(rows, names, complete=True)["k"], 1)
            store.release("PAUSED")
            restored = Store(Path(temporary) / "new_local", store.remote, identity, resume=True)
            self.assertEqual(len(restored.rows), 101)
            self.assertEqual(search(restored.state, identity, names, X, y, X, y, Mock(), **kwargs), rows)
            self.assertEqual(len(calls), 100)

    def test_session_budget_is_configurable_and_not_inferred_from_5h_path(self):
        store = Mock(); store.local = Path("/nonexistent_a1_fixture"); store.identity = small_identity()
        now = [10.0]
        request = {"runtime": {"session_budget_seconds": 36000, "stop_margin_seconds": 100, "heartbeat_seconds": 15}}
        telemetry = Telemetry(store, request, clock=lambda: now[0])
        now[0] += 18001
        self.assertFalse(telemetry.stop_requested())
        now[0] += 18000
        self.assertTrue(telemetry.stop_requested())


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.request, self.parent, self.proof, self.folder = dataset_fixture(Path(self.temp.name))

    def inspect(self):
        with patch("phuong_anA.dataset.evidence", return_value=(self.parent, self.proof)):
            return inspect(self.request)

    def test_actual_manifest_counts_and_test_never_opened_or_hashed(self):
        import pyarrow.parquet as pq
        test_path = self.folder / "test_windows.parquet"
        with patch("phuong_anA.dataset.sha", wraps=sha) as hashes, patch("pyarrow.parquet.ParquetFile", wraps=pq.ParquetFile) as footers:
            audit = self.inspect()
        self.assertEqual(audit["binding"]["splits"]["train"]["rows"], 8)
        self.assertEqual(audit["label_mapping"]["status"], "PROVISIONAL_FOLDER")
        self.assertFalse(audit["binding"]["splits"]["test"]["sha256_verified"])
        self.assertNotIn(test_path, [Path(c.args[0]) for c in hashes.call_args_list])
        self.assertNotIn(test_path, [Path(c.args[0]) for c in footers.call_args_list])

    def test_test_loader_rejected_before_any_io(self):
        with patch("phuong_anA.dataset.core") as module:
            with self.assertRaises(Blocked):
                load_selection({}, {}, Path(self.temp.name), {}, "test")
            module.assert_not_called()

    def test_train_validation_checksum_mismatch_blocks(self):
        with (self.folder / "train_windows.parquet").open("ab") as f: f.write(b"changed")
        with patch("pyarrow.parquet.ParquetFile", side_effect=AssertionError("Footer opened before checksum validation")), \
             self.assertRaisesRegex(Blocked, "SHA256 differs: train"):
            self.inspect()

    def test_validation_checksum_checked_before_any_footer(self):
        with (self.folder / "val_windows.parquet").open("ab") as f: f.write(b"changed")
        with patch("pyarrow.parquet.ParquetFile", side_effect=AssertionError("Footer opened before checksum validation")), \
             self.assertRaisesRegex(Blocked, "SHA256 differs: val"):
            self.inspect()

    def test_missing_manifest_and_wrong_split_or_label_provenance_blocked(self):
        with patch("phuong_anA.dataset.evidence", side_effect=Blocked("Missing frozen manifest")):
            with self.assertRaisesRegex(Blocked, "Missing frozen manifest"):
                inspect(self.request)
        self.parent["v2_preflight"]["checks"]["source_row_identity_unique"] = "FAIL"
        with self.assertRaisesRegex(Blocked, "isolation check failed"):
            self.inspect()
        self.parent["v2_preflight"]["checks"]["source_row_identity_unique"] = "PASS"
        self.parent["v2_preflight"]["label_policy"] = "UNKNOWN"
        with self.assertRaisesRegex(Blocked, "proxy labels"):
            self.inspect()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.identity = small_identity()
        self.local, self.remote = self.root / "local", self.root / "drive/run"
        self.store = Store(self.local, self.remote, self.identity, resume=False)
        atomic(self.store.state / "fixture.json", {"fixture": True})

    def test_commit_restore_and_incomplete_upload_ignored(self):
        row = self.store.commit("mi", {"fixture.json": self.store.state / "fixture.json"})
        self.store.release("PAUSED")
        incomplete = self.remote / "checkpoints/.incomplete_bad"
        incomplete.mkdir(); (incomplete / "COMMITTED.json").write_text('{}')
        restored = Store(self.root / "newlocal", self.remote, self.identity, resume=True)
        self.assertEqual(read(restored.state / "fixture.json"), {"fixture": True})
        self.assertEqual(restored.rows[0]["sha256"], row["sha256"])
        self.assertEqual(len(ledger(self.remote, self.identity)), 1)

    def test_upload_failure_preserves_previous_checkpoint(self):
        before = self.store.commit("mi", {"fixture.json": self.store.state / "fixture.json"})
        with patch("phuong_anA.store.shutil.copyfileobj", side_effect=OSError("Drive disconnected")):
            with self.assertRaises(OSError):
                self.store.commit("scaler", {"fixture.json": self.store.state / "fixture.json"})
        self.assertEqual(ledger(self.remote, self.identity)[0]["sha256"], before["sha256"])

    def test_dataset_source_config_params_and_other_algorithm_mismatch(self):
        for key, value in (("dataset_fingerprint", "changed"), ("source", {"sha256": "changed"}),
                           ("config", "FW512_FS16"), ("contract", {"rfe": {"step": 1}}), ("algorithm", "A0_OR_B")):
            with self.subTest(key=key), self.assertRaises(Blocked):
                Store(self.root / key, self.remote, small_identity(**{key: value}), resume=True)

    def test_duplicate_stage_run_and_foreign_output_never_overwritten(self):
        self.store.commit("mi", {"fixture.json": self.store.state / "fixture.json"})
        with self.assertRaises(Blocked): self.store.commit("mi", {"fixture.json": self.store.state / "fixture.json"})
        with self.assertRaises(Blocked): Store(self.root / "other", self.remote, self.identity, resume=False)
        foreign = self.root / "old"; foreign.mkdir(); (foreign / "result.json").write_text("preserve")
        with self.assertRaises(Blocked): Store(self.root / "another", foreign, self.identity, resume=True)
        self.assertEqual((foreign / "result.json").read_text(), "preserve")

    def test_corrupt_committed_file_blocked(self):
        row = self.store.commit("mi", {"fixture.json": self.store.state / "fixture.json"})
        (row["path"] / "fixture.json").write_text('{}')
        with self.assertRaises(Blocked): ledger(self.remote, self.identity)

    def test_selection_publication_verify_and_completed_protection(self):
        from phuong_anA.results import create_selection, publish, OUTPUTS
        from phuong_anA.pipeline import verify
        from .fixtures import csv_file
        from mi_rfe_svm_colab.runner import freeze
        state, identity = self.store.state, dict(self.identity)
        names = [f"feature_{i}__mean" for i in range(380)]
        identity["dataset"] = {"names": names, "splits": {"train": {"rows": 8}, "val": {"rows": 4}}}
        # Use a fresh owned run for the complete synthetic dependency chain.
        identity = {k: v for k, v in identity.items() if k != "sha256"}
        identity["sha256"] = digest(identity)
        store = Store(self.root / "selection_local", self.root / "selection_drive", identity, resume=False)
        state, top = store.state, names[:100]
        atomic(state / "mi_top100.json", top)
        with (state / "mi_feature_scores.csv").open("x") as f:
            f.write(",0\n" + "".join(f"{name},{380-i}\n" for i, name in enumerate(names)))
        atomic(state / "mi_state.json", {"identity": identity["sha256"],
            "files": {n: info(state / n) for n in ("mi_top100.json", "mi_feature_scores.csv")}})
        (state / "top100_scaler.joblib").write_bytes(b"synthetic scaler; never deserialize")
        atomic(state / "top100_scaler_state.json", {"binding": {"identity": identity["sha256"],
            "mi": sha(state / "mi_state.json"), "top": top}, "fit_role": "train", "rows": 8,
            "files": {"top100_scaler.joblib": info(state / "top100_scaler.joblib")}})
        result = fit_step5(np.ones((8, 100), np.float32), np.array([0, 1]*4), top, TiedForest())
        csv_file(state / "rfe_feature_ranking.csv", result["records"])
        # csv_file uses BOM for source manifests; ranking production uses UTF-8.
        text = (state / "rfe_feature_ranking.csv").read_text(encoding="utf-8-sig")
        (state / "rfe_feature_ranking.csv").write_text(text, encoding="utf-8")
        atomic(state / "rfe_state.json", {"binding": {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"),
            "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}, "result": result,
            "files": {"rfe_feature_ranking.csv": info(state / "rfe_feature_ranking.csv")}})
        ranked = result["names"]
        history = KProgress(state, identity, ranked, sha(state / "rfe_state.json"))
        metrics = core(SOURCE).metrics(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        for k in range(1, 101):
            history.append({"k": k, "features": ranked[:k], **metrics,
                            "fit": {"seconds": 0.0, "converged": True}})
        frozen = freeze(state, identity, history.rows, ranked, core(SOURCE))
        create_selection(state, identity, history.rows, ranked, frozen)
        files = {p.relative_to(state).as_posix(): p for p in state.rglob("*") if p.is_file()}
        store.commit("selection", files)
        with patch("joblib.load", side_effect=AssertionError("Selection verification must not load models")):
            self.assertEqual(publish(store)["best_k"], 1)
            self.assertEqual(verify(store.remote, identity, self.root)["test_evaluation_calls"], 0)
            before = {n: info(store.remote / "results" / n) for n in (*OUTPUTS, "artifact_manifest.json", "SELECTION_COMPLETE.json")}
            publish(store)
            self.assertEqual(before, {n: info(store.remote / "results" / n) for n in before})
        with self.assertRaises(Blocked):
            store.commit("after_completion", {"mi_top100.json": state / "mi_top100.json"})


class SourceEvidenceTests(unittest.TestCase):
    def source_fixture(self, root):
        from phuong_anA.contract import RUNTIME_SOURCE_FILES
        extra = ("stage4_mi/mi_selection.py", "stage5_rfe/rfe_selection.py", "colab/test_cicflow_reference_pipeline.py")
        for name in (*RUNTIME_SOURCE_FILES, *extra):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if path.suffix == '.json' else '# Synthetic source placeholder\n', encoding="utf-8")
        return Mock(SOURCE_FILES=extra)

    def test_runtime_snapshot_binds_frozen_evidence_without_legacy_generators(self):
        from phuong_anA.contract import source_snapshot
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = self.source_fixture(root)
            with patch("phuong_anA.contract.core", return_value=module):
                before = source_snapshot(root)
                self.assertIn("mi_rfe_svm_colab/runner.py", before["files"])
                self.assertIn("phuong_anA/evidence/cicflow_split_evidence.json", before["files"])
                self.assertNotIn("stage29_cplus_integrity.py", before["files"])
                (root / "stage29_cplus_integrity.py").write_text("# Historical generator, not executed by A1\n")
                self.assertEqual(source_snapshot(root), before)
                (root / "phuong_anA/evidence/cicflow_split_evidence.json").write_text('{"changed":true}')
                self.assertNotEqual(source_snapshot(root)["sha256"], before["sha256"])

    def test_missing_runtime_source_is_still_blocked(self):
        from phuong_anA.contract import source_snapshot
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = self.source_fixture(root)
            (root / "mi_rfe_svm_colab/common.py").unlink()  # Only this synthetic temporary fixture.
            with patch("phuong_anA.contract.core", return_value=module), \
                 self.assertRaisesRegex(Blocked, "Missing required A1 runtime source: mi_rfe_svm_colab/common.py"):
                source_snapshot(root)

    def test_evidence_collection_errors_never_become_pass(self):
        from phuong_anA import selftest
        successful = Mock(testsRun=1, skipped=[], failures=[], errors=[])
        successful.wasSuccessful.return_value = True
        for failure in ("source", "environment"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary) / "evidence.json"
                with patch.object(selftest.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                     patch.object(selftest.unittest, "TextTestRunner") as runner, \
                     patch.object(selftest, "source_snapshot") as snapshot, \
                     patch.object(selftest, "environment") as env, \
                     patch.object(selftest.sys, "stderr", new_callable=io.StringIO) as diagnostics:
                    runner.return_value.run.return_value = successful
                    snapshot.return_value = {"sha256": "a"*64}
                    env.return_value = {"versions": {}}
                    if failure == "source":
                        snapshot.side_effect = Blocked("Missing required runtime source")
                    else:
                        env.side_effect = RuntimeError("Environment unavailable")
                    self.assertEqual(selftest.main(["--source", temporary, "--evidence", str(evidence)]), 1)
                    self.assertIn("EVIDENCE FAIL:", diagnostics.getvalue())
                record = read(evidence)
                self.assertEqual(record["status"], "FAIL")
                self.assertEqual(record["tests_run"], 1)
                self.assertEqual(len(record["diagnostic_errors"]), 1)
                self.assertIsNone(record["source_sha256"] if failure == "source" else record["environment"])
