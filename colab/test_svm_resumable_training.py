"""Mocked orchestration and tiny preprocessing checks; no real SVM fit."""
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from colab import svm_resumable_training as runner
from colab.svm_resumable_store import CheckpointStore
from colab.experiment_launcher import _training


class ScientificTests(unittest.TestCase):
    def test_completed_final_model_resumes_at_test_without_refitting(self):
        from colab.svm_resumable_store import digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            top = [f"f{i}" for i in range(100)]
            binding = {"rows_by_split": {"train": 4, "val": 2, "test": 2},
                       "scientific_contract": {"experiment": "mi_rfe_svm"}}
            checkpoints = CheckpointStore(root, binding, resume=True)
            checkpoints.save("data_prepared", {"label_counts": {
                "train": {"benign": 2, "attack": 2}, "val": {"benign": 1, "attack": 1}}})
            checkpoints.save("mi", {"features": top, "indices": list(range(100)),
                                    "measurement": {"seconds": 0}},
                             parents={"data": checkpoints.reference("data_prepared")})
            mi_ref = checkpoints.reference("mi")
            checkpoints.save("rfe", {"features": top, "complete_ranking": True,
                                     "forest_fits": 100, "measurement": {"seconds": 0}},
                             parents={"mi": mi_ref})
            k_parents = {"mi": mi_ref, "rfe": checkpoints.reference("rfe")}
            rows = []
            for k in range(1, 101):
                row = {"k": k, "features": top[:k], "metrics": {"accuracy": .5},
                       "fit": {"seconds": 0, "converged": True}}
                rows.append(row)
                checkpoints.save(f"k_search/k_{k:03d}", row, parents=k_parents)
            checkpoints.save("selection", {"best_k": 1, "features": top[:1],
                                           "best_validation_accuracy": .5},
                             parents={**k_parents, "k_history": digest(rows)})
            final_meta = {"features": top[:1], "fit": {"seconds": 0, "converged": True},
                          "fit_rows": 6, "fit_role": "train+val", "mi_rfe_refit": False}
            checkpoints.save("final", final_meta,
                             parents={"selection": checkpoints.reference("selection")}, writers={
                "final_model.joblib": lambda p: p.write_bytes(b"synthetic-model"),
                "final_scaler.joblib": lambda p: p.write_bytes(b"synthetic-scaler")})
            context = SimpleNamespace(root=root, binding=binding, resume=True,
                base=SimpleNamespace(identity={"config": "FW_512_FS_16", "run_name": "run"}),
                policy={"session_id": "s02", "session_budget_seconds": 36000,
                        "stop_starting_new_work_margin_seconds": 300, "heartbeat_interval_seconds": 3600})
            fake_store = SimpleNamespace(snapshot=lambda: {}, files={}, root=Path(temp))
            training = MagicMock(COMPACT_SVM_PARAMS={"kernel": "rbf"})
            training.WindowStore.return_value = fake_store
            training.fit_final_classifier.side_effect = AssertionError("final fit must be reused")
            model = SimpleNamespace(get_params=lambda: {"kernel": "rbf"}, n_features_in_=1)
            scaler = SimpleNamespace(n_features_in_=1)
            test = SimpleNamespace(y=np.array([0, 1], dtype=np.int8))
            training.evaluate_final.return_value = {
                **final_meta, "model": model, "scaler": scaler,
                "metrics": {"accuracy": 1.0}, "test_evaluation_calls": 1,
                "test_rows_evaluated": 2, "predict_seconds": 0}
            with patch.object(runner, "validate_authorization", return_value=context), \
                 patch.object(runner, "_training", return_value=training), \
                 patch.object(runner, "manifest_root_for", return_value=Path(temp)), \
                 patch.object(runner, "prepared_split", return_value=test) as prepare, \
                 patch.object(runner.joblib, "load", side_effect=[model, scaler]):
                first = runner.run_resumable(Path(temp), Path(temp), Path(temp), context)
                second = runner.run_resumable(Path(temp), Path(temp), Path(temp), context)
            self.assertEqual(first["status"], "COMPLETED_VERIFIED")
            self.assertEqual(second["test_evaluation_calls_this_session"], 0)
            training.fit_final_classifier.assert_not_called()
            training.evaluate_final.assert_called_once()
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[3], "test")

    def test_resume_12_completed_k_starts_at_13_without_repeating_mi_rfe(self):
        with tempfile.TemporaryDirectory() as temp:
            root, local = Path(temp) / "run", Path(temp) / "local"
            names = [f"f{i}" for i in range(380)]
            top = names[:100]
            binding = {"rows_by_split": {"train": 4, "val": 2, "test": 2}}
            store = CheckpointStore(root, binding, resume=True)
            store.save("data_prepared", {"train_rows": 4, "val_rows": 2})
            store.save("mi", {"features": top, "indices": list(range(100)),
                              "measurement": {"seconds": 0}},
                       parents={"data": store.reference("data_prepared")}, writers={
                           "mi.joblib": lambda p: p.write_bytes(b"mocked"),
                           "top100_scaler.joblib": lambda p: p.write_bytes(b"mocked")})
            mi_ref = store.reference("mi")
            store.save("rfe", {"features": top, "complete_ranking": True,
                               "forest_fits": 100, "measurement": {"seconds": 0}},
                       parents={"mi": mi_ref})
            parents = {"mi": mi_ref, "rfe": store.reference("rfe")}
            for k in range(1, 13):
                store.save(f"k_search/k_{k:03d}", {
                    "k": k, "features": top[:k], "metrics": {"accuracy": .5}}, parents=parents)
            context = SimpleNamespace(root=root, binding=binding, resume=True,
                base=SimpleNamespace(identity={"config": "FW_512_FS_16", "run_name": "run"}),
                policy={"session_id": "s02", "session_budget_seconds": 36000,
                        "stop_starting_new_work_margin_seconds": 300, "heartbeat_interval_seconds": 3600})
            fake_store = SimpleNamespace(snapshot=lambda: {}, files={}, root=Path(temp))
            train = SimpleNamespace(y=np.array([0, 1, 0, 1], dtype=np.int8))
            val = SimpleNamespace(y=np.array([0, 1], dtype=np.int8))
            training = MagicMock()
            training.FEATURE_NAMES = tuple(names)
            training.WindowStore.return_value = fake_store
            training.select_mi.side_effect = AssertionError("MI must be reused")
            training.rank_rfe.side_effect = AssertionError("RFE must be reused")
            training.search_k.return_value = {"results": [{
                "k": 13, "accuracy": .6, "fit": {"seconds": 0, "converged": True},
                "predict_seconds": 0}]}
            mi = {"names": top, "indices": list(range(100))}
            deadline = SimpleNamespace(may_start=MagicMock(side_effect=[True, True, False]))
            with patch.object(runner, "validate_authorization", return_value=context), \
                 patch.object(runner, "_training", return_value=training), \
                 patch.object(runner, "manifest_root_for", return_value=Path(temp)), \
                 patch.object(runner, "SessionDeadline", return_value=deadline), \
                 patch.object(runner, "prepared_split", side_effect=[train, val]), \
                 patch.object(runner, "_scaled_top100", return_value=(object(), object())), \
                 patch.object(runner.joblib, "load", side_effect=[mi, SimpleNamespace(n_features_in_=100)]):
                result = runner.run_resumable(Path(temp), Path(temp), local, context)
            self.assertEqual(result["status"], "PAUSED_AT_SAFE_CHECKPOINT")
            self.assertEqual(result["next_stage"], "k=14")
            training.search_k.assert_called_once()
            self.assertEqual(training.search_k.call_args.kwargs["k_values"], (13,))
            training.select_mi.assert_not_called()
            training.rank_rfe.assert_not_called()
            self.assertEqual(len(store.completed_k(parents, top)), 13)

    def test_deadline_stops_starting_new_fits_at_margin(self):
        now = [0]
        deadline = runner.SessionDeadline(36000, 300, clock=lambda: now[0])
        self.assertTrue(deadline.may_start())
        now[0] = 35700
        self.assertFalse(deadline.may_start())

    def test_worker_deadline_accounts_for_time_spent_in_authorization(self):
        now = [100.0]
        deadline = runner.SessionDeadline(36000, 300, clock=lambda: now[0],
                                         deadline_epoch=10350, wall_clock=lambda: 10000)
        self.assertTrue(deadline.may_start())
        now[0] = 150.0
        self.assertFalse(deadline.may_start())

    def test_empty_artifact_manifest_never_reports_publication_success(self):
        from colab.svm_resumable_store import atomic_json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            output = root / "completed"
            output.mkdir()
            atomic_json(output / "artifact_manifest.json", {
                "binding_sha256": "same", "files": {}}, exclusive=True)
            with self.assertRaises(ValueError):
                runner.verify_publication(output, {"result.json": source}, "same")

    def test_all_100_required_and_tie_keeps_smaller_k(self):
        rows = [{"k": k, "features": list(range(k)), "metrics": {"accuracy": .8}}
                for k in range(1, 101)]
        self.assertEqual(runner.choose_best(rows)["k"], 1)
        rows[11]["metrics"]["accuracy"] = .9
        rows[12]["metrics"]["accuracy"] = .9
        self.assertEqual(runner.choose_best(rows)["k"], 12)
        with self.assertRaises(ValueError):
            runner.choose_best(rows[:-1])

    def test_hyperparameters_and_four_configs_are_preserved(self):
        training = _training(Path(__file__).resolve().parents[1])
        self.assertEqual(set(training.CONFIGS), {
            "FW_256_FS_8", "FW_256_FS_16", "FW_512_FS_8", "FW_512_FS_16"})
        params = training.make_model("svm").get_params()
        self.assertEqual((params["C"], params["kernel"], params["gamma"]), (10., "rbf", "scale"))
        self.assertEqual(params["max_iter"], 20000)
        self.assertEqual(params["cache_size"], 10000)
        self.assertFalse(params["probability"])
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(training.SVM_RFE_PARAMS["max_depth"], 15)
        self.assertEqual(training.SVM_RFE_PARAMS["n_estimators"], 100)

    def test_scaler_reload_transform_matches_canonical_top100(self):
        # Only small StandardScaler operations; never fits SVM/RF/MI.
        import joblib
        training = _training(Path(__file__).resolve().parents[1])
        X = np.arange(8 * 380, dtype=np.float32).reshape(8, 380)
        train = training.PreparedSplit("train", X[:6], np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
        val = training.PreparedSplit("val", X[6:], np.array([0, 1], dtype=np.int8))
        mi = {"indices": list(reversed(range(100)))}
        Xt, Xv, scaler = training.scale_top100(train, val, mi)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scaler.joblib"
            joblib.dump(scaler, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(Xt, loaded.transform(train.X[:, mi["indices"]]))
            np.testing.assert_array_equal(Xv, loaded.transform(val.X[:, mi["indices"]]))

    def test_first_test_read_is_after_verified_final_and_claim(self):
        source = inspect.getsource(runner._execute_resumable)
        claim = source.index("checkpoints.claim_test_once(")
        test = source.index('prepared_split(training, store, config, "test"')
        evaluate = source.index("training.evaluate_final(fitted, test)")
        self.assertLess(claim, test)
        self.assertLess(test, evaluate)
        self.assertLess(source.index("verified_selection(checkpoints)"), claim)
        self.assertEqual(source.count("training.evaluate_final(fitted, test)"), 1)
        self.assertLess(source.index('checkpoints.save("final",'), claim)
        self.assertIn('k_values=(k,)', source)
        self.assertIn('"svm", target=1', source)

    def test_single_k_uses_canonical_metrics_without_real_fit(self):
        training = _training(Path(__file__).resolve().parents[1])
        class Model:
            def predict(self, X):
                return np.array([0, 1, 0, 1], dtype=np.int8)
        names = [f"f{i}" for i in range(100)]
        X = np.zeros((4, 100), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        with patch.object(training, "make_model", return_value=Model()), \
             patch.object(training, "timed_fit", return_value=(Model(), {"seconds": 0, "converged": True})):
            row = training.search_k(X, y, X, y, names, names, "svm", k_values=(13,))["results"][0]
        self.assertEqual(row["k"], 13)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(row["macro_f1"], 1.0)

    def test_durable_progress_persists_stage_k_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with runner.DurableProgress(root, "s01", interval=3600) as progress:
                progress("k_fit_started", k=13, completed_k=12, checkpoint="k_search/k_012")
                progress("heartbeat")
            from colab.svm_resumable_store import read_json
            value = read_json(root / "progress.json")
            self.assertEqual(value["current_k"], 13)
            self.assertEqual(value["completed_k"], 12)
            self.assertEqual(value["stage"], "k_fit_started")
            self.assertTrue((root / "heartbeat.jsonl").is_file())

    def test_existing_test_claim_blocks_before_any_loader_or_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            binding = {"identity": "synthetic"}
            checkpoints = CheckpointStore(root, binding, resume=True)
            (root / "results").mkdir()
            (root / "results" / "test_evaluation_claim.json").write_text("{}")
            context = SimpleNamespace(root=root, binding=binding, resume=True,
                base=SimpleNamespace(identity={"config": "FW_512_FS_16"}),
                policy={"session_id": "s01", "session_budget_seconds": 36000,
                        "stop_starting_new_work_margin_seconds": 300,
                        "heartbeat_interval_seconds": 3600})
            training = SimpleNamespace(WindowStore=lambda *a, **kw: self.fail("must not load dataset"))
            with patch.object(runner, "validate_authorization", return_value=context), \
                 patch.object(runner, "_training", return_value=training):
                with self.assertRaises(PermissionError):
                    runner.run_resumable(Path(temp), Path(temp), Path(temp), context)

    def test_publication_recovery_does_not_fit_or_read_test(self):
        source = inspect.getsource(runner.publish_results)
        self.assertNotIn("evaluate_final", source)
        self.assertNotIn("load_full", source)
        self.assertNotIn("fit_final_classifier", source)


if __name__ == "__main__":
    unittest.main()
