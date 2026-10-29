import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from mi_rfe_svm_colab.common import (Blocked, FORMAT, SVM_PARAMS, atomic, contract, core,
                                    digest, info, read, sha)
from mi_rfe_svm_colab.local_progress import KProgress
from mi_rfe_svm_colab.runner import freeze, scaled_top, test_once
from mi_rfe_svm_colab.artifact_reuse import validate_selection
from mi_rfe_svm_colab.preflight import resource_gate
from mi_rfe_svm_colab.verification import validate_state

SOURCE = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def test_mi_is_one_train_matrix_call_with_reference_scaler_and_seed(self):
        from sklearn.preprocessing import StandardScaler
        module = core(SOURCE)
        train = np.random.default_rng(42).normal(size=(12, 380)).astype(np.float32)
        y = np.array([0, 1]*6)
        names = [f"f{i}" for i in range(380)]
        with patch("sklearn.feature_selection.mutual_info_classif", return_value=np.zeros(380)) as mi:
            scores = module.reference_mi(train, y, names)
        self.assertEqual(mi.call_count, 1)
        np.testing.assert_array_equal(mi.call_args.args[0], StandardScaler().fit_transform(train))
        self.assertIs(mi.call_args.args[1], y)
        self.assertEqual(mi.call_args.kwargs, {"random_state": 42})
        self.assertEqual(scores.index.tolist(), pd.Series(np.zeros(380), index=names).sort_values(ascending=False).index.tolist())

    def test_rfe_ties_and_step1_match_installed_sklearn(self):
        module = core(SOURCE)
        from stage5_rfe.rfe_selection import manual_rfe_step1
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        # Tiny forest fixture; no SVC. Constant features deliberately tie.
        X = np.ones((12, 5), dtype=np.float32)
        y = np.array([0, 1]*6)
        estimator = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
        expected = RFE(estimator, step=1, n_features_to_select=1).fit(X, y)
        states = []
        actual, support, steps = manual_rfe_step1(X, y, estimator, checkpoint_callback=states.append)
        np.testing.assert_array_equal(actual, expected.ranking_)
        np.testing.assert_array_equal(support, expected.support_)
        resumed, _, _ = manual_rfe_step1(X, y, estimator, initial_support=states[1]["support"],
            initial_ranking=states[1]["ranking"], completed_steps=2)
        np.testing.assert_array_equal(resumed, expected.ranking_)
        self.assertEqual(steps, 4)

    def test_resource_gate_blocks_without_scientific_parameter_reduction(self):
        budget = {"min_available_ram_bytes": 1024**3, "max_process_rss_bytes": 40*1024**3}
        result = resource_gate({"train": 100, "val": 30}, budget, sample={
            "available_ram_bytes": 1, "disk_free_bytes": 100*1024**3, "logical_cpus": 2, "rss_bytes": 1})
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["estimate"]["cache_size_mib"], 10000)
        self.assertEqual(SVM_PARAMS["max_iter"], 20000)


class FinalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state, self.remote = self.root / "state", self.root / "remote"
        self.state.mkdir()
        self.remote.mkdir()
        self.test_request = {"source": str(SOURCE), "mode": "finalize", "phase": "test"}
        self.names = [f"f{i}" for i in range(380)]
        self.top = self.names[:100]
        self.identity = {"format": FORMAT, "spec": {"experiment": "mi_rfe_svm", "config": "FW256_FS8"},
            "contract": contract(SOURCE),
            "dataset": {"names": self.names, "splits": {"train": {"rows": 4}, "val": {"rows": 2}, "test": {"rows": 2}}},
            "dataset_audit": {"fixture": True}}
        self.identity["sha256"] = digest(self.identity)
        atomic(self.state / "run_identity.json", self.identity)
        atomic(self.state / "mi_top100.json", self.top)
        (self.state / "mi_feature_scores.csv").write_text(
            ",0\n" + "".join(f"{n},{380-i}\n" for i, n in enumerate(self.names)), encoding="utf-8")
        mi = {"identity": self.identity["sha256"], "files": {n: info(self.state / n) for n in ("mi_top100.json", "mi_feature_scores.csv")}}
        atomic(self.state / "mi_state.json", mi)
        (self.state / "rfe_feature_ranking.csv").write_text(
            ",0\n" + "".join(f"{n},{i+1}\n" for i, n in enumerate(self.top)), encoding="utf-8")
        rfe = {"binding": {"identity": self.identity["sha256"], "mi_state": digest(mi), "top_order": digest(self.top)},
               "names": self.top, "ranking": list(range(1, 101)), "completed_eliminations": 99,
               "final_estimator_fit_performed": True, "files": {"rfe_feature_ranking.csv": info(self.state / "rfe_feature_ranking.csv")}}
        atomic(self.state / "rfe_state.json", rfe)
        (self.state / "top100_scaler.joblib").write_bytes(b"fixture scaler; not deserialized")
        atomic(self.state / "top100_scaler_state.json", {
            "binding": {"identity": self.identity["sha256"], "mi": sha(self.state / "mi_state.json"), "top": self.top},
            "files": {"top100_scaler.joblib": info(self.state / "top100_scaler.joblib")}, "fit_role": "train", "rows": 4})
        history = KProgress(self.state, self.identity, self.top, sha(self.state / "rfe_state.json"))
        for k in range(1, 101):
            history.append({"k": k, "features": self.top[:k], "accuracy": 1.0,
                            "fit": {"seconds": 1, "converged": True}})
        self.frozen = freeze(self.state, self.identity, history.rows, self.top, core(SOURCE))
        for name in ("model.joblib", "final_scaler.joblib"):
            (self.state / name).write_bytes(b"fixture, mocked loader only")
        atomic(self.state / "final_state.json", {"binding": {"identity": self.identity["sha256"], "frozen": digest(self.frozen)},
            "fit_role": "train+val", "fit_rows": 6, "fit": {"converged": True},
            "files": {n: info(self.state / n) for n in ("model.joblib", "final_scaler.joblib")}})

    def test_mi_order_binding_and_rfe_mapping(self):
        self.assertEqual(validate_selection(self.state, self.identity), (self.top, self.top))
        state = read(self.state / "rfe_state.json")
        state["binding"]["top_order"] = "wrong"
        atomic(self.state / "rfe_state.json", state)
        with self.assertRaises(Blocked):
            validate_selection(self.state, self.identity)

    def test_frozen_selection_and_final_chain(self):
        self.assertEqual(validate_state(self.state, self.identity, final_required=True)["best_k"], 1)
        final = read(self.state / "final_state.json")
        final["fit_rows"] = 5
        atomic(self.state / "final_state.json", final)
        with self.assertRaises(Blocked):
            validate_state(self.state, self.identity, final_required=True)

    def invoke_test(self, loader):
        model, scaler, checkpoint = Mock(), Mock(), Mock()
        model.get_params.return_value = SVM_PARAMS.copy()
        model.n_features_in_ = scaler.n_features_in_ = 1
        model.predict.return_value = np.array([0, 1])
        scaler.transform.side_effect = lambda X: X

        def load_after_claim(request, audit, local, identity, role, *, test_claim):
            # Mock only the dataset I/O boundary. Claim creation/readback,
            # final-state validation and core(source) all run unchanged.
            self.assertEqual(request, self.test_request)
            self.assertEqual(audit, self.identity["dataset_audit"])
            self.assertEqual(local, self.root)
            self.assertEqual(identity, self.identity)
            self.assertEqual(role, "test")
            self.assertEqual(read(self.remote / "TEST_CLAIM.json"), test_claim)
            self.assertEqual(read(self.state / "TEST_CLAIM.json"), test_claim)
            self.assertEqual(test_claim["identity"], self.identity["sha256"])
            self.assertEqual(test_claim["final_state"], digest(read(self.state / "final_state.json")))
            self.assertEqual(test_claim["evaluation_number"], 1)
            self.assertIs(test_claim["retry_forbidden"], True)
            checkpoint.assert_called_once_with(force=True, reason="final_refit_before_test")
            return loader(request, audit, local, identity, role, test_claim=test_claim)

        with patch("joblib.load", side_effect=[model, scaler]) as model_load, \
             patch("mi_rfe_svm_colab.runner.inspect", return_value=self.identity["dataset_audit"]) as inspect_data, \
             patch("mi_rfe_svm_colab.runner.raw_split", side_effect=load_after_claim) as raw:
            self.invocation = {"model": model, "scaler": scaler, "model_load": model_load,
                               "inspect": inspect_data, "raw": raw, "checkpoint": checkpoint}
            result = test_once(self.test_request, self.state, self.root, self.remote,
                               self.identity, Mock(), checkpoint)
        return result, model, raw

    def test_test_claim_precedes_load_and_second_call_does_not_predict(self):
        def load(*args, **kw):
            self.assertTrue((self.remote / "TEST_CLAIM.json").exists())
            return np.zeros((2, 380), np.float32), np.array([0, 1])
        result, model, raw = self.invoke_test(load)
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(model.predict.call_count, 1)
        self.assertEqual(result["test_evaluation_calls"], 1)
        claims_before = {str(p): info(p) for p in
                         (self.remote / "TEST_CLAIM.json", self.state / "TEST_CLAIM.json")}
        second, second_model, second_raw = self.invoke_test(lambda *a, **kw: self.fail("Test loaded twice"))
        self.assertEqual(result, second)
        second_model.predict.assert_not_called()
        second_raw.assert_not_called()
        self.invocation["model_load"].assert_not_called()
        self.invocation["inspect"].assert_not_called()
        self.invocation["checkpoint"].assert_not_called()
        self.assertEqual(claims_before, {p: info(p) for p in claims_before})

    def test_interrupted_test_claim_blocks_retry(self):
        with self.assertRaisesRegex(OSError, "lost runtime"):
            self.invoke_test(lambda *a, **kw: (_ for _ in ()).throw(OSError("lost runtime")))
        self.invocation["raw"].assert_called_once()
        self.invocation["model"].predict.assert_not_called()
        self.assertFalse((self.state / "test_result.json").exists())
        claims_before = {str(p): info(p) for p in
                         (self.remote / "TEST_CLAIM.json", self.state / "TEST_CLAIM.json")}
        with self.assertRaisesRegex(Blocked, "Test was already claimed without a verified result"):
            self.invoke_test(lambda *a, **kw: self.fail("Test retry forbidden"))
        self.invocation["model_load"].assert_not_called()
        self.invocation["inspect"].assert_not_called()
        self.invocation["raw"].assert_not_called()
        self.invocation["model"].predict.assert_not_called()
        self.invocation["checkpoint"].assert_not_called()
        self.assertFalse((self.state / "test_result.json").exists())
        self.assertEqual(claims_before, {p: info(p) for p in claims_before})

    def test_saved_scaler_and_transforms_reused_without_refit(self):
        # Independent tiny cache fixture, real scaler only; no SVC.
        root = self.root / "scaler_fixture"
        root.mkdir()
        atomic(root / "mi_state.json", {"fixture": True})
        identity = {"sha256": "scaler-fixture", "dataset": {"names": self.names}}
        train = np.arange(1520, dtype=np.float32).reshape(4, 380)
        val = train[:2].copy()
        first = scaled_top({"source": str(SOURCE)}, root, self.root / "local", identity,
                           train, np.array([0, 1, 0, 1]), val, np.array([0, 1]), self.top)
        with patch("sklearn.preprocessing.StandardScaler", side_effect=AssertionError("Unexpected scaler refit")):
            second = scaled_top({"source": str(SOURCE)}, root, self.root / "local", identity,
                                train, np.array([0, 1, 0, 1]), val, np.array([0, 1]), self.top)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_reference_svm_artifact_import_without_mi_or_rfe_fit(self):
        from sklearn.ensemble import RandomForestClassifier
        from mi_rfe_svm_colab.artifact_reuse import FILES, inspect_import, import_selection
        module = core(SOURCE)
        old_dir, target = self.root / "legacy", self.root / "imported"
        old_dir.mkdir(); target.mkdir()
        env = {"python": "fixture", "versions": {"numpy": "fixture"}}
        old = {"format": module.FORMAT, "spec": self.identity["spec"], "contract": module.contract("svm"),
               "dataset": self.identity["dataset"], "environment": env,
               "source": module.source_snapshot(SOURCE), "git_commit": "a"*40,
               "resolved_estimators": {"classifier": module.make_model("svm").get_params(),
                   "rfe": RandomForestClassifier(**module.RFE_PARAMS["svm"]).get_params()}}
        old["sha256"] = digest(old)
        for name in FILES:
            shutil.copyfile(self.state / name, old_dir / name)
        atomic(old_dir / "run_identity.json", old)
        mi = read(old_dir / "mi_state.json"); mi["identity"] = old["sha256"]
        atomic(old_dir / "mi_state.json", mi)
        rfe = read(old_dir / "rfe_state.json")
        rfe["binding"] = {"identity": old["sha256"], "mi_state": digest(mi), "top_order": digest(self.top)}
        atomic(old_dir / "rfe_state.json", rfe)
        binding = inspect_import({"source": str(SOURCE), "config": "FW256_FS8", "reuse_from": str(old_dir)},
                                 {"binding": self.identity["dataset"]}, env)
        before = {name: info(old_dir / name) for name in FILES}
        with patch.object(module, "reference_mi", side_effect=AssertionError("MI refit")), \
             patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("RFE refit")):
            import_selection(target, self.identity, binding)
        self.assertEqual(validate_selection(target, self.identity), (self.top, self.top))
        self.assertEqual(before, {name: info(old_dir / name) for name in FILES})
        old["spec"] = {**old["spec"], "experiment": "mi_rfe_rf"}
        old["sha256"] = digest({k: v for k, v in old.items() if k != "sha256"})
        atomic(old_dir / "run_identity.json", old)
        with self.assertRaises(Blocked):
            inspect_import({"source": str(SOURCE), "config": "FW256_FS8", "reuse_from": str(old_dir)},
                           {"binding": self.identity["dataset"]}, env)

    def test_publication_verify_and_nonoverwrite_without_model_deserialization(self):
        from mi_rfe_svm_colab.drive_checkpoint import DriveCheckpoint
        from mi_rfe_svm_colab.verification import REQUIRED, publish, verify, result_record
        local = self.root / "sync"
        local.mkdir()
        manager = DriveCheckpoint(local, self.remote, self.identity)
        manager.sync(self.state, reason="final", safe_boundary=True, force=True)
        test_result, model, raw = self.invoke_test(
            lambda *a, **kw: (np.zeros((2, 380), np.float32), np.array([0, 1])))
        self.assertEqual(test_result["test_evaluation_calls"], 1)
        raw.assert_called_once()
        model.predict.assert_called_once()
        atomic(self.state / "result.json", result_record(self.state, self.identity))
        atomic(self.state / "hyperparameters.json", self.identity["contract"])
        atomic(self.state / "preflight_audit.json", {"status": "PASS", "identity": self.identity})
        for name in ("resource_usage.json", "progress.json"):
            atomic(self.state / name, {"fixture": True})
        (self.state / "events.jsonl").write_text('{"event":"fixture"}\n')
        (self.state / "console_snapshot.log").write_text("fixture\n")
        manager.sync(self.state, reason="complete", safe_boundary=True, force=True)
        with patch("joblib.load", side_effect=AssertionError("Verify must not deserialize models")):
            self.assertEqual(publish(self.state, self.remote, self.identity)["status"], "PASS")
            before = {name: info(self.remote / name) for name in
                      (*REQUIRED, "artifact_manifest.json", "RUN_COMPLETE.json")}
            self.assertEqual(verify(self.remote)["test_evaluation_calls"], 1)
            self.assertEqual(publish(self.state, self.remote, self.identity)["status"], "PASS")
            self.assertEqual(before, {name: info(self.remote / name) for name in before})
            self.assertEqual(model.predict.call_count, 1)
        with (self.remote / "model.joblib").open("ab") as f:
            f.write(b"corruption")
        with self.assertRaises(Blocked):
            verify(self.remote)
        corrupt_artifact = info(self.remote / "model.joblib")
        with self.assertRaisesRegex(Blocked, "Never overwrite a different existing artifact: model.joblib"):
            publish(self.state, self.remote, self.identity)
        self.assertEqual(info(self.remote / "model.joblib"), corrupt_artifact)
