import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from phuong_anB_step2.contract import Blocked, atomic, digest, info, read
from phuong_anB_step2.controller import _execute
from phuong_anB_step2.history import check_committed_candidates
from phuong_anB_step2.notebook_helpers import Queue, monitor_snapshot
from phuong_anB_step2.pipeline import verify
from phuong_anB_step2.results import OUTPUTS, measurements, publish
from phuong_anB_step2.store import Store
from .fixtures import SOURCE, small_identity
from .helpers import complete_state, commit_history


class PublicationUI(unittest.TestCase):
    def test_final_and_test_modes_are_rejected_before_preflight(self):
        from phuong_anB_step2.pipeline import run
        with patch("phuong_anB_step2.pipeline.review", side_effect=AssertionError("unsafe mode reached preflight")):
            for mode in ("final", "finalize", "test", "evaluate"):
                with self.assertRaises(Blocked):
                    run({"mode": mode})

    def full_store(self, root, commit_all=True):
        identity = small_identity()
        store = Store(root / "local", root / "drive", identity, resume=False)
        _, history = complete_state(store.state, identity)
        for stage, names in (
            ("init", ("run_identity.json",)),
            ("mi", ("mi_state.json", "mi_top100.json", "mi_feature_scores.csv")),
            ("scaler", ("top100_scaler.joblib", "top100_scaler_state.json")),
            ("rfe", ("rfe_state.json", "rfe_feature_ranking.csv")),
            ("session_unit", ("sessions/session_unit.json",)),
        ):
            store.commit(stage, {name: store.state / name for name in names})
        if commit_all:
            for sequence, row in enumerate(history.rows, 1):
                if row["phase"] == "FINE" and store.saved("fine_plan") is None:
                    store.commit("fine_plan", {"fine_plan.json": store.state / "fine_plan.json"})
                path = history.path(sequence)
                store.commit(history.stage(sequence), {path.relative_to(store.state).as_posix(): path},
                             metadata={"phase": row["phase"], "k": row["k"]})
        store.commit("selection", {name: store.state / name for name in OUTPUTS})
        return store, history

    def test_publish_verify_full_chain_nonoverwrite_and_no_test_or_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, history = self.full_store(root)
            result = publish(store)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["best_k"], 47)
            self.assertEqual(result["rf_fits"], 51)
            self.assertEqual(result["svm_fits"], 16)
            saved_selection = read(store.remote / "results/selection.json")
            self.assertEqual(saved_selection["coarse"], [10, 20, 30, 50, 75, 100])
            self.assertEqual(saved_selection["coarse_best_k"], 50)
            self.assertEqual(saved_selection["fine_radius"], 5)
            before = info(store.remote / "results/selection.json")
            with patch("sklearn.svm.SVC.fit", side_effect=AssertionError("unexpected fit")), \
                 patch("pyarrow.parquet.read_table", side_effect=AssertionError("unexpected data read")):
                verified = verify(store.remote, store.identity, root / "verify")
            self.assertEqual(verified["status"], "PASS")
            self.assertFalse(verified["test_values_read"])
            self.assertEqual(verified["test_evaluation_calls"], 0)
            with self.assertRaises(Blocked):
                store.mutable("heartbeat.json", {"status": "rewritten"})
            self.assertEqual(info(store.remote / "results/selection.json"), before)

    def test_publication_rejects_local_candidates_without_individual_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, history = self.full_store(Path(temporary), commit_all=False)
            self.assertEqual(check_committed_candidates(store, history), 0)
            with self.assertRaisesRegex(Blocked, "COMMITTED"):
                publish(store)
            self.assertFalse((store.remote / "results/SELECTION_COMPLETE.json").exists())

    def test_publication_rejects_fine_plan_committed_after_fine_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, history = self.full_store(Path(temporary), commit_all=False)
            commit_history(store, history)
            store.commit("fine_plan", {"fine_plan.json": store.state / "fine_plan.json"})
            with self.assertRaisesRegex(Blocked, "before fine candidates"):
                publish(store)
            self.assertFalse((store.remote / "results/SELECTION_COMPLETE.json").exists())

    def test_published_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, _ = self.full_store(root)
            publish(store)
            (store.remote / "results/features.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(Blocked):
                verify(store.remote, store.identity, root / "verify")

    def test_post_selection_failure_receipt_does_not_change_frozen_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            identity, history = complete_state(state)
            original = measurements(state, identity, history)
            receipt = {"identity": identity["sha256"], "attempt": "later", "outcome": "FAILED",
                "wall_seconds": 123.0, "rf_started": 0, "rf_completed": 0, "svm_started": 0, "svm_completed": 0,
                "peak_rss_bytes": 1024, "prior_receipts": {}}
            receipt["sha256"] = digest(receipt)
            atomic(state / "sessions/session_later.json", receipt, exclusive=True)
            self.assertEqual(measurements(state, identity, history), original)

    def test_waiting_monitor_has_time_path_identity_and_no_false_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"source": str(SOURCE), "dataset": str(root / "CICFLOWMETER_FEATURE_CSV"),
                "output": str(root / "experiment_results"), "local_root": str(root / "local"),
                "config": "FW512_FS8", "run_id": "RUN_B2_01", "production": False}
            snapshot = monitor_snapshot(request)
            self.assertIsNone(snapshot["time"])
            self.assertIsNone(snapshot["path"])
            self.assertIsNone(snapshot["heartbeat"])
            self.assertIsNone(snapshot["identity"])

    def test_safe_stop_is_a_flag_in_b_local_namespace_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"source": str(SOURCE), "dataset": str(root / "CICFLOWMETER_FEATURE_CSV"),
                "output": str(root / "experiment_results"), "local_root": str(root / "local"),
                "config": "FW256_FS8", "run_id": "RUN_B2_01", "production": False, "mode": "run"}
            queue = Queue()
            queue.state.update(running=True, current_request=request)
            message = queue.safe_stop()
            from phuong_anB_step2.contract import paths
            local, remote = paths(request)
            self.assertTrue((local / "STOP_REQUESTED").is_file())
            self.assertTrue(queue.state["stop_queue"])
            self.assertFalse(remote.exists())
            self.assertIn("REQUESTED LOCALLY", message)

    def test_stop_after_config_does_not_create_worker_stop_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"source": str(SOURCE), "dataset": str(root / "CICFLOWMETER_FEATURE_CSV"),
                "output": str(root / "experiment_results"), "local_root": str(root / "local"),
                "config": "FW256_FS8", "run_id": "RUN_B2_01", "production": False, "mode": "run"}
            queue = Queue()
            queue.state.update(running=True, current_request=request)
            queue.stop_after_config()
            from phuong_anB_step2.contract import paths
            local, _ = paths(request)
            self.assertTrue(queue.state["stop_queue"])
            self.assertFalse((local / "STOP_REQUESTED").exists())

    def test_native_exit_and_reported_status_are_separate_and_log_appends(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, marker = Path(temporary), "__UNIT_RESULT__"
            aggregate = root / "aggregate.log"
            for native, status in ((0, "BLOCKED"), (7, "FAIL")):
                process = SimpleNamespace(stdout=iter(["stderr fixture combined\n", marker+json.dumps({"status": status})+"\n"]),
                    wait=lambda native=native: native, poll=lambda native=native: native)
                with patch("phuong_anB_step2.controller.subprocess.Popen", return_value=process):
                    result = _execute(["synthetic"], SOURCE, root / "logs", aggregate, {"task": "unit"}, marker=marker)
                self.assertEqual(result["native_exit_code"], native)
                self.assertEqual(result["status"], status)
            content = aggregate.read_text()
            self.assertIn('"native_exit_code": 7', content)
            self.assertIn('"reported_status": "BLOCKED"', content)
            self.assertEqual(content.count("stderr fixture combined"), 2)

    def test_aggregate_log_failure_warns_and_preserves_native_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, marker = Path(temporary), "__UNIT_RESULT__"
            aggregate = root / "not_a_file"
            aggregate.mkdir()
            process = SimpleNamespace(stdout=iter([marker+json.dumps({"status": "FAIL"})+"\n"]),
                                      wait=lambda: 7, poll=lambda: 7)
            with patch("phuong_anB_step2.controller.subprocess.Popen", return_value=process):
                result = _execute(["synthetic"], SOURCE, root / "logs", aggregate, {"task": "unit"}, marker=marker)
            self.assertEqual(result["native_exit_code"], 7)
            self.assertFalse(result["aggregate_log_fully_saved"])
