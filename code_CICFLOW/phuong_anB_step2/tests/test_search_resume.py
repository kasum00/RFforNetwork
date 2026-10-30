import tempfile
import unittest
from pathlib import Path
from phuong_anB_step2.contract import Blocked, atomic, read, sha
from phuong_anB_step2.history import History, check_committed_candidates, validate_rows
from phuong_anB_step2.policy import COARSE, coarse_grid, winner
from phuong_anB_step2.store import Store
from phuong_anB_step2.telemetry import Pause
from .fixtures import small_identity
from .helpers import Progress, history_fixture, run_search, commit_history


class SearchResume(unittest.TestCase):
    def test_fine_accuracy_tie_selects_smaller_k_than_coarse_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            _, _, history = history_fixture(state)
            calls, predictions = [], []
            run_search(state, history, calls=calls, predictions=predictions, scores={50: 18, 45: 18, 47: 16})
            self.assertEqual(calls, list(COARSE)+[45, 46, 47, 48, 49, 51, 52, 53, 54, 55])
            self.assertEqual(winner(history.rows)["k"], 45)

    def test_original_coarse_then_radius_fine_exact_k_no_duplicate_svc(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            identity, names, history = history_fixture(state)
            calls, predictions = [], []
            run_search(state, history, calls=calls, predictions=predictions, scores={50: 18, 45: 19, 47: 20})
            expected = list(COARSE)+[45, 46, 47, 48, 49, 51, 52, 53, 54, 55]
            self.assertEqual(calls, expected)
            self.assertEqual(predictions, expected)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(len(calls), 16)
            self.assertEqual(winner(history.rows)["k"], 47)
            self.assertEqual(winner(history.rows)["accuracy"], 1.0)
            validate_rows(history.rows, names, identity["search_policy"], validation_count=20, complete=True)
            restored = History(state, identity, names, sha(state / "rfe_state.json"))
            run_search(state, restored, calls=calls, predictions=predictions)
            self.assertEqual(calls, expected)
            self.assertEqual(read(state / "fine_plan.json")["coarse_best_k"], 50)

    def test_svm_fits_only_train_rows_and_predicts_only_validation_rows(self):
        from unittest.mock import patch
        from .helpers import FakeSVC
        seen_fit, seen_predict = [], []
        original_fit, original_predict = FakeSVC.fit, FakeSVC.predict
        def fit(model, matrix, labels):
            seen_fit.append((len(matrix), len(labels), matrix.dtype, matrix.flags.c_contiguous))
            return original_fit(model, matrix, labels)
        def predict(model, matrix):
            seen_predict.append(len(matrix))
            return original_predict(model, matrix)
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            _, _, history = history_fixture(state)
            with patch.object(FakeSVC, "fit", fit), patch.object(FakeSVC, "predict", predict):
                run_search(state, history, calls=[], predictions=[])
            import numpy as np
            self.assertTrue(seen_fit)
            self.assertTrue(all(row == (8, 8, np.dtype("float64"), True) for row in seen_fit))
            self.assertEqual(seen_predict, [20]*len(seen_fit))

    def test_equal_accuracy_refines_only_smallest_coarse_k_and_final_tie_is_smaller_k(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            _, _, history = history_fixture(state)
            calls, predictions = [], []
            run_search(state, history, calls=calls, predictions=predictions)
            self.assertEqual(calls, list(COARSE)+[5, 6, 7, 8, 9, 11, 12, 13, 14, 15])
            self.assertEqual(read(state / "fine_plan.json")["coarse_best_k"], 10)
            self.assertEqual(winner(history.rows)["k"], 5)
            self.assertNotIn(1, calls)

    def test_best_100_clamps_radius_and_requires_only_11_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            _, _, history = history_fixture(state)
            calls, predictions = [], []
            run_search(state, history, calls=calls, predictions=predictions, scores={100: 20})
            self.assertEqual(calls, list(COARSE)+[95, 96, 97, 98, 99])
            self.assertEqual(winner(history.rows)["k"], 100)
            self.assertEqual(len(calls), 11)

    def make_store(self, root):
        identity = small_identity()
        store = Store(root / "local", root / "drive", identity, resume=False)
        atomic(store.state / "rfe_state.json", {"synthetic_rank": True}, exclusive=True)
        store.commit("rfe", {"rfe_state.json": store.state / "rfe_state.json"})
        names = identity["dataset"]["names"][:100]
        return store, History(store.state, identity, names, sha(store.state / "rfe_state.json"))

    def checkpoint(self, store):
        def save(stage, files, **metadata):
            return store.saved(stage) or store.commit(stage, files, metadata=metadata)
        return save

    def test_failure_before_drive_commit_recommits_local_then_skips_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, history = self.make_store(root)
            calls, predictions = [], []
            def fail_before_commit(current):
                if current.rows:
                    raise RuntimeError("synthetic upload failure before COMMITTED")
            with self.assertRaisesRegex(RuntimeError, "upload failure"):
                run_search(store.state, history, calls=calls, predictions=predictions, commit=fail_before_commit)
            self.assertEqual(calls, [10])
            self.assertEqual(check_committed_candidates(store, history), 0)
            atomic(store.state / "failure.json", {"completed": False}, exclusive=True)
            store.commit("failed_unit", {"failure.json": store.state / "failure.json"})
            self.assertEqual(check_committed_candidates(store, history), 0)
            store.release("FAILED")
            restored = Store(store.local, store.remote, store.identity, resume=True)
            loaded = History(restored.state, store.identity, history.ranked, sha(restored.state / "rfe_state.json"))
            run_search(restored.state, loaded, calls=calls, predictions=predictions,
                       commit=lambda current: commit_history(restored, current), checkpoint=self.checkpoint(restored))
            self.assertEqual(calls, list(COARSE)+[5, 6, 7, 8, 9, 11, 12, 13, 14, 15])
            self.assertEqual(check_committed_candidates(restored, loaded), len(loaded.rows))

    def test_failure_after_commit_restores_on_fresh_local_without_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, history = self.make_store(root)
            calls, predictions = [], []
            def fail_after_commit(current):
                commit_history(store, current)
                if current.rows:
                    raise RuntimeError("synthetic failure after COMMITTED")
            with self.assertRaisesRegex(RuntimeError, "after COMMITTED"):
                run_search(store.state, history, calls=calls, predictions=predictions, commit=fail_after_commit)
            self.assertEqual(check_committed_candidates(store, history), 1)
            store.release("FAILED")
            restored = Store(root / "fresh_local", store.remote, store.identity, resume=True)
            loaded = History(restored.state, store.identity, history.ranked, sha(restored.state / "rfe_state.json"))
            run_search(restored.state, loaded, calls=calls, predictions=predictions,
                       commit=lambda current: commit_history(restored, current), checkpoint=self.checkpoint(restored))
            self.assertEqual(calls.count(10), 1)
            self.assertEqual(len(calls), 16)

    def test_safe_stop_after_committed_candidate_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, history = self.make_store(Path(temporary))
            calls, predictions = [], []
            with self.assertRaises(Pause):
                run_search(store.state, history, calls=calls, predictions=predictions,
                    progress=Progress(stop_after=2), commit=lambda h: commit_history(store, h))
            self.assertEqual(calls, [10, 20])
            self.assertEqual(check_committed_candidates(store, history), 2)
            run_search(store.state, history, calls=calls, predictions=predictions,
                       commit=lambda h: commit_history(store, h), checkpoint=self.checkpoint(store))
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(calls[:6], list(COARSE))

    def test_resume_inside_fine_preserves_coarse_anchor_and_never_refits_tried_k(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, history = self.make_store(root)
            calls, predictions = [], []
            with self.assertRaises(Pause):
                run_search(store.state, history, calls=calls, predictions=predictions,
                    scores={50: 18, 45: 19, 47: 20}, progress=Progress(stop_after=8),
                    commit=lambda h: commit_history(store, h), checkpoint=self.checkpoint(store))
            self.assertEqual(calls, list(COARSE)+[45, 46])
            self.assertEqual(read(store.state / "fine_plan.json")["coarse_best_k"], 50)
            store.release("PAUSED")
            restored = Store(root / "new_local", store.remote, store.identity, resume=True)
            loaded = History(restored.state, store.identity, history.ranked, sha(restored.state / "rfe_state.json"))
            run_search(restored.state, loaded, calls=calls, predictions=predictions,
                scores={50: 18, 45: 19, 47: 20}, commit=lambda h: commit_history(restored, h),
                checkpoint=self.checkpoint(restored))
            self.assertEqual(calls[8], 47)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(winner(loaded.rows)["k"], 47)

    def test_fit_failure_does_not_append_or_claim_candidate(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            _, _, history = history_fixture(state)
            with patch("phuong_anB_step2.tests.helpers.FakeSVC.fit", side_effect=RuntimeError("interrupted fit")):
                with self.assertRaisesRegex(RuntimeError, "interrupted fit"):
                    run_search(state, history, calls=[], predictions=[])
            self.assertEqual(history.rows, [])
            self.assertEqual(list((state / "candidates").glob("*")), [])

    def test_local_candidate_corruption_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            identity, names, history = history_fixture(state)
            with self.assertRaises(Pause):
                run_search(state, history, calls=[], predictions=[], progress=Progress(stop_after=1))
            history.path(1).write_text("{}", encoding="utf-8")
            with self.assertRaises(Blocked):
                History(state, identity, names, sha(state / "rfe_state.json"))

    def test_full100_history_not_required_and_partial_grid_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            identity, names, history = history_fixture(state)
            with self.assertRaises(Pause):
                run_search(state, history, calls=[], predictions=[], progress=Progress(stop_after=1))
            with self.assertRaises(Blocked):
                validate_rows(history.rows, names, identity["search_policy"], complete=True)

    def test_missing_fine_candidate_cannot_freeze_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            identity, names, history = history_fixture(state)
            run_search(state, history, calls=[], predictions=[], scores={50: 18, 47: 20})
            with self.assertRaises(Blocked):
                validate_rows(history.rows[:-1], names, identity["search_policy"], complete=True)
