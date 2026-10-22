"""Supervisor/checkpoint tests using mocks only; no worker or estimator is run."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from colab import svm_official_supervisor as supervisor
from colab.checkpoint_manager import publish_completed_checkpoint


class SupervisorTests(unittest.TestCase):
    def test_resource_limits_have_distinct_stop_reasons(self):
        budget = {"max_elapsed_seconds": 100, "max_tree_rss_bytes": 1000,
                  "min_runtime_available_ram_bytes": 500,
                  "min_free_disk_bytes": 700}
        cases = (
            ((101, 10, 1000, 1000), "MAX_ELAPSED_SECONDS_EXCEEDED"),
            ((10, 1001, 1000, 1000), "MAX_PROCESS_TREE_RSS_EXCEEDED"),
            ((10, 10, 499, 1000), "HOST_AVAILABLE_RAM_BELOW_APPROVED_RESERVE"),
            ((10, 10, 1000, 699), "FREE_DISK_BELOW_APPROVED_THRESHOLD"),
            ((10, 10, 1000, 1000), None),
        )
        for values, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(supervisor.resource_stop_reason(*values, budget), expected)
        source = Path(supervisor.__file__).read_text(encoding="utf-8")
        self.assertIn("retry_performed\": False", source)
        self.assertIn("automatic_resume_performed\": False", source)

    def test_stop_targets_only_captured_worker_tree(self):
        root, child, unrelated = MagicMock(), MagicMock(), MagicMock()
        root.pid, child.pid, unrelated.pid = 100, 101, 999
        with patch.object(supervisor, "_process_tree", return_value=[root, child]), \
             patch.object(supervisor.psutil, "wait_procs", return_value=([], [])):
            supervisor._stop_only_worker_tree(100, grace_seconds=0)
        root.terminate.assert_called_once()
        child.terminate.assert_called_once()
        unrelated.terminate.assert_not_called()

    def test_incomplete_checkpoint_cannot_resume_or_re_evaluate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoints" / "svm01.incomplete"
            checkpoint.mkdir(parents=True)
            (checkpoint / "checkpoint.json").write_text(
                '{"stage":"test_evaluation_completed","resume_supported":false}',
                encoding="utf-8")
            (checkpoint / "result.json").write_text(
                '{"status":"INCOMPLETE_DO_NOT_USE"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                publish_completed_checkpoint(root, "svm01", {})

    def test_manual_review_flag_when_test_started_without_validation(self):
        source = Path(supervisor.__file__).read_text(encoding="utf-8")
        self.assertIn('test_started and not run_validated', source)
        self.assertIn('manual_review_required_before_any_future_action', source)

    def test_publish_recovery_source_never_calls_fit_or_test_evaluation(self):
        import inspect
        source = inspect.getsource(publish_completed_checkpoint)
        self.assertNotIn("model.fit", source)
        self.assertNotIn("evaluate_final", source)
        self.assertIn('state.get("stage") != "run_validated"', source)


if __name__ == "__main__":
    unittest.main()
