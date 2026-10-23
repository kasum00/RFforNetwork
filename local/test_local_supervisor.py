"""Windows orchestration mocks and tiny logs; NOT EXECUTED by Codex."""
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from local import local_authorization as auth
from local import local_supervisor as supervisor
from local.progress_console import LocalDeadline, LocalProgress
from colab.svm_resumable_store import atomic_json, read_json, file_sha
from colab import svm_resumable_training as engine
from local.windows_job import WindowsJob


class WindowsSupervisorTests(unittest.TestCase):
    def test_recycled_worker_pid_is_never_terminated(self):
        with patch.object(supervisor, "same_process", return_value=False), \
             patch.object(supervisor.psutil, "Process") as process, \
             patch.object(supervisor.psutil, "wait_procs", return_value=([], [])):
            result = supervisor.terminate_owned(123, 1.0)
        self.assertEqual(result["captured_processes"], 0)
        process.assert_not_called()

    def test_only_matching_creation_time_descendants_are_terminated(self):
        owned, recycled = MagicMock(pid=222), MagicMock(pid=333)
        owned.create_time.return_value = 2.0
        recycled.create_time.return_value = 99.0
        with patch.object(supervisor.psutil, "wait_procs", return_value=([], [])):
            supervisor.terminate_owned(-1, -1, {(222, 2.0): owned, (333, 3.0): recycled})
        owned.terminate.assert_called_once()
        recycled.terminate.assert_not_called()

    def test_cleanup_access_denied_is_reported_as_unconfirmed_not_raised(self):
        process = MagicMock(pid=222)
        process.create_time.return_value = 2.0
        process.terminate.side_effect = supervisor.psutil.AccessDenied(222)
        with patch.object(supervisor.psutil, "wait_procs", return_value=([], [])):
            result = supervisor.terminate_owned(-1, -1, {(222, 2.0): process})
        self.assertIn(222, result["survivors"])
        self.assertTrue(result["cleanup_errors"])

    def test_job_handoff_uses_atomic_publish_not_partial_visible_json(self):
        source = inspect.getsource(supervisor.supervise)
        self.assertIn("os.rename(temporary_ready, ready_path)", source)
        self.assertLess(source.index("job.assign(process)"), source.index("os.rename(temporary_ready, ready_path)"))

    def test_windows_held_lock_acquires_and_releases_byte(self):
        fake = MagicMock(LK_NBLCK=1, LK_UNLCK=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict("sys.modules", {"msvcrt": fake}), patch.object(supervisor.os, "name", "nt"):
                with supervisor.windows_run_lock(root):
                    pass
        self.assertEqual([c.args[1] for c in fake.locking.call_args_list], [1, 2])

    def test_windows_job_binds_retained_process_handle_and_closes(self):
        kernel = MagicMock()
        kernel.CreateJobObjectW.return_value = 789
        with patch("local.windows_job.os.name", "nt"):
            with WindowsJob(kernel=kernel) as job:
                job.assign(SimpleNamespace(_handle=456))
        kernel.AssignProcessToJobObject.assert_called_once_with(789, 456)
        kernel.CloseHandle.assert_called_once_with(789)

    def test_failed_job_assignment_blocks_instead_of_unprotected_fit(self):
        kernel = MagicMock()
        kernel.CreateJobObjectW.return_value = 789
        kernel.AssignProcessToJobObject.return_value = False
        with patch("local.windows_job.os.name", "nt"):
            with self.assertRaises(OSError):
                with WindowsJob(kernel=kernel) as job:
                    job.assign(SimpleNamespace(_handle=456))
        kernel.CloseHandle.assert_called_once_with(789)

    def test_recent_or_changed_prior_state_blocks_takeover(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context = SimpleNamespace(root=root, approval={"prior_state": {"different": True}})
            with patch.object(supervisor, "_session_claim_guard") as guard, \
                 patch.object(supervisor, "prior_state", return_value={}), \
                 patch.object(supervisor, "_claim_session_locked") as claim:
                guard.return_value.__enter__.return_value = None
                with self.assertRaises(PermissionError):
                    supervisor.claim_local(context)
            claim.assert_not_called()

    def test_still_live_local_supervisor_cannot_be_taken_over(self):
        import socket
        previous = {"host": socket.gethostname(), "supervisor_pid": 123, "supervisor_create_time": 1}
        state = {"active_session": previous}
        context = SimpleNamespace(root=Path("synthetic"), approval={"prior_state": state})
        with patch.object(supervisor, "_session_claim_guard"), \
             patch.object(supervisor, "prior_state", return_value=state), \
             patch.object(supervisor, "same_process", return_value=True), \
             patch.object(supervisor, "_claim_session_locked") as claim:
            with self.assertRaises(PermissionError):
                supervisor.claim_local(context)
        claim.assert_not_called()

    def test_orphan_worker_blocks_takeover_even_if_supervisor_dead(self):
        import socket
        previous = {"host": socket.gethostname(), "supervisor_pid": 123, "supervisor_create_time": 1,
                    "worker_pid": 456, "worker_create_time": 2}
        state = {"active_session": previous}
        context = SimpleNamespace(root=Path("synthetic"), approval={"prior_state": state})
        with patch.object(supervisor, "_session_claim_guard"), \
             patch.object(supervisor, "prior_state", return_value=state), \
             patch.object(supervisor, "same_process", side_effect=[False, True]), \
             patch.object(supervisor, "_claim_session_locked") as claim:
            with self.assertRaisesRegex(PermissionError, "worker is still alive"):
                supervisor.claim_local(context)
        claim.assert_not_called()

    def test_guard_stops_owned_worker_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context = SimpleNamespace(paths=SimpleNamespace(output_root=root), root=root,
                deadline_epoch=10**12, base=SimpleNamespace(budgets={"max_elapsed_seconds": 36000,
                "poll_interval_seconds": 10}))
            proc = MagicMock(pid=123, returncode=7)
            proc.poll.side_effect = [None, 7]
            parent = MagicMock(pid=123)
            parent.create_time.return_value = 1.0
            parent.children.return_value = []
            parent.memory_info.return_value = SimpleNamespace(rss=100)
            with patch.object(supervisor, "same_process", return_value=True), \
                 patch.object(supervisor.psutil, "Process", return_value=parent), \
                 patch.object(supervisor.psutil, "virtual_memory", return_value=SimpleNamespace(available=100)), \
                 patch.object(supervisor.psutil, "disk_usage", return_value=SimpleNamespace(free=100)), \
                 patch.object(supervisor, "resource_stop_reason", return_value="MAX_PROCESS_TREE_RSS_EXCEEDED"), \
                 patch.object(supervisor, "terminate_owned", return_value={"survivors": []}) as stop:
                report = supervisor.monitor(proc, 1.0, context, root, sleeper=lambda _: self.fail("must not retry"))
            stop.assert_called_once_with(123, 1.0, unittest.mock.ANY)
            self.assertEqual(report["stop_reason"], "MAX_PROCESS_TREE_RSS_EXCEEDED")

    def test_console_worker_is_unbuffered_inherited_not_captured(self):
        source = inspect.getsource(supervisor.supervise)
        self.assertIn('"-u"', source)
        self.assertIn('PYTHONUNBUFFERED="1"', source)
        self.assertIn("subprocess.Popen(command, cwd=paths.repo, env=env)", source)
        self.assertNotIn("capture_output=True", source)

    def test_no_automatic_training_loop_or_parameter_replacement(self):
        source = inspect.getsource(supervisor.supervise)
        self.assertIn("automatic_retry=False", source)
        self.assertIn("automatic_new_session=False", source)
        self.assertNotIn("LinearSVC", source)
        self.assertEqual(source.count("subprocess.Popen("), 1)


class LocalAuthorizationAndProgressTests(unittest.TestCase):
    def test_failed_preflight_never_creates_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "review.json"
            atomic_json(report, {"status": "RESOURCE_ADMISSION_FAIL"}, exclusive=True)
            with self.assertRaises(ValueError):
                auth.verify_review(None, None, report)

    def test_legacy_approval_is_rejected_not_resealed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.json"
            atomic_json(path, {"decision": "APPROVED_STAGE73B", "effective": True}, exclusive=True)
            args = SimpleNamespace(preflight_report=path, approval=path)
            with patch.object(auth, "verify_review", return_value={}):
                with self.assertRaisesRegex(ValueError, "new effective Windows"):
                    auth.authorize_local(None, args)

    def test_changed_approval_file_invalidates_sealed_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "approval.json"
            path.write_text("{}", encoding="utf-8")
            paths = SimpleNamespace(repo=root, data_root=root, output_root=root)
            base = SimpleNamespace(approval_path=str(path), approval_sha256="not-the-hash")
            context = auth.LocalContext(base, {}, {}, root, paths, {}, True, 300, auth._SEAL, 10**12)
            with self.assertRaisesRegex(PermissionError, "Approval changed"):
                auth.validate_local_context(context, root, root, root)

    def test_worker_authorization_rehashes_train_validation_before_any_fit(self):
        source = inspect.getsource(supervisor.worker)
        self.assertIn("verify_content_now=True", source)
        self.assertLess(source.index("verify_content_now=True"), source.index("run_windows_resumable("))
        self.assertLess(source.index('read_json(ready)'), source.index("verify_content_now=True"))

    def test_test_content_verification_is_gated_on_claim_and_final(self):
        source = inspect.getsource(LocalProgress.__call__)
        self.assertLess(source.index('read_json(root / "results" / "test_evaluation_claim.json")'),
                        source.index('verify_content(paths, read_json(evidence), roles=("test",))'))
        self.assertIn('store.reference("final")', source)

    def test_stop_request_blocks_next_fit_but_not_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp) / "STOP_REQUESTED.json"
            deadline = LocalDeadline(36000, 300, clock=lambda: 0, stop_path=stop)
            self.assertTrue(deadline.may_start())
            stop.write_text("{}", encoding="utf-8")
            self.assertFalse(deadline.may_start())

    def test_local_events_and_heartbeat_persist_console_throttled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("local.progress_console.time.monotonic", return_value=1000), \
                 patch("builtins.print") as output:
                with LocalProgress(root, "session02", interval=3600, console_interval=300,
                                   session_dir=root) as progress:
                    progress("k_fit_started", k=76, completed_k=75, checkpoint="k_search/k_075")
                    before = output.call_count
                    progress("heartbeat")
                    self.assertEqual(output.call_count, before)
            current = read_json(root / "progress.json")
            self.assertEqual(current["current_k"], 76)
            self.assertEqual(current["completed_k"], 75)
            self.assertEqual(current["stage"], "k_fit_started")
            self.assertTrue((root / "heartbeat.jsonl").is_file())
            self.assertTrue((root / "events.jsonl").is_file())

    def test_local_engine_still_uses_canonical_scientific_functions(self):
        source = inspect.getsource(engine._execute_resumable)
        for function in ("training.select_mi(", "training.rank_rfe(", "training.search_k(",
                         "training.fit_final_classifier(", "training.evaluate_final("):
            self.assertIn(function, source)
        self.assertIn("k_values=(k,)", source)
        self.assertLess(source.index('checkpoints.save("final",'), source.index("checkpoints.claim_test_once("))
        self.assertLess(source.index("checkpoints.claim_test_once("), source.index('prepared_split(training, store, config, "test"'))
        self.assertEqual(source.count("training.evaluate_final(fitted, test)"), 1)


if __name__ == "__main__":
    unittest.main()
