"""Authorization/supervisor mocks. No subprocess, estimator fit or dataset audit."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from colab import svm_resumable_authorization as auth
from colab import svm_resumable_execution as execution
from colab.experiment_launcher import RunSpec


class AuthorizationTests(unittest.TestCase):
    def test_explicit_new_session_policy_can_pass_without_opening_legacy_gate(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "approval.json"
            modules = {"engine.py": "verified"}
            identity = {name: "same" for name in (
                "manifest_sha256", "validation_sha256", "window_summary_sha256",
                "dataset_fingerprint_sha256", "dataset_fingerprint_scope_sha256")}
            identity.update(experiment="mi_rfe_svm", config="FW_512_FS_16",
                            run_name="newrun", git_commit="commit")
            policy = {
                "schema": auth.POLICY_SCHEMA, "allow_completed_stage_resume": True,
                "repeat_only_uncommitted_work": True, "never_repeat_claimed_test": True,
                "session_id": "s01", "session_budget_seconds": 36000,
                "stop_starting_new_work_margin_seconds": 300, "heartbeat_interval_seconds": 60,
                "resumable_module_hashes": modules,
            }
            path.write_text(json.dumps({"execution_mode": auth.POLICY_SCHEMA,
                                        "identity": identity, "resume_policy": policy}))
            base = SimpleNamespace(identity=identity, budgets={"max_elapsed_seconds": 36000})
            training = SimpleNamespace(FEATURE_NAMES=tuple(f"f{i}" for i in range(380)),
                SEED=42, official_hyperparameters=lambda kind: {"experiment": kind})
            evidence = {"modules": list(auth.REQUIRED_TESTS), "git_commit": "commit",
                        "resumable_module_hashes": modules, "returncode": 0}
            with patch.object(auth, "extension_hashes", return_value=modules), \
                 patch.object(auth, "authorize_run", return_value=base) as legacy, \
                 patch.object(auth, "_training", return_value=training):
                context = auth.authorize_resume(root, root, root, root,
                    RunSpec("mi_rfe_svm", "FW_512_FS_16", "newrun"), path, root / "probe",
                    "phrase", unit_test_evidence=evidence, session_id="s01", resume=True,
                    session_budget_seconds=36000)
            legacy.assert_called_once()
            self.assertIsNone(context.deadline_epoch)
            with patch.object(auth, "validate_context_for_training", return_value=base):
                with self.assertRaises(PermissionError):
                    auth.validate_authorization(context)
                supervised = replace(context, deadline_epoch=100.0)
                self.assertIs(auth.validate_authorization(supervised), supervised)

    def test_missing_or_legacy_approval_never_opens_resume_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "approval.json"
            arguments = (Path(temp), Path(temp), Path(temp), Path(temp),
                         RunSpec("mi_rfe_svm", "FW_512_FS_16", "run01"), path,
                         Path(temp) / "probe.json", "phrase")
            with self.assertRaises(FileNotFoundError):
                auth.authorize_resume(*arguments, unit_test_evidence={}, session_id="s01",
                                      resume=True, session_budget_seconds=36000)
            path.write_text(json.dumps({"decision": "APPROVED_STAGE73B", "effective": True}))
            with self.assertRaises(PermissionError), patch.object(auth, "authorize_run") as legacy:
                auth.authorize_resume(*arguments, unit_test_evidence={}, session_id="s01",
                                      resume=True, session_budget_seconds=36000)
            legacy.assert_not_called()

    def test_baseline_and_rf_are_rejected(self):
        for experiment in ("baseline_svm", "mi_rfe_rf"):
            with self.subTest(experiment=experiment), self.assertRaises(PermissionError):
                auth.authorize_resume(Path("repo"), Path("windows"), Path("runs"), Path("drive"),
                    RunSpec(experiment, "FW_512_FS_16", "run01"), Path("approval"), Path("probe"),
                    "phrase", unit_test_evidence={}, session_id="s01", resume=True,
                    session_budget_seconds=36000)

    def test_sealed_context_is_required(self):
        with self.assertRaises(PermissionError):
            auth.validate_authorization(object())

    def test_wrong_session_budget_and_module_hash_are_rejected_before_legacy_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "approval.json"
            policy = {
                "schema": auth.POLICY_SCHEMA, "allow_completed_stage_resume": True,
                "repeat_only_uncommitted_work": True, "never_repeat_claimed_test": True,
                "session_id": "s01", "session_budget_seconds": 36000,
                "stop_starting_new_work_margin_seconds": 300, "heartbeat_interval_seconds": 60,
                "resumable_module_hashes": {"module": "old"},
            }
            path.write_text(json.dumps({"execution_mode": auth.POLICY_SCHEMA, "resume_policy": policy}))
            kwargs = dict(unit_test_evidence={}, session_id="s01", resume=True,
                          session_budget_seconds=36000)
            args = (Path(temp), Path(temp), Path(temp), Path(temp),
                    RunSpec("mi_rfe_svm", "FW_512_FS_16", "run01"), path, Path("probe"), "phrase")
            with patch.object(auth, "authorize_run") as legacy:
                with self.assertRaises(PermissionError):
                    auth.authorize_resume(*args, **{**kwargs, "session_id": "s02"})
                with self.assertRaises(PermissionError):
                    auth.authorize_resume(*args, **{**kwargs, "session_budget_seconds": 10})
                with patch.object(auth, "extension_hashes", return_value={"module": "new"}):
                    with self.assertRaises(PermissionError):
                        auth.authorize_resume(*args, **kwargs)
            legacy.assert_not_called()


class SupervisorTests(unittest.TestCase):
    def test_each_resource_stop_kills_only_its_worker_without_retry(self):
        budget = {"max_elapsed_seconds": 10, "max_tree_rss_bytes": 100,
                  "min_runtime_available_ram_bytes": 50, "min_free_disk_bytes": 50,
                  "poll_interval_seconds": 1}
        for expected in ("MAX_ELAPSED_SECONDS_EXCEEDED", "MAX_PROCESS_TREE_RSS_EXCEEDED",
                         "HOST_AVAILABLE_RAM_BELOW_APPROVED_RESERVE", "FREE_DISK_BELOW_APPROVED_THRESHOLD"):
            with self.subTest(reason=expected), tempfile.TemporaryDirectory() as temp:
                proc = MagicMock(pid=123)
                proc.poll.side_effect = [None, -15]
                root = MagicMock(pid=123)
                root.children.return_value = []
                root.create_time.return_value = 1
                root.memory_info.return_value = SimpleNamespace(rss=10)
                with patch.object(execution.psutil, "Process", return_value=root), \
                     patch.object(execution.psutil, "virtual_memory", return_value=SimpleNamespace(available=100)), \
                     patch.object(execution.psutil, "disk_usage", return_value=SimpleNamespace(free=100)), \
                     patch.object(execution, "resource_stop_reason", return_value=expected), \
                     patch.object(execution, "_stop_only_worker_tree") as stop:
                    report = execution.monitor_worker(proc, budget, Path(temp), Path(temp),
                        clock=lambda: 0, sleeper=lambda seconds: self.fail("should stop immediately"))
                self.assertEqual(report["stop_reason"], expected)
                stop.assert_called_once_with(123)

    def test_monitor_exception_stops_worker_and_does_not_continue_to_test(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = MagicMock(pid=123)
            proc.poll.side_effect = [None, None, -15]
            with patch.object(execution.psutil, "Process", side_effect=RuntimeError("monitor failed")), \
                 patch.object(execution, "_stop_only_worker_tree") as stop:
                report = execution.monitor_worker(proc, {}, Path(temp), Path(temp), clock=lambda: 0)
            self.assertEqual(report["stop_reason"], "SUPERVISOR_EXCEPTION_OR_INTERRUPT")
            stop.assert_called_once_with(123)

    def test_worker_failure_cannot_be_reported_as_complete(self):
        proc = MagicMock(pid=123)
        proc.poll.return_value = 7
        with tempfile.TemporaryDirectory() as temp:
            report = execution.monitor_worker(proc, {}, Path(temp), Path(temp), clock=lambda: 0)
        self.assertEqual(report["worker_returncode"], 7)

    def test_old_session_lock_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            approval = root / "approval.json"
            approval.write_text("{}")
            (root / "active_session.json").write_text('{"session_id":"s01"}')
            context = SimpleNamespace(policy={"session_id": "s02"},
                base=SimpleNamespace(approval_path=str(approval)))
            with self.assertRaises(PermissionError):
                execution.claim_session(root, context, os.getpid())


if __name__ == "__main__":
    unittest.main()
