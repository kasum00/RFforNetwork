"""Synthetic V2 regression tests. No real dataset or old run is opened."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colab.colab_pipeline_v2 import (
    CONFIGS, ContractError, REQUIRED, RunSpec, activate_source, completed_results,
    discover_test_files, initialize_run, require_launch_context,
    execute_task, resume_plan, run_test_files, source_snapshot, tests_status,
    validate_checkpoint_state, write_json_once,
)


def fake_layout(root):
    source = root / "code_CICFLOW"
    dataset = root / "CICFLOWMETER_FEATURE_CSV"
    dataset.mkdir()
    for relative in REQUIRED:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 'fixture'\n", encoding="utf-8")
    return source, dataset


class ColabV2RegressionTests(unittest.TestCase):
    def test_reference_run_spec_names_the_real_rfe_branch(self):
        self.assertEqual(RunSpec("mi_rfe_svm", "FW256_FS8", "fixture_run").branch, "svm")
        with self.assertRaises(ContractError):
            RunSpec("baseline_svm", "FW256_FS8", "fixture_run").validate()

    def test_changed_selection_after_preflight_cannot_launch(self):
        payload = {"source": "/fixture/source", "dataset": "/fixture/dataset", "root": "/fixture/run",
                   "mode": "full", "configs": ["FW256_FS8"], "branches": ["rf"],
                   "git_commit": "fixture", "confirm_provisional": True, "confirmation": "RUN PROVISIONAL run"}
        keys = ("source", "dataset", "root", "mode", "configs", "branches", "git_commit")
        audit = {"status": "PASS", "request": {k: payload[k] for k in keys}}
        evidence = {"passed": True, "source_sha256": "same"}
        with patch("colab.colab_pipeline_v2.source_snapshot", return_value={"sha256": "same"}):
            require_launch_context(payload, audit, dataset_ready=True, unit_test_evidence=evidence)
            with self.assertRaisesRegex(ContractError, "changed after preflight"):
                require_launch_context({**payload, "branches": ["svm"]}, audit,
                                       dataset_ready=True, unit_test_evidence=evidence)

    def test_saved_summary_reads_latest_failure_instead_of_trusting_complete_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = {config: "fixture_" + config for config in CONFIGS}
            write_json_once(root / "RUN_INFO.json", {"run_ids": ids})
            write_json_once(root / "stage4_mi/runs/FW256_FS8" / ids["FW256_FS8"] / "MI_COMPLETE.json",
                            {"status": "COMPLETE"})
            for index, status in enumerate(("PASS", "FAIL"), start=1):
                write_json_once(root / "reports" / f"verify_{index}.json", {
                    "action": "verify_mi", "config": "FW256_FS8", "branch": None,
                    "program_status": status, "exit_code": 0 if status == "PASS" else 2,
                    "end": f"2026-10-27T10:00:0{index}+00:00", "result": {"status": status}})
            row = next(r for r in completed_results(root) if r["config"] == "FW256_FS8" and r["part"] == "mi")
            self.assertEqual(row["verification_status"], "FAIL")
            self.assertEqual(row["verify_exit_code"], 2)

    def test_dry_run_cannot_enter_scientific_fit_api(self):
        with patch("colab.colab_pipeline_v2.science_modules") as science:
            with self.assertRaisesRegex(ContractError, "full/resume"):
                execute_task({"root": "/unused", "source": "/unused", "dataset": "/unused",
                              "config": "FW256_FS8", "branch": "rf", "mode": "dry_run"}, "mi")
            science.assert_not_called()

    def test_run_ids_are_persisted_and_resume_never_generates_a_new_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive = Path(tmp)
            source, dataset = fake_layout(drive)
            first = initialize_run(drive, source, dataset, "dry_run")
            second = initialize_run(drive, source, dataset, "full", session_root=first["root"])
            resumed = initialize_run(drive, source, dataset, "resume", existing=first["root"])
            self.assertEqual(first["root"], second["root"])
            self.assertEqual(first["info"]["run_ids"], resumed["info"]["run_ids"])
            self.assertEqual(len(set(first["info"]["run_ids"].values())), 4)
            with self.assertRaises(ContractError):
                initialize_run(drive, source, dataset, "resume")

    def test_legacy_run_is_not_adopted_as_fresh_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive = Path(tmp)
            source, dataset = fake_layout(drive)
            old = drive / "COMPACT_MI_RFE_FRESH_RUNS/OLD"
            old.mkdir(parents=True)
            write_json_once(old / "RUN_INFO.json", {"run_id": "OLD"})
            with self.assertRaisesRegex(ContractError, "UNSUPPORTED"):
                initialize_run(drive, source, dataset, "resume", existing=str(old))
            self.assertEqual(json.loads((old / "RUN_INFO.json").read_text()), {"run_id": "OLD"})

    def test_import_ignores_an_existing_colab_package_and_uses_exact_drive_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            source, _ = fake_layout(Path(tmp))
            previous = list(sys.path)
            try:
                import types
                stale = types.ModuleType("colab")
                stale.__path__ = ["/content/RFforNetwork/code_CICFLOW/colab"]
                sys.modules["colab"] = stale
                loaded = activate_source(source)
                self.assertEqual(Path(loaded.__file__).resolve(), (source / "colab/colab_pipeline.py").resolve())
            finally:
                sys.path[:] = previous

    def test_loader_finds_real_files_without_package_init_and_runs_each_once(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            source, _ = fake_layout(Path(tmp))
            previous = list(sys.path)
            try:
                for relative in ("colab/test_root.py", "stage4_mi/checks/test_nested.py", "stage5_rfe/test_root.py"):
                    path = source / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("import unittest\nclass Fixture(unittest.TestCase):\n    def test_one(self):\n        self.assertEqual(1, 1)\n", encoding="utf-8")
                found = discover_test_files(source)
                self.assertEqual(sum(map(len, found.values())), 3)
                report = run_test_files(source)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(sum(r["tests_run"] for r in report["areas"].values()), 3)
            finally:
                sys.path[:] = previous

    def test_zero_skipped_and_failed_tests_are_not_pass(self):
        self.assertEqual(tests_status(0), "NO_TESTS_FOUND")
        self.assertEqual(tests_status(2, skipped=1), "SKIPPED")
        self.assertEqual(tests_status(2, errors=1), "FAIL")
        self.assertEqual(tests_status(2), "PASS")

    def test_source_hash_changes_when_scientific_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = fake_layout(Path(tmp))
            old = source_snapshot(source)
            (source / "stage4_mi/mi_selection.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
            self.assertNotEqual(old["sha256"], source_snapshot(source)["sha256"])

    def test_checkpoint_scores_must_match_completed_indices(self):
        validate_checkpoint_state("mi", {"completed_indices": [0, 1], "scores": {"0": 0.0, "1": 0.5}})
        with self.assertRaises(ContractError):
            validate_checkpoint_state("mi", {"completed_indices": [0, 1], "scores": {"0": 0.0}})

    def test_corrupt_rfe_elimination_history_is_rejected(self):
        state = {"completed_steps": 1, "support": [False] + [True] * 99,
                 "ranking": [2] + [1] * 99, "active_indices": list(range(1, 100))}
        validate_checkpoint_state("rfe", state)
        state["ranking"][0] = 100
        with self.assertRaises(ContractError):
            validate_checkpoint_state("rfe", state)

    def test_complete_task_is_verified_instead_of_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json_once(run / "MI_COMPLETE.json", {"status": "COMPLETE"})
            self.assertEqual(resume_plan(run, "mi"), "VERIFY_COMPLETED_TASK")

    def test_partial_publication_is_unsupported_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            original = {"partial": True}
            write_json_once(run / "rfe_ranking.json", original)
            with self.assertRaisesRegex(ContractError, "UNSUPPORTED"):
                resume_plan(run, "rfe")
            with self.assertRaises(FileExistsError):
                write_json_once(run / "rfe_ranking.json", {})
            self.assertEqual(json.loads((run / "rfe_ranking.json").read_text()), original)

    def test_manual_rfe_resumed_ranking_matches_uninterrupted_and_sklearn(self):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        from stage5_rfe.rfe_selection import manual_rfe_step1

        rng = np.random.RandomState(9)
        X = rng.normal(size=(32, 6)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Small fixture estimator only; production estimator parameters stay unchanged.
        estimator = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42, n_jobs=1)
        snapshots = []

        class InterruptedFixture(Exception):
            pass

        def stop_after_two(state):
            snapshots.append(state)
            if state["completed_steps"] == 2:
                raise InterruptedFixture()

        with self.assertRaises(InterruptedFixture):
            manual_rfe_step1(X, y, estimator, checkpoint_callback=stop_after_two)
        last = snapshots[-1]
        resumed, support, steps = manual_rfe_step1(
            X, y, estimator, initial_support=last["support"], initial_ranking=last["ranking"],
            completed_steps=last["completed_steps"])
        reference = RFE(estimator, n_features_to_select=1, step=1).fit(X, y)
        np.testing.assert_array_equal(resumed, reference.ranking_)
        np.testing.assert_array_equal(support, reference.support_)
        self.assertEqual(steps, 5)

if __name__ == "__main__":
    unittest.main()
