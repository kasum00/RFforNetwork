"""Mock-only Stage 7.3B authorization tests; no dataset rows or model fits."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colab import svm_run_authorization as auth
from colab.experiment_launcher import RunSpec


def fake_identity() -> dict:
    values = (
        "baseline_svm", "FW_512_FS_16", "baseline_svm_FW512_FS16_official01",
        "a" * 40, *["b" * 64 for _ in range(len(auth.IDENTITY_FIELDS) - 4)],
    )
    return dict(zip(auth.IDENTITY_FIELDS, values))


class AuthorizationTests(unittest.TestCase):
    def test_generated_template_is_deliberately_invalid(self):
        template = auth.invalid_approval_template(fake_identity())
        self.assertFalse(template["effective"])
        self.assertNotEqual(template["decision"], "APPROVED_STAGE73B")
        self.assertEqual(template["confirmation"], "")
        self.assertTrue(all(value is None for value in template["budgets"].values()))

    def test_confirmation_binds_experiment_config_run_and_commit(self):
        identity = fake_identity()
        phrase = auth.expected_confirmation(identity)
        for value in (identity["experiment"], identity["config"],
                      identity["run_name"], identity["git_commit"]):
            self.assertIn(value, phrase)

    def test_each_identity_or_budget_mismatch_blocks(self):
        identity = fake_identity()
        budgets = {name: 10 for name in auth.REQUIRED_BUDGET_FIELDS}
        budgets["poll_interval_seconds"] = 5
        for field in auth.IDENTITY_FIELDS:
            altered = {**identity, field: "different"}
            self.assertFalse(auth.identity_matches(altered, identity))
        self.assertTrue(auth.identity_matches(identity.copy(), identity))
        self.assertFalse(auth.identity_matches(
            {key: value for key, value in identity.items()
             if key != "dataset_fingerprint_scope_sha256"}, identity))
        for field in auth.REQUIRED_BUDGET_FIELDS:
            with self.subTest(field=field), self.assertRaises(PermissionError):
                auth._positive_integer({**budgets, field: 0}[field], field)

    def test_context_cannot_be_forged_with_plain_object(self):
        with self.assertRaises(PermissionError):
            auth.validate_context_for_training(
                object(), "FW_512_FS_16", "baseline_svm",
                Path("baseline_svm_FW512_FS16_official01"))

    def test_existing_completed_or_incomplete_name_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local, drive = root / "local", root / "drive"
            local.mkdir(); drive.mkdir()
            candidates = (
                local / "run01", local / "run01.incomplete",
                drive / "experiment_results" / "run01",
                drive / "checkpoints" / "run01.incomplete",
                drive / "environment" / "run01.json",
            )
            for candidate in candidates:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                if candidate.suffix == ".json": candidate.write_text("{}", encoding="utf-8")
                else: candidate.mkdir()
                self.assertIn(str(candidate), auth._output_conflicts(local, drive, "run01"))
                if candidate.is_file(): candidate.unlink()
                else: candidate.rmdir()

    def test_direct_svm_runner_blocks_before_window_store(self):
        from huyen_code import ciciomt24_training as training
        with patch.object(training, "WindowStore",
                          side_effect=AssertionError("dataset must not open")):
            with self.assertRaises(PermissionError):
                training.run_official(Path("windows"), Path("original"), Path("run"),
                                      "FW_512_FS_16", "baseline_svm",
                                      confirm_full_training=True)

    def test_rf_path_does_not_require_svm_authorization(self):
        from huyen_code import ciciomt24_training as training
        marker = RuntimeError("RF reached its unchanged store path")
        with patch.object(training, "WindowStore", side_effect=marker):
            with self.assertRaisesRegex(RuntimeError, "unchanged store"):
                training.run_official(Path("windows"), Path("original"), Path("rf_run"),
                                      "FW_512_FS_16", "mi_rfe_rf",
                                      confirm_full_training=True)

    def test_exact_probe_and_replacement_policy_are_not_equivalent(self):
        identity = fake_identity()
        with tempfile.TemporaryDirectory() as temp:
            evidence_path = Path(temp) / "probe.json"
            evidence = {
                "status": "PASS_TECHNICAL_ONLY", "full_training_authorized": False,
                "experiment": "baseline_svm", "config": "FW_256_FS_8",
                "training_code_sha256": "x", "window_summary_sha256": "y",
                "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                "scientific_result": False, "official_best_k": None,
                "runs": [{"status": "PASS_TECHNICAL_ONLY",
                          "validation_csv_rows_read": 0, "test_csv_rows_read": 0,
                          "scientific_result": False, "official_best_k": None,
                          "worker_returncode": 0, "supervisor_stop_reason": None}],
            }
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            observed = {key: evidence.get(key) for key in (
                "experiment", "config", "training_code_sha256", "window_summary_sha256")}
            policy = {"technical_evidence": {
                "policy": "EXACT_CONFIG_PROBE", "evidence_sha256": auth.sha256(evidence_path),
                "evidence_identity": observed}}
            with self.assertRaises(PermissionError):
                auth._validate_technical_evidence(policy, evidence_path, identity)
            policy["technical_evidence"].update(
                policy="EXPLICIT_APPROVED_REPLACEMENT",
                approved_replacement_policy="Human accepts this bounded cross-config evidence is not an equivalent full-run measurement.",
                human_accepts_non_equivalence=True)
            result = auth._validate_technical_evidence(policy, evidence_path, identity)
            self.assertEqual(result["policy"], "EXPLICIT_APPROVED_REPLACEMENT")


if __name__ == "__main__":
    unittest.main()
