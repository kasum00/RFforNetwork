import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from stage9_svm_final.common import Blocked, METHODS, digest, once, paths, sha
from stage9_svm_final.preflight import positive_labels, resource_gate, review
from stage9_svm_final.reports import collect, export
from stage9_svm_final.selection import discover, validate_extract, verify_selection
from .fixtures import fixture


class AdmissionReports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request, self.plan, _ = fixture(self.root)

    def extract(self):
        identity = dict(algorithm=METHODS["A1"][1], config="FW256_FS8", run_id="RUN_A1_UNIT", git_commit="a"*40,
            contract=dict(rfe=dict(step=5)), dataset=self.plan["identity"]["dataset"]["binding"],
            audit=self.plan["identity"]["dataset"], dataset_fingerprint=self.plan["identity"]["dataset"]["fingerprint"])
        identity["sha256"] = digest(identity)
        ranked = self.plan["identity"]["features"] + [n for n in identity["dataset"]["names"]
                                                          if n not in self.plan["identity"]["features"]][:97]
        return dict(identity=identity, features=self.plan["identity"]["features"], best_k=3, ranked=ranked,
            audit=identity["audit"], frozen=dict(features=self.plan["identity"]["features"], best_k=3, identity=identity["sha256"]),
            verification=dict(status="PASS", phase_status="SELECTION_COMPLETE", test_evaluation_calls=0))

    def test_discovery_never_equates_completion_marker_to_verified(self):
        root = Path(self.request["output"]) / "phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_UNIT"
        once(root / "run_manifest.json", {"identity": self.extract()["identity"]})
        once(root / "results/SELECTION_COMPLETE.json", {"status": "SELECTION_COMPLETE"})
        result = discover(self.request["output"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "BLOCKED_SELECTION_NOT_VERIFIED")
        self.assertTrue(result[0]["completion_present"])

    def test_discovery_incomplete_is_explicitly_blocked(self):
        root = Path(self.request["output"]) / "phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_UNIT"
        once(root / "run_manifest.json", {"identity": self.extract()["identity"]})
        self.assertEqual(discover(self.request["output"])[0]["status"], "BLOCKED_SELECTION_INCOMPLETE")

    def test_valid_extraction_preserves_feature_order(self):
        value = self.extract()
        self.assertEqual(validate_extract(value, self.request)["features"], value["ranked"][:3])

    def test_extraction_rejects_method_config_run_step_and_sha_mismatch(self):
        for key, wrong in (("algorithm", METHODS["B"][1]), ("config", "FW512_FS8"), ("run_id", "OTHER")):
            value = self.extract()
            value["identity"][key] = wrong
            value["identity"]["sha256"] = digest({k: v for k, v in value["identity"].items() if k != "sha256"})
            with self.assertRaises(Blocked):
                validate_extract(value, self.request)
        value = self.extract()
        value["identity"]["sha256"] = "0"*64
        with self.assertRaises(Blocked):
            validate_extract(value, self.request)

    def test_extraction_rejects_unverified_and_duplicate_or_wrong_order_features(self):
        for change in ("unverified", "duplicate", "order", "count", "frozen"):
            value = self.extract()
            if change == "unverified":
                value["verification"]["status"] = "BLOCKED"
            elif change == "duplicate":
                value["features"] = value["features"][:1]*3
            elif change == "order":
                value["features"] = list(reversed(value["features"]))
            elif change == "count":
                value["best_k"] = 4
            else:
                value["frozen"]["features"] = []
            with self.assertRaises(Blocked):
                validate_extract(value, self.request)

    def test_original_verifier_failure_cannot_be_promoted_by_marker(self):
        result = SimpleNamespace(returncode=2, stderr="SHA mismatch", stdout="__STAGE9_SELECTION__"+json.dumps(self.extract()))
        with patch("stage9_svm_final.selection.subprocess.run", return_value=result):
            with self.assertRaisesRegex(Blocked, "original verifier failed"):
                verify_selection(self.request)

    def test_selection_worker_incomplete_blocks_before_git_or_data(self):
        from stage9_svm_final.selection_worker import verify
        with patch("stage9_svm_final.selection_worker.subprocess.run", side_effect=AssertionError("Git before completion gate")), \
             patch("pyarrow.parquet.read_table", side_effect=AssertionError("data in discovery")):
            with self.assertRaisesRegex(Blocked, "INCOMPLETE"):
                verify(self.request)

    def test_selection_worker_bad_identity_blocks_before_import(self):
        from stage9_svm_final.selection_worker import verify
        root = Path(self.request["output"]) / "phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_UNIT"
        identity = self.extract()["identity"]
        identity["sha256"] = "bad"
        once(root / "run_manifest.json", {"identity": identity})
        once(root / "results/SELECTION_COMPLETE.json", {"status": "SELECTION_COMPLETE"})
        with patch("stage9_svm_final.selection_worker.importlib.import_module", side_effect=AssertionError("source imported before integrity")):
            with self.assertRaisesRegex(Blocked, "self-digest"):
                verify(self.request)

    def test_selection_worker_source_hash_checked_before_import_or_test(self):
        from stage9_svm_final.selection_worker import verify
        root = Path(self.request["output"]) / "phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_UNIT"
        source = self.root / "archived_repo/code_CICFLOW"
        source.mkdir(parents=True)
        (source / "owned.py").write_text("# fixture only", encoding="utf-8")
        identity = self.extract()["identity"]
        identity["git_commit"] = "a"*40
        identity["source"] = {"files": {"owned.py": "0"*64}}
        identity["source"]["sha256"] = digest(identity["source"]["files"])
        identity["sha256"] = digest({k: v for k, v in identity.items() if k != "sha256"})
        once(root / "run_manifest.json", {"identity": identity})
        once(root / "results/SELECTION_COMPLETE.json", {"status": "SELECTION_COMPLETE"})
        def git(args, **kw):
            return SimpleNamespace(stdout="a"*40 if "rev-parse" in args else "")
        # Resolve the parquet module before patching the worker's import hook.
        # Otherwise mock.patch() itself can trigger the guarded import.
        import pyarrow.parquet as parquet
        with patch("stage9_svm_final.selection_worker.subprocess.run", side_effect=git), \
             patch("stage9_svm_final.selection_worker.importlib.import_module", side_effect=AssertionError("import before source hashes")), \
             patch.object(parquet, "read_table", side_effect=AssertionError("unexpected Test read")):
            with self.assertRaisesRegex(Blocked, "source byte mismatch"):
                verify({**self.request, "selection_source": str(source)})

    def test_positive_label_requires_verified_producer_mapping(self):
        audit = copy.deepcopy(self.plan["identity"]["dataset"])
        self.assertEqual(positive_labels(audit)["attack"], 1)
        audit["label_mapping"]["1"] = "Unknown"
        with self.assertRaises(Blocked):
            positive_labels(audit)

    def test_resource_gate_blocks_without_parameter_reduction(self):
        gate = resource_gate(self.plan["identity"]["dataset"], self.request["budget"], self.root,
                             sample=dict(available_ram_bytes=1, disk_free_bytes=1, logical_cpus=2, rss_bytes=1))
        self.assertEqual(gate["status"], "BLOCKED")
        self.assertGreater(gate["estimate"]["required_ram_bytes"], 10000*1024**2)

    def test_wrong_method_directory_is_not_shared(self):
        a = paths(self.request)[1]
        b = paths({**self.request, "method": "B"})[1]
        self.assertNotEqual(a, b)
        self.assertEqual(a.parts[-2:], ("A1", "FW256_FS8"))

    def test_preflight_selection_blocker_does_not_trigger_model_training(self):
        with patch("stage9_svm_final.preflight.environment", return_value={"errors": []}), \
             patch("stage9_svm_final.preflight.check_source", return_value={}), \
             patch("stage9_svm_final.preflight.verify_selection", side_effect=Blocked("incomplete chain")), \
             patch("sklearn.svm.SVC.fit", side_effect=AssertionError("fit in preflight")):
            with self.assertRaisesRegex(Blocked, "incomplete chain"):
                review(self.request)

    def test_preflight_fit_requires_current_source_test_evidence(self):
        identity = self.plan["identity"]
        selected = dict(reference_final_role="Train+Validation after frozen selection", audit=identity["dataset"],
            identity=dict(sha256="b"*64, git_commit="a"*40, resolved_estimators={"svm": identity["svm_resolved"]}),
            features=identity["features"], best_k=3, frozen=self.plan["selected"]["frozen"],
            checkpoint_sha256="c"*64, publication={"sha256": "e"*64, "bytes": 20},
            verified_helpers={name: sha(Path(self.request["source"]) / name) for name in
                              ("colab/cicflow_reference_pipeline.py", "mi_rfe_svm_colab/common.py")})
        with patch("stage9_svm_final.preflight.environment", return_value={"errors": []}), \
             patch("stage9_svm_final.preflight.check_source", return_value={"sha256": "d"*64}), \
             patch("stage9_svm_final.preflight.verify_selection", return_value=selected), \
             patch("stage9_svm_final.preflight.resource_gate", return_value={"status": "PASS"}):
            with self.assertRaisesRegex(Blocked, "unit-test PASS"):
                review({**self.request, "git_commit": "a" * 40}, need_fit=True)

    def test_preflight_rejects_archived_current_preprocessing_difference(self):
        selected = dict(verified_helpers={"colab/cicflow_reference_pipeline.py": "0"*64})
        with patch("stage9_svm_final.preflight.environment", return_value={"errors": []}), \
             patch("stage9_svm_final.preflight.check_source", return_value={}), \
             patch("stage9_svm_final.preflight.verify_selection", return_value=selected), \
             patch("mi_rfe_svm_colab.common.core", side_effect=AssertionError("changed helper reached final implementation")):
            with self.assertRaisesRegex(Blocked, "preprocessing helper differs"):
                review(self.request)

    def test_reports_all_eight_rows_blank_until_verified_test(self):
        rows, records = collect(self.request["output"])
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["Method"] for r in rows}, {"A1", "B"})
        self.assertTrue(all(r["Acc (%)"] == "" for r in rows))
        self.assertTrue(all(r["test_metrics"] is None for r in records))

    def test_reports_do_not_substitute_validation_or_accept_fake_test_json(self):
        root = paths(self.request)[1]
        once(root / "identity_manifest.json", self.plan["identity"])
        once(root / "test_metrics.json", {"accuracy": 1.0, "validation_accuracy": 0.999})
        rows, _ = collect(self.request["output"])
        row = next(r for r in rows if r["Method"] == "A1" and r["Config"] == "FW256_FS8")
        self.assertEqual(row["Acc (%)"], "")
        self.assertEqual(row["Verification status"], "BLOCKED")

    def test_exports_use_new_version_directory_and_keep_history(self):
        a = export(self.request["output"])
        before = (Path(a["directory"]) / "SVM_A1_B_FINAL_RESULTS.csv").read_bytes()
        b = export(self.request["output"])
        self.assertNotEqual(a["directory"], b["directory"])
        self.assertEqual((Path(a["directory"]) / "SVM_A1_B_FINAL_RESULTS.csv").read_bytes(), before)
