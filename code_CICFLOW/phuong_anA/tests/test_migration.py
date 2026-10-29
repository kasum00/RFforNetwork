"""Synthetic migration fixtures; no Git, real dataset, model load or training."""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .fixtures import SOURCE, small_identity
from .test_a1 import TiedForest
from phuong_anA.contract import Blocked, atomic, core, digest, info, paths, read, scientific, sha
from phuong_anA.migration import (PHASE_FILES, REVIEWED_TRANSITIONS, apply, artifacts, phrase,
                                  prepare, source_compatibility, verify_migration)
from phuong_anA.recovery import inspect_run
from phuong_anA.rfe import fit_step5
from phuong_anA.store import Store, ledger
from mi_rfe_svm_colab.local_progress import KProgress


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = {"source": str(SOURCE), "dataset": str(self.root / "CICFLOWMETER_FEATURE_CSV"),
            "output": str(self.root / "experiment_results"), "local_root": str(self.root / "local"),
            "production": False, "config": "FW256_FS8", "run_id": "RUN_A1_01_FIX01"}
        before = {name: pair[0] for name, pair in REVIEWED_TRANSITIONS.items()}
        before["phuong_anA/rfe.py"] = sha(SOURCE / "phuong_anA/rfe.py")
        old_source = {"files": before, "sha256": digest(before)}
        after = {**before, **{name: pair[1] for name, pair in REVIEWED_TRANSITIONS.items()},
                 "phuong_anA/migration.py": "c"*64}
        self.new_source = {"files": after, "sha256": digest(after)}
        names = [f"f{i}" for i in range(380)]
        binding = {"names": names, "splits": {"train": {"rows": 8}, "val": {"rows": 4}, "test": {"rows": 4}}}
        audit = {"binding": binding}; audit["fingerprint"] = digest(audit)
        self.old = small_identity(run_id="RUN_A1_01", source=old_source, dataset=binding, audit=audit,
            dataset_fingerprint=audit["fingerprint"], contract=scientific(SOURCE), environment={"fixture": True})
        self.new = {**self.old, "run_id": self.request["run_id"], "source": self.new_source, "git_commit": "b"*40}
        self.new["sha256"] = digest({k: v for k, v in self.new.items() if k != "sha256"})
        self.origin = Path(self.request["output"]) / "phuong_anA/mi_rfe_svm/FW256_FS8/RUN_A1_01"
        self.old_store = Store(self.root / "origin_local", self.origin, self.old, resume=False)
        state = self.old_store.state
        atomic(state / "run_identity.json", self.old)
        top = names[:100]
        atomic(state / "mi_top100.json", top)
        with (state / "mi_feature_scores.csv").open("x", encoding="utf-8") as f:
            f.write(",0\n" + "".join(f"{n},{380-i}\n" for i, n in enumerate(names)))
        atomic(state / "mi_state.json", {"identity": self.old["sha256"], "seconds": 1.25,
            "files": {n: info(state / n) for n in ("mi_top100.json", "mi_feature_scores.csv")}})
        (state / "top100_scaler.joblib").write_bytes(b"opaque synthetic scaler; loading is forbidden")
        atomic(state / "top100_scaler_state.json", {"binding": {"identity": self.old["sha256"],
            "mi": sha(state / "mi_state.json"), "top": top}, "fit_role": "train", "rows": 8,
            "files": {"top100_scaler.joblib": info(state / "top100_scaler.joblib")}})
        result = fit_step5(np.ones((8, 100), np.float32), np.array([0, 1]*4), top, TiedForest())
        with (state / "rfe_feature_ranking.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["records"][0]))
            writer.writeheader(); writer.writerows(result["records"])
        atomic(state / "rfe_state.json", {"binding": {"identity": self.old["sha256"], "mi": sha(state / "mi_state.json"),
            "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)}, "result": result,
            "files": {"rfe_feature_ranking.csv": info(state / "rfe_feature_ranking.csv")}})
        self.ranked = result["names"]
        for stage, phase_files in PHASE_FILES.items():
            self.old_store.commit(stage, {n: state / n for n in phase_files})
        metrics = core(SOURCE).metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]))
        self.row = {"k": 1, "features": self.ranked[:1], **metrics,
            "fit": {"seconds": 1.2, "converged": True, "warnings": []}, "predict_seconds": 0.1}
        history = KProgress(state, self.old, self.ranked, sha(state / "rfe_state.json"))
        history.append(self.row)  # Only LOCAL, as in the actual failed run.
        atomic(state / "failure.json", {"error": "best_validation_accuracy", "completed": False})
        self.old_store.commit("failed_keyerror", {"failure.json": state / "failure.json"})
        self.old_store.release("FAILED")
        self.backup = Path(self.request["output"]) / "phuong_anA/recovery/FW256_FS8/RUN_A1_01/LOCAL_K001_fixture"
        self.backup.mkdir(parents=True)
        sources = {"k_001.json": state / "k/k_001.json", "original_run_manifest.json": self.origin / "run_manifest.json",
                   "original_rfe_state.json": state / "rfe_state.json"}
        for name, path in sources.items():
            shutil.copyfile(path, self.backup / name)
        atomic(self.backup / "recovery_manifest.json", {"purpose": "RECOVERY_COPY_ONLY_NOT_RUN_CHECKPOINT",
            "original_run": str(self.origin), "original_identity": self.old["sha256"], "original_git_commit": self.old["git_commit"],
            "dataset_fingerprint": self.old["dataset_fingerprint"], "files": {n: info(p) for n, p in sources.items()}})
        self.preflight = patch("phuong_anA.preflight.review", return_value={"status": "PASS", "identity": self.new})
        self.review = self.preflight.start(); self.addCleanup(self.preflight.stop)

    def plan(self):
        return prepare(self.request, self.origin, self.backup)

    def file_snapshot(self, root):
        return {p.relative_to(root).as_posix(): info(p) for p in root.rglob("*") if p.is_file()}

    def test_plan_is_read_only_and_exact_reviewed_transition_required(self):
        before = self.file_snapshot(self.root)
        with patch("pathlib.Path.mkdir", side_effect=AssertionError("Plan must not create directories")):
            plan = self.plan()
        self.assertEqual(before, self.file_snapshot(self.root))
        self.assertEqual(plan["next_fit_k"], 2)
        self.assertEqual(plan["original_committed_k"], [])
        self.assertEqual(plan["validation_accuracy_k1"], self.row["accuracy"])
        self.assertFalse(plan["training_performed"])

    def test_apply_preserves_origin_and_k1_and_has_no_fit_or_deserialization(self):
        plan = self.plan()
        origin_before, backup_before = self.file_snapshot(self.origin), self.file_snapshot(self.backup)
        with patch("joblib.load", side_effect=AssertionError("No model/scaler load during migration")), \
             patch("sklearn.svm.SVC.fit", side_effect=AssertionError("No SVC fit")), \
             patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("No RF fit")), \
             patch.object(core(SOURCE), "reference_mi", side_effect=AssertionError("No MI refit")), \
             patch.object(core(SOURCE), "read_values", side_effect=AssertionError("No dataset values during migration")):
            result = apply(self.request, self.origin, self.backup,
                plan_sha256=plan["plan_sha256"], authorization=phrase(plan))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["next_fit_k"], 2)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["test_values_read"])
        self.assertEqual(origin_before, self.file_snapshot(self.origin))
        self.assertEqual(backup_before, self.file_snapshot(self.backup))
        local, remote = paths(self.request)
        self.assertEqual(inspect_run(remote)["committed_k"], [1])
        self.assertEqual(read(local / "state/k/k_001.json")["row"], self.row)
        self.assertNotEqual(read(local / "state/k/k_001.json")["binding"]["identity"], self.old["sha256"])
        restored = Store(self.root / "new_runtime", remote, self.new, resume=True)
        for stage in ("mi", "scaler", "rfe", "k_001"):
            self.assertIsNotNone(restored.saved(stage))
        history = KProgress(restored.state, self.new, self.ranked, sha(restored.state / "rfe_state.json"))
        self.assertEqual(history.rows, [self.row])

    def test_unreviewed_numerical_source_change_is_blocked(self):
        changed = {**self.new_source["files"], "phuong_anA/rfe.py": "f"*64}
        with self.assertRaisesRegex(Blocked, "Unreviewed source change"):
            source_compatibility(self.old["source"], {"files": changed, "sha256": digest(changed)})

    def test_reviewed_fixed_files_match_current_checkout_bytes(self):
        for name, pair in REVIEWED_TRANSITIONS.items():
            with self.subTest(file=name):
                self.assertEqual(sha(SOURCE / name), pair[1], "Source drift requires a new migration review")

    def test_unknown_fix_bytes_or_source_addition_is_blocked(self):
        for name in ("phuong_anA/telemetry.py", "phuong_anA/unknown_algorithm.py"):
            changed = {**self.new_source["files"], name: "f"*64}
            with self.subTest(name=name), self.assertRaises(Blocked):
                source_compatibility(self.old["source"], {"files": changed, "sha256": digest(changed)})

    def test_changed_dataset_environment_or_contract_cannot_migrate(self):
        for field, value in (("dataset_fingerprint", "changed"), ("environment", {"changed": True}),
                             ("contract", {"rfe_step": 1})):
            changed = {**self.new, field: value}
            changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
            self.review.return_value = {"status": "PASS", "identity": changed}
            with self.subTest(field=field), self.assertRaisesRegex(Blocked, "Dataset/splits/environment"):
                self.plan()

    def test_recovery_copy_checksum_failure_blocks(self):
        with (self.backup / "k_001.json").open("ab") as f: f.write(b"tamper")
        with self.assertRaises(Blocked): self.plan()

    def test_exact_plan_authorization_and_existing_target_guard(self):
        plan = self.plan()
        with self.assertRaisesRegex(Blocked, "authorization"):
            apply(self.request, self.origin, self.backup, plan_sha256=plan["plan_sha256"], authorization="wrong")
        self.assertFalse(paths(self.request)[1].exists())
        paths(self.request)[1].mkdir()
        with self.assertRaisesRegex(Blocked, "already exists"):
            self.plan()

    def test_source_run_requires_a_released_worker(self):
        claims = sorted((self.origin / "logs").glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        lease = read(claims[-1])
        release = self.origin / "logs" / (lease["id"] + ".RELEASED.json")
        release.unlink()  # Only this synthetic temporary fixture.
        with self.assertRaisesRegex(Blocked, "not proven released"):
            self.plan()

    def test_partial_publication_is_not_an_admissible_run(self):
        plan = self.plan()
        original_copy = __import__("phuong_anA.migration", fromlist=["copy_once"]).copy_once
        target = paths(self.request)[1] / "run_manifest.json"
        def fail_manifest(source, destination):
            if Path(destination) == target:
                raise OSError("Disconnected before admission manifest")
            return original_copy(source, destination)
        with patch("phuong_anA.migration.copy_once", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "Disconnected"):
                apply(self.request, self.origin, self.backup,
                    plan_sha256=plan["plan_sha256"], authorization=phrase(plan))
        self.assertFalse(target.exists())
        self.assertTrue((target.parent / "MIGRATION_INTENT.json").exists())
        with self.assertRaises(Blocked):
            Store(self.root / "forbidden_resume", target.parent, self.new, resume=True)
