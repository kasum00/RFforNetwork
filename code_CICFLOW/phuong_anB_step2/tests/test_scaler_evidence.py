import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.preprocessing import StandardScaler
from phuong_anB_step2.contract import Blocked, atomic, info, paths, read, sha
from phuong_anB_step2.dataset import frozen_evidence
from mi_rfe_svm_colab.dataset import svc_prefix
from mi_rfe_svm_colab.runner import scaled_top, ranked_arrays
from .fixtures import SOURCE, small_identity


class ScalerEvidence(unittest.TestCase):
    def test_train_only_top_scaler_dtype_mapping_cache_and_svc_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, local = root / "state", root / "local"
            state.mkdir()
            identity = small_identity()
            top = identity["dataset"]["names"][:100]
            atomic(state / "mi_state.json", {"identity": identity["sha256"]}, exclusive=True)
            train = np.arange(8*380, dtype=np.float32).reshape(8, 380)
            val = np.arange(4*380, dtype=np.float32).reshape(4, 380) + 100000
            request = {"source": str(SOURCE)}
            X, V = scaled_top(request, state, local, identity, train, np.array([0, 1]*4),
                              val, np.array([0, 1]*2), top)
            expected = StandardScaler()
            np.testing.assert_allclose(X, expected.fit_transform(train[:, :100]))
            np.testing.assert_allclose(V, expected.transform(val[:, :100]))
            self.assertEqual(X.dtype, np.float32)
            self.assertEqual(V.dtype, np.float32)
            marker = read(state / "top100_scaler_state.json")
            self.assertEqual(marker["fit_role"], "train")
            self.assertEqual(marker["rows"], 8)
            before = info(state / "top100_scaler.joblib")
            with patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("unexpected refit")):
                repeated, _ = scaled_top(request, state, local, identity, train, np.array([0, 1]*4),
                                         val, np.array([0, 1]*2), top)
            np.testing.assert_array_equal(repeated, X)
            self.assertEqual(info(state / "top100_scaler.joblib"), before)
            ranked = list(reversed(top))
            Xr, _ = ranked_arrays(local, identity, X, V, top, ranked)
            np.testing.assert_array_equal(Xr[:, 0], X[:, -1])
            prefix = svc_prefix(Xr, 7)
            self.assertEqual(prefix.shape, (8, 7))
            self.assertEqual(prefix.dtype, np.float64)
            self.assertTrue(prefix.flags.c_contiguous)

    def evidence_fixture(self, root, absolute=True):
        bundle = read(SOURCE / "phuong_anB_step2/evidence/cicflow_split_evidence.json")
        folder = root / "CICFLOWMETER_FEATURE_CSV/FW512_FS8"
        folder.mkdir(parents=True)
        binding = {"input_paths": {}, "input_sha256": {}}
        for key in ("validation", "v2_preflight", "manifest", "v2_binding"):
            entry = bundle["files"][key]
            path = entry["source_relative_path"]
            binding["input_paths"][key] = "D:\\COMPACT-MI-RFE\\" + path.replace("/", "\\") if absolute else path
            binding["input_sha256"][key] = entry["sha256"]
        atomic(folder / ".stage3_preflight.json", {"binding": binding}, exclusive=True)
        request = {"source": str(SOURCE), "dataset": str(folder.parent),
                   "output": str(root / "experiment_results"), "local_root": str(root / "local"),
                   "config": "FW512_FS8", "run_id": "RUN_B2_UNIT", "production": False}
        return request, folder, bundle

    def test_windows_path_alias_uses_identical_frozen_bytes_without_dataset_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, folder, bundle = self.evidence_fixture(Path(temporary))
            before = info(folder / ".stage3_preflight.json")
            resolved = Path(frozen_evidence(request))
            self.assertNotIn(folder, resolved.parents)
            for entry in bundle["files"].values():
                name = entry["source_relative_path"].rsplit("/", 1)[-1]
                self.assertEqual(sha(resolved / name), entry["sha256"])
            self.assertEqual(info(folder / ".stage3_preflight.json"), before)
            self.assertEqual([p.name for p in folder.iterdir()], [".stage3_preflight.json"])

    def test_wrong_split_hash_is_blocked_and_no_evidence_cache_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, folder, _ = self.evidence_fixture(Path(temporary))
            value = read(folder / ".stage3_preflight.json")
            value["binding"]["input_sha256"]["manifest"] = "0"*64
            (folder / ".stage3_preflight.json").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(Blocked):
                frozen_evidence(request)
            local, _ = paths(request)
            self.assertFalse((local / "frozen_evidence").exists())

    def test_matching_relative_evidence_keeps_bundled_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, _, _ = self.evidence_fixture(Path(temporary), absolute=False)
            self.assertIsNone(frozen_evidence(request))
