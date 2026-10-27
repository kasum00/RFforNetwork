from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage5_rfe.rfe_selection import (
    LABEL_COLUMN,
    MI_TOP_K,
    Stage5ContractError,
    _validate_resume,
    manual_rfe_step1,
    preprocess_top_matrix,
    ranking_records,
    read_train_top_values,
    resource_gate,
    validate_mi_top100_payload,
    write_exclusive,
    write_json_exclusive,
)


class _ImportanceColumn:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, zero_copy_only=False):
        del zero_copy_only
        return self._values


class _FakeTable:
    def __init__(self, columns):
        self.schema = type("Schema", (), {"names": list(columns)})()
        self._columns = {
            name: _ImportanceColumn(np.arange(4, dtype=np.float32) + index)
            for index, name in enumerate(columns[:-1])
        }
        self._columns[LABEL_COLUMN] = _ImportanceColumn([0, 1, 0, 1])

    def column(self, name):
        return self._columns[name]


class _FixedImportanceEstimator:
    def __init__(self, offset=0):
        self.offset = offset

    def get_params(self, deep=True):
        del deep
        return {"offset": self.offset}

    def set_params(self, **params):
        self.offset = params.get("offset", self.offset)
        return self

    def fit(self, X, y):
        del y
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float) + self.offset
        return self


class _TieImportanceEstimator(_FixedImportanceEstimator):
    def fit(self, X, y):
        del y
        self.feature_importances_ = np.ones(X.shape[1], dtype=float)
        return self


class Stage5RFEUnitTests(unittest.TestCase):
    def test_train_preprocessing_preserves_rows_labels_and_replaces_nonfinite(self):
        matrix = np.zeros((3, MI_TOP_K), dtype=float)
        matrix[0, 0] = np.nan
        matrix[1, 1] = np.inf
        matrix[2, 2] = -np.inf
        labels = np.array([0, 1, 0])
        processed, processed_labels = preprocess_top_matrix(matrix, labels)
        self.assertEqual(processed.shape, (3, MI_TOP_K))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.isfinite(processed).all())
        np.testing.assert_array_equal(processed_labels, labels)

    def test_mi_binding_requires_100_unique_features_and_correct_mapping(self):
        names = [f"feature_{index:03d}" for index in range(380)]
        payload = {
            "top_k": MI_TOP_K,
            "features": [
                {"rank": index + 1, "feature_index": index, "feature_name": names[index], "mi_score": float(100 - index)}
                for index in range(MI_TOP_K)
            ],
        }
        result = validate_mi_top100_payload(payload, names)
        self.assertEqual(len(result), MI_TOP_K)
        payload["features"][1]["feature_name"] = payload["features"][0]["feature_name"]
        with self.assertRaises(Stage5ContractError):
            validate_mi_top100_payload(payload, names)

    def test_train_reader_receives_only_selected_train_path_and_columns(self):
        names = [f"mi_feature_{index:03d}" for index in range(MI_TOP_K)]
        calls = []

        def reader(path, columns):
            calls.append((path, columns))
            return _FakeTable(columns)

        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.parquet"
            matrix, labels = read_train_top_values(train_path, names, table_reader=reader)
        self.assertEqual(calls, [(train_path, names + [LABEL_COLUMN])])
        self.assertEqual(matrix.shape, (4, MI_TOP_K))
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))

    def test_manual_rfe_matches_sklearn_rfe_step_one(self):
        from sklearn.feature_selection import RFE

        X = np.arange(48, dtype=np.float32).reshape(8, 6)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        estimator = _FixedImportanceEstimator()
        manual_ranking, manual_support, steps = manual_rfe_step1(X, y, estimator)
        reference = RFE(_FixedImportanceEstimator(), n_features_to_select=1, step=1).fit(X, y)
        np.testing.assert_array_equal(manual_ranking, reference.ranking_)
        np.testing.assert_array_equal(manual_support, reference.support_)
        self.assertEqual(steps, X.shape[1] - 1)

    def test_tie_semantics_match_sklearn_stable_active_order(self):
        from sklearn.feature_selection import RFE

        X = np.arange(40, dtype=np.float32).reshape(8, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        manual_ranking, manual_support, _ = manual_rfe_step1(X, y, _TieImportanceEstimator())
        reference = RFE(_TieImportanceEstimator(), n_features_to_select=1, step=1).fit(X, y)
        np.testing.assert_array_equal(manual_ranking, reference.ranking_)
        np.testing.assert_array_equal(manual_support, reference.support_)

    def test_ranking_records_keep_mi_order_mapping(self):
        top = [
            {"rank": 1, "feature_index": 7, "feature_name": "a", "mi_score": 1.0},
            {"rank": 2, "feature_index": 2, "feature_name": "b", "mi_score": 0.5},
        ]
        records = ranking_records(top, [2, 1], [False, True])
        self.assertEqual([row["feature_name"] for row in records], ["b", "a"])
        self.assertEqual(records[0]["mi_rank"], 2)

    def test_resource_gate_blocks_without_sampling_or_parameter_reduction(self):
        result = resource_gate(
            {
                "memory": {"available_bytes": 1},
                "disk": {"free_bytes": 10 * 1024**3},
                "library_versions": {"numpy": "1", "scikit-learn": "1", "pyarrow": "1"},
            },
            rows=1000,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["no_sampling_or_parameter_reduction"])

    def test_resume_identity_and_duplicate_writer_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity = {"stage": "5A", "config": "FW256_FS8", "branch": "rf", "run_id": "R"}
            write_json_exclusive(run_dir / "RUN_INCOMPLETE.json", {"status": "INCOMPLETE"})
            write_json_exclusive(run_dir / "rfe_manifest.json", {"run_identity": identity})
            write_json_exclusive(run_dir / "rfe_progress.json", {"run_identity": identity, "completed_steps": 0})
            self.assertEqual(_validate_resume(run_dir, identity)["completed_steps"], 0)
            with self.assertRaises(Stage5ContractError):
                _validate_resume(run_dir, {**identity, "branch": "svm"})
            with self.assertRaises(FileExistsError):
                write_exclusive(run_dir / "RUN_INCOMPLETE.json", "overwrite")

    def test_checkpoint_missing_artifacts_is_not_silently_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaises(Stage5ContractError):
                _validate_resume(run_dir, {"run_id": "R"})


if __name__ == "__main__":
    unittest.main()

