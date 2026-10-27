from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage4_mi.mi_selection import (
    EXPECTED_FEATURE_COUNT,
    LABEL_COLUMN,
    Stage4ContractError,
    estimate_required_ram,
    expected_full_schema,
    inspect_train_footer,
    load_feature_schema,
    preprocess_matrix,
    read_train_values,
    rank_scores,
    resource_gate,
    verify_run,
    write_exclusive,
    write_json_exclusive,
    _run_parameters,
    _validate_resume,
)


class _FakeStatistics:
    min = 0
    max = 1


class _FakeColumn:
    statistics = _FakeStatistics()


class _FakeRowGroup:
    def column(self, _index):
        return _FakeColumn()


class _FakeMetadata:
    num_rows = 12
    num_row_groups = 1

    def row_group(self, _index):
        return _FakeRowGroup()


class _FakeSchema:
    def __init__(self, names):
        self.names = names


class _FakeParquet:
    def __init__(self, names):
        self.schema_arrow = _FakeSchema(names)
        self.metadata = _FakeMetadata()


class _FakeArrayColumn:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, zero_copy_only=False):
        del zero_copy_only
        return self._values


class _FakeTable:
    def __init__(self, names):
        self._columns = {
            name: _FakeArrayColumn(np.arange(4, dtype=np.float32) + index)
            for index, name in enumerate(names[:-1])
        }
        self._columns[LABEL_COLUMN] = _FakeArrayColumn([0, 1, 0, 1])

    def column(self, name):
        return self._columns[name]


class Stage4MIUnitTests(unittest.TestCase):
    def setUp(self):
        self.features = [f"feature_{index:03d}" for index in range(EXPECTED_FEATURE_COUNT)]

    def test_preprocessing_replaces_non_finite_values_without_row_drop_or_label_change(self):
        matrix = np.zeros((3, EXPECTED_FEATURE_COUNT), dtype=np.float64)
        matrix[0, 0] = np.nan
        matrix[1, 1] = np.inf
        matrix[2, 2] = -np.inf
        labels = np.array([0, 1, 0])
        processed, processed_labels = preprocess_matrix(matrix, labels)
        self.assertEqual(processed.shape, (3, EXPECTED_FEATURE_COUNT))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.isfinite(processed).all())
        np.testing.assert_array_equal(processed_labels, labels)

    def test_invalid_feature_count_and_non_binary_labels_are_rejected(self):
        with self.assertRaises(Stage4ContractError):
            preprocess_matrix(np.zeros((2, 379)), np.array([0, 1]))
        with self.assertRaises(Stage4ContractError):
            preprocess_matrix(np.zeros((3, EXPECTED_FEATURE_COUNT)), np.array([0, 1, 2]))

    def test_ranking_is_descending_and_ties_use_original_feature_order(self):
        scores = {index: 0.0 for index in range(EXPECTED_FEATURE_COUNT)}
        scores[9] = 2.0
        scores[2] = 2.0
        ranking = rank_scores(scores, self.features)
        self.assertEqual(ranking[0]["feature_index"], 2)
        self.assertEqual(ranking[1]["feature_index"], 9)
        self.assertEqual(len(ranking[:100]), 100)

    def test_schema_and_provenance_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)
            (config / "feature_schema.json").write_text(
                json.dumps({"feature_names": self.features}), encoding="utf-8"
            )
            self.assertEqual(load_feature_schema(config), self.features)
        with self.assertRaises(Stage4ContractError):
            expected_full_schema(self.features[:-1])
        with self.assertRaises(Stage4ContractError):
            expected_full_schema(["source_file"] + self.features[1:])

    def test_schema_falls_back_to_train_footer_without_reading_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)
            train_path = config / "train_windows.parquet"
            train_path.write_bytes(b"metadata-fixture")
            names = load_feature_schema(
                config,
                lambda _path: _FakeParquet(expected_full_schema(self.features)),
            )
            self.assertEqual(names, self.features)

    def test_footer_uses_schema_and_label_statistics_without_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_windows.parquet"
            path.write_bytes(b"metadata-fixture")
            result = inspect_train_footer(
                path,
                self.features,
                lambda _path: _FakeParquet(expected_full_schema(self.features)),
            )
            self.assertEqual(result["rows"], 12)
            self.assertTrue(result["label_statistics"][0]["has_statistics"])

    def test_footer_rejects_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_windows.parquet"
            path.write_bytes(b"metadata-fixture")
            with self.assertRaises(Stage4ContractError):
                inspect_train_footer(
                    path,
                    self.features,
                    lambda _path: _FakeParquet(self.features + ["extra"]),
                )

    def test_footer_rejects_missing_train_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Stage4ContractError):
                inspect_train_footer(Path(directory) / "missing.parquet", self.features)

    def test_train_value_reader_receives_only_train_path_and_requested_columns(self):
        calls = []

        def reader(path, columns):
            calls.append((path, columns))
            return _FakeTable(columns)

        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train_windows.parquet"
            matrix, labels = read_train_values(train_path, self.features, table_reader=reader)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], train_path)
        self.assertEqual(calls[0][1], self.features + [LABEL_COLUMN])
        self.assertEqual(matrix.shape, (4, EXPECTED_FEATURE_COUNT))
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))

    def test_resource_gate_blocks_insufficient_ram_without_sampling(self):
        result = resource_gate(
            {"memory": {"available_bytes": 1}, "disk": {"free_bytes": 10 * 1024**3}},
            rows=1000,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["no_sampling_or_retry"])
        self.assertGreater(estimate_required_ram(1000), 1)

    def test_exclusive_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_exclusive(path, "first")
            with self.assertRaises(FileExistsError):
                write_exclusive(path, "second")
            self.assertEqual(path.read_text(encoding="utf-8"), "first")

    def test_resume_checkpoint_identity_and_parameters_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity = {"identity_digest": "same"}
            write_json_exclusive(
                run_dir / "RUN_INCOMPLETE.json",
                {"source_identity": identity, "parameters": _run_parameters()},
            )
            _validate_resume(run_dir, identity, _run_parameters())
            with self.assertRaises(Stage4ContractError):
                _validate_resume(run_dir, {"identity_digest": "different"}, _run_parameters())

    def test_verify_requires_complete_artifacts_and_does_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = verify_run(run_dir)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["missing"])


if __name__ == "__main__":
    unittest.main()

