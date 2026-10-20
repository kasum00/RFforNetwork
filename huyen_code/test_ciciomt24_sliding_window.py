import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ciciomt24_sliding_window as sw


class SlidingWindowUnitTests(unittest.TestCase):
    def test_schema_has_only_76_behavior_features_and_380_stats(self):
        self.assertEqual(len(sw.FEATURE_COLUMNS), 76)
        self.assertEqual(len(sw.OUTPUT_FEATURE_COLUMNS), 380)
        forbidden = {
            "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
            "Timestamp", "Label", "Source_CSV", "Source_CSV_ID",
            "Original_Row_Index", "Split", "Proxy_Label", "Proxy_Label_Source",
        }
        self.assertFalse(forbidden.intersection(sw.FEATURE_COLUMNS))
        self.assertEqual(sw.OUTPUT_FEATURE_COLUMNS, sw.output_feature_names())

    def test_window_count_matches_original_range_formula(self):
        for fw, fs in sw.WINDOW_CONFIGS:
            for n in (0, fw - 1, fw, fw + 1, fw + fs * 3 + 2):
                self.assertEqual(sw.window_count(n, fw, fs), len(range(0, n - fw + 1, fs)))

    def test_batch_stats_match_original_numpy_scipy_semantics(self):
        rng = np.random.default_rng(20261019)
        windows = rng.normal(size=(5, 5, 37)).astype(np.float64)
        windows[0, 0, 0] = np.nan
        windows[0, 0, 1] = np.inf
        windows[0, 0, 2] = -np.inf
        windows[1, 1, :] = 4.25  # constant -> SciPy NaN skew/kurtosis
        windows[2, 2, :] = 0.0
        windows[3, 3, 3] = np.nan
        windows[4, 4, :] = 8.008  # non-binary decimal constant exposes axis-reduction rounding
        actual = sw.compute_window_stats_batch(windows)
        expected = np.vstack([sw.original_reference_features(window.T) for window in windows])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))

    def test_window_label_uses_existential_attack_rule(self):
        flags = np.array([False, False, True, False, False, False])
        starts = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(sw.existential_window_labels(flags, starts, 3), [1, 1, 1, 0])

    def test_streaming_chunk_carry_preserves_counts_and_window_ids(self):
        rng = np.random.default_rng(7)
        n_rows = 650
        features = rng.normal(size=(n_rows, len(sw.FEATURE_COLUMNS)))
        row_ids = np.arange(n_rows, dtype=np.int64) * 2 + 7
        timestamps = np.array([f"2023-09-13 21:{i // 60:02d}:{i % 60:02d}" for i in range(n_rows)], dtype=object)
        attack = np.ones(n_rows, dtype=np.bool_)
        segment = sw.Segment(
            source_csv_id=9,
            source_csv=r"D:\raw\sample.csv",
            scenario="unit-test",
            split="train",
            flow_count=n_rows,
            first_timestamp=str(timestamps[0]),
            last_timestamp=str(timestamps[-1]),
            min_original_row_index=int(row_ids.min()),
            max_original_row_index=int(row_ids.max()),
            output_relative_path=r"train\unit-test\sid009__sample.csv",
            proxy_label="Attack",
            proxy_label_source="SCENARIO_PROXY",
        )
        next_starts = {(fw, fs): 0 for fw, fs in sw.WINDOW_CONFIGS}
        next_indices = {(fw, fs): 0 for fw, fs in sw.WINDOW_CONFIGS}
        actual = {f"FW{fw}_FS{fs}": 0 for fw, fs in sw.WINDOW_CONFIGS}

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            # The second chunk retains only the 511-row carry needed by FW=512.
            for lo, hi in ((0, 600), (600, n_rows)):
                seen = hi
                carry_lo = max(0, lo - 511)
                buffer_features = features[carry_lo:hi]
                buffer_rows = row_ids[carry_lo:hi]
                buffer_ts = timestamps[carry_lo:hi]
                buffer_attack = attack[carry_lo:hi]
                counts = sw._write_batches_for_buffer(
                    buffer_features=buffer_features,
                    buffer_rows=buffer_rows,
                    buffer_timestamps=buffer_ts,
                    buffer_attack=buffer_attack,
                    buffer_start=carry_lo,
                    seen_rows=seen,
                    next_starts=next_starts,
                    next_window_indices=next_indices,
                    segment=segment,
                    partial_root=out_root,
                    batch_windows=13,
                )
                for key, count in counts.items():
                    actual[key] += count

            for fw, fs in sw.WINDOW_CONFIGS:
                key = f"FW{fw}_FS{fs}"
                self.assertEqual(actual[key], sw.window_count(n_rows, fw, fs))
                output_dir = out_root / f"FW_{fw}_FS_{fs}" / "train" / "unit-test"
                output_files = list(output_dir.glob("*.csv"))
                self.assertEqual(len(output_files), 1)
                output = pd.read_csv(output_files[0])
                self.assertEqual(len(output), actual[key])
                self.assertFalse(output["Window_ID"].duplicated().any())
                self.assertEqual(output["Window_Index"].tolist(), list(range(actual[key])))
                self.assertEqual(len(output.columns), len(sw.OUTPUT_METADATA_COLUMNS) + 380)
                self.assertTrue(output["Source_CSV_ID"].astype(str).str.startswith("sid").all())
                self.assertTrue(output["Original_Row_Index_Start"].astype(str).str.startswith("row:").all())
                self.assertTrue((output["Label_Origin"] == "SCENARIO_PROXY").all())
                self.assertTrue((output["Num_Flows"] == fw).all())
                # IDs and source-row endpoints identify one source segment only.
                self.assertEqual(output["Split"].unique().tolist(), ["train"])
                self.assertEqual(output["Scenario"].unique().tolist(), ["unit-test"])
                self.assertTrue(np.all(output["Label"].to_numpy() == 1))

    def test_streaming_file_reader_validates_and_keeps_chunk_boundaries(self):
        rng = np.random.default_rng(19)
        n_rows = 650
        source_path = r"D:\fixture\capture.csv"
        row_ids = np.arange(n_rows, dtype=np.int64) + 100
        timestamps = pd.date_range("2023-09-13 21:00:00", periods=n_rows, freq="s").strftime("%d/%m/%Y %I:%M:%S %p")
        data = {name: rng.normal(size=n_rows) for name in sw.FEATURE_COLUMNS}
        data.update({
            "Flow ID": [f"flow-{i}" for i in range(n_rows)],
            "Src IP": ["192.0.2.1"] * n_rows,
            "Src Port": np.full(n_rows, 12345),
            "Dst IP": ["198.51.100.2"] * n_rows,
            "Dst Port": np.full(n_rows, 443),
            "Protocol": np.full(n_rows, 6),
            "Timestamp": timestamps,
            "Label": ["No Label"] * n_rows,
            "Source_CSV": [source_path] * n_rows,
            "Source_CSV_ID": np.full(n_rows, 4),
            "Original_Row_Index": row_ids,
            "Split": ["train"] * n_rows,
            "Proxy_Label": ["Attack"] * n_rows,
            "Proxy_Label_Source": ["SCENARIO_PROXY"] * n_rows,
        })
        frame = pd.DataFrame(data, columns=sw.EXPECTED_SOURCE_HEADER)
        segment = sw.Segment(
            source_csv_id=4,
            source_csv=source_path,
            scenario="fixture-attack",
            split="train",
            flow_count=n_rows,
            first_timestamp=timestamps[0],
            last_timestamp=timestamps[-1],
            min_original_row_index=int(row_ids.min()),
            max_original_row_index=int(row_ids.max()),
            output_relative_path=r"train\fixture-attack\sid004__fixture.csv",
            proxy_label="Attack",
            proxy_label_source="SCENARIO_PROXY",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            out = Path(tmp) / "out"
            relative = Path(segment.output_relative_path)
            input_csv = root / relative
            input_csv.parent.mkdir(parents=True)
            frame.to_csv(input_csv, index=False)
            reference_results = []
            nonfinite_counts = {}
            progress = []
            result = sw.process_segment_streaming(
                root, segment, out, chunksize=600, batch_windows=13,
                reference_results=reference_results,
                nonfinite_counts=nonfinite_counts,
                progress_callback=lambda rows, counts: progress.append((rows, dict(counts))),
            )
            for fw, fs in sw.WINDOW_CONFIGS:
                key = f"FW{fw}_FS{fs}"
                self.assertEqual(result[key], sw.window_count(n_rows, fw, fs))
                generated = list((out / f"FW_{fw}_FS_{fs}" / "train" / "fixture-attack").glob("*.csv"))
                self.assertEqual(len(generated), 1)
                output = pd.read_csv(generated[0])
                self.assertEqual(len(output), result[key])
                self.assertEqual(len(output.columns), 394)
                self.assertEqual(output["Original_Row_Index_Start"].iloc[0], "row:100")
                self.assertTrue((output["Label"] == 1).all())
            self.assertEqual(len(reference_results), 8)
            self.assertTrue(all(item["passed"] for item in reference_results))
            self.assertEqual(progress[-1][0], n_rows)
            self.assertTrue(nonfinite_counts)
            expected = {f"FW{fw}_FS{fs}": {split: 0 for split in sw.SPLITS} for fw, fs in sw.WINDOW_CONFIGS}
            for fw, fs in sw.WINDOW_CONFIGS:
                expected[f"FW{fw}_FS{fs}"]["train"] = result[f"FW{fw}_FS{fs}"]
            readback = sw.validate_full_outputs(out, [segment], expected)
            self.assertEqual(readback["status"], "PASS")
            self.assertEqual(readback["verified_rows"], sum(result.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
