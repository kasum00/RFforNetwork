"""Bounded synthetic tests; no population scan, windows or model training."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ciciomt24_split as split

HERE = Path(__file__).resolve().parent
DUMMY = HERE / "_not_created" / "CSV" / "train" / "Benign_train.pcap_Flow.csv"


def raw_frame(times, durations=None):
    n = len(times)
    return pd.DataFrame({
        "Flow ID": ["192.168.1.1-192.168.1.2-50000-443-6"] * n,
        "Src IP": ["192.168.1.1"] * n, "Src Port": ["50000"] * n,
        "Dst IP": ["192.168.1.2"] * n, "Dst Port": ["443"] * n,
        "Protocol": ["6"] * n,
        "Timestamp": [pd.Timestamp(t).strftime("%d/%m/%Y %I:%M:%S %p") for t in times],
        "Flow Duration": [str(v) for v in (durations if durations is not None else [1000] * n)],
        "Label": ["No Label"] * n,
    })


def manifest(times, durations=None, scenario="Benign", source_id=0, path=DUMMY):
    return split.assign_b2(split.metadata_from_raw(
        raw_frame(times, durations), path, source_id, scenario
    ))


def all_blocks():
    return (
        ["2023-09-14 11:54:58"] * 5
        + ["2023-09-14 12:00:00"] * 511
        + ["2023-09-14 12:15:27"] * 5
        + ["2023-09-14 18:00:00"] * 511
        + ["2023-09-14 18:13:37"] * 5
    )


class IdentityAndMetadataTests(unittest.TestCase):
    def test_five_tuple_is_bidirectional(self):
        a = split.connection_key("192.168.1.1", 50000, "192.168.1.2", 443, 6)
        b = split.connection_key("192.168.1.2", 443, "192.168.1.1", 50000, 6)
        self.assertEqual(a, b)

    def test_flow_id_can_use_opposite_direction(self):
        raw = raw_frame(["2023-09-14 11:00:00"])
        raw.loc[0, ["Src IP", "Dst IP", "Src Port", "Dst Port"]] = [
            "192.168.1.2", "192.168.1.1", "443", "50000",
        ]
        result = split.metadata_from_raw(raw, DUMMY, 0, "Benign")
        self.assertTrue(result["Flow_ID_Consistent_With_5Tuple"].all())

    def test_unknown_scenario_is_not_silently_attack(self):
        with self.assertRaises(split.SplitError):
            split.scenario_from_path(Path("CSV/train/Unknown_train.pcap_Flow.csv"))

    def test_numeric_suffix_is_not_capture_id(self):
        p = Path("CSV/train/TCP_IP-DDoS-SYN4_train.pcap_Flow.csv")
        self.assertEqual(split.scenario_from_path(p), "TCP_IP-DDoS-SYN")
        frame = manifest(["2023-09-05 21:40:00"], scenario="TCP_IP-DDoS-SYN")
        self.assertTrue(frame["Capture_Verification"].str.contains("NOT_VERIFIED").all())

    def test_original_row_offsets_and_labels_preserved(self):
        raw = raw_frame(["2023-09-14 13:00:00", "2023-09-14 11:00:00"])
        result = split.metadata_from_raw(raw, DUMMY, 7, "Benign", offset=3000)
        self.assertEqual(result["Original_Row_Index"].tolist(), [3000, 3001])
        self.assertEqual(result["Original_Label"].tolist(), ["No Label", "No Label"])
        self.assertEqual(raw["Label"].tolist(), ["No Label", "No Label"])
        self.assertEqual(result["Proxy_Label"].tolist(), [0, 0])

    def test_sort_is_stable_and_retains_raw_identity(self):
        result = manifest(["2023-09-14 13:00:00", "2023-09-14 11:00:00", "2023-09-14 13:00:00"])
        self.assertEqual(result["Original_Row_Index"].tolist(), [1, 0, 2])
        self.assertEqual(result["Source_Sorted_Rank"].tolist(), [0, 1, 2])

    def test_invalid_timestamp_rejected_without_dropping(self):
        raw = raw_frame(["2023-09-14 11:00:00", "2023-09-14 13:00:00"])
        raw.loc[1, "Timestamp"] = "invalid"
        with self.assertRaisesRegex(split.SplitError, "original rows.*101"):
            split.metadata_from_raw(raw, DUMMY, 0, "Benign", offset=100)
        self.assertEqual(len(raw), 2)

    def test_invalid_ports_do_not_invent_session_key(self):
        raw = raw_frame(["2023-09-14 11:00:00"])
        raw.loc[0, "Src Port"] = "999999"
        result = split.metadata_from_raw(raw, DUMMY, 0, "Benign")
        self.assertFalse(result["Connection_Key_Valid"].iloc[0])
        self.assertEqual(result["Connection_Key"].iloc[0], "")

    def test_flow_id_mismatch_is_flagged(self):
        raw = raw_frame(["2023-09-14 11:00:00"])
        raw.loc[0, "Flow ID"] = "192.168.1.8-192.168.1.2-50000-443-6"
        result = split.metadata_from_raw(raw, DUMMY, 0, "Benign")
        self.assertFalse(result["Flow_ID_Consistent_With_5Tuple"].iloc[0])


class PartitionInvariantTests(unittest.TestCase):
    def test_inclusive_frozen_cutoffs_and_both_embargos(self):
        result = manifest([
            "2023-09-14 11:54:58", "2023-09-14 11:54:59",
            "2023-09-14 12:15:27", "2023-09-14 17:52:56",
            "2023-09-14 17:52:57", "2023-09-14 18:13:37",
        ])
        self.assertEqual(result["Split"].tolist(), ["train", "embargo", "val", "val", "embargo", "test"])
        self.assertEqual(result["Is_Embargo"].sum(), 2)

    def test_same_timestamp_across_original_files_stays_one_split(self):
        a = split.metadata_from_raw(raw_frame(["2023-09-14 12:15:27"] * 3), DUMMY, 0, "Benign")
        bpath = DUMMY.parent.parent / "test" / "Benign_test.pcap_Flow.csv"
        b = split.metadata_from_raw(raw_frame(["2023-09-14 12:15:27"] * 4), bpath, 1, "Benign")
        combined = split.assign_b2(pd.concat([a, b], ignore_index=True))
        report = split.validate_manifest(combined, {0: 3, 1: 4})
        self.assertEqual(report["checks"]["timestamp_bucket_not_cut"], "PASS")
        self.assertEqual(combined["Split"].unique().tolist(), ["val"])

    def test_each_small_group_is_train_only(self):
        for scenario in split.SMALL_GROUPS:
            result = manifest(["2023-08-01", "2023-10-01"], scenario=scenario)
            self.assertEqual(result["Split"].tolist(), ["train", "train"])

    def test_complete_fixture_has_two_real_511_guards(self):
        result = manifest(all_blocks())
        report = split.validate_manifest(result, {0: len(result)}, complete_population=True)
        self.assertEqual(report["structural_status"], "PASS", report)
        self.assertEqual(report["checks"]["minimum_511_each_group_boundary"], "PASS")
        self.assertEqual(result["Is_Embargo"].sum(), 1022)

    def test_too_short_complete_guard_fails(self):
        result = manifest([
            "2023-09-14 11:00:00", "2023-09-14 12:00:00",
            "2023-09-14 13:00:00", "2023-09-14 18:00:00", "2023-09-14 19:00:00",
        ])
        report = split.validate_manifest(result, {0: 5}, complete_population=True)
        self.assertEqual(report["checks"]["minimum_511_each_group_boundary"], "FAIL")

    def test_duplicate_source_row_fails_even_when_total_matches(self):
        result = manifest(["2023-09-14 11:00:00"] * 3)
        result.loc[2, "Original_Row_Index"] = 1
        report = split.validate_manifest(result, {0: 3})
        self.assertEqual(report["checks"]["source_identity_unique"], "FAIL")
        self.assertEqual(report["checks"]["every_scanned_source_row_accounted_once"], "FAIL")

    def test_missing_source_row_fails_conservation(self):
        result = manifest(["2023-09-14 11:00:00"] * 3)
        report = split.validate_manifest(result.iloc[:2], {0: 3})
        self.assertEqual(report["checks"]["row_count_conservation"], "FAIL")

    def test_cut_timestamp_bucket_is_detected(self):
        result = manifest(["2023-09-14 11:00:00"] * 3)
        result.loc[1, "Split"] = "val"
        report = split.validate_manifest(result, {0: 3})
        self.assertEqual(report["checks"]["timestamp_bucket_not_cut"], "FAIL")
        self.assertEqual(report["checks"]["frozen_b2_membership"], "FAIL")

    def test_original_row_range_is_not_treated_as_sorted_range(self):
        result = manifest(["2023-09-14 19:00:00", "2023-09-14 11:00:00", "2023-09-14 13:00:00"])
        segments = split.build_segments(result)
        self.assertEqual(segments["Start_Sorted_Rank"].tolist(), [0, 1, 2])
        self.assertEqual(result["Original_Row_Index"].tolist(), [1, 2, 0])
        self.assertEqual(split.validate_segments(segments), [])

    def test_overlapping_segments_rejected(self):
        result = manifest(["2023-09-14 11:00:00", "2023-09-14 13:00:00", "2023-09-14 19:00:00"])
        segments = split.build_segments(result)
        segments.loc[1, "Start_Sorted_Rank"] = 0
        self.assertTrue(split.validate_segments(segments))

    def test_prefix_cannot_claim_population_guard_validation(self):
        result = manifest(["2023-09-14 11:00:00"])
        report = split.validate_manifest(result, {0: 1})
        self.assertEqual(report["checks"]["minimum_511_each_group_boundary"], "NOT_ESTABLISHED_PREFIX_SAMPLE")


class ActivityAndSessionTests(unittest.TestCase):
    def test_microseconds_and_clock_uncertainty(self):
        result = manifest(["2023-09-14 11:00:00"], [2_500_000])
        t = pd.Timestamp("2023-09-14 11:00:00")
        self.assertEqual(result["Activity_End_Estimate"].iloc[0], t + pd.Timedelta(seconds=2.5))
        self.assertEqual(result["Activity_End_Upper"].iloc[0], t + pd.Timedelta(seconds=3.5))
        self.assertEqual(result["Duration_Unit_Status"].iloc[0], "ASSUMED_NOT_VERIFIED")

    def test_declared_unit_does_not_become_verified(self):
        raw = raw_frame(["2023-09-14 11:00:00"], [2])
        result = split.metadata_from_raw(raw, DUMMY, 0, "Benign", duration_unit="seconds", duration_evidence="test assertion")
        self.assertEqual(result["Activity_End_Estimate"].iloc[0], pd.Timestamp("2023-09-14 11:00:02"))
        self.assertEqual(result["Duration_Unit_Status"].iloc[0], "EVIDENCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED")

    def test_bad_durations_are_retained_and_marked_unknown(self):
        result = manifest(["2023-09-14 11:00:00"] * 3, [-1, "NaN", "Infinity"])
        self.assertEqual(len(result), 3)
        self.assertFalse(result["Duration_Valid"].any())
        self.assertTrue(result["Activity_End_Estimate"].isna().all())

    def test_active_flow_crosses_next_split(self):
        result = manifest(
            ["2023-08-30 03:25:41"], [3_000_000],
            scenario="MQTT-DDoS-Connect_Flood",
        )
        crossing = split.activity_crossings(result)
        self.assertEqual(crossing["Boundary"].tolist(), ["train_to_val"])
        self.assertEqual(crossing["Crossing_Evidence"].iloc[0], "ESTIMATED_ACTIVITY_REACHES_NEXT_SPLIT")
        self.assertEqual(result["Split"].iloc[0], "train")  # No silent reassignment.

    def test_timestamp_uncertainty_only_is_distinguished(self):
        result = manifest(
            ["2023-08-30 03:25:41"], [1_500_000],
            scenario="MQTT-DDoS-Connect_Flood",
        )
        crossing = split.activity_crossings(result)
        self.assertEqual(crossing["Crossing_Evidence"].iloc[0], "POSSIBLE_WITH_ONE_SECOND_TIMESTAMP_UNCERTAINTY")

    def test_reused_tuple_and_flow_id_do_not_prove_session(self):
        result = manifest(["2023-09-14 11:00:00", "2023-09-14 13:00:00", "2023-09-14 19:00:00"])
        candidates = split.session_candidates([split.key_summary(result)])
        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates["Risk"].eq("KEY_REUSED_NO_ENVELOPE_OVERLAP_OBSERVED").all())
        self.assertTrue(candidates["Session_Conclusion"].eq("NOT_A_VERIFIED_SESSION_ID").all())
        self.assertTrue(split.activity_crossings(result).empty)

    def test_overlapping_activity_same_key_is_only_candidate(self):
        result = manifest(
            ["2023-08-30 03:25:41", "2023-08-30 03:25:43"],
            [3_000_000, 1000], scenario="MQTT-DDoS-Connect_Flood",
        )
        candidates = split.session_candidates([split.key_summary(result)])
        self.assertEqual(candidates["Risk"].tolist(), ["POSSIBLE_ACTIVITY_ENVELOPE_OVERLAP"])
        self.assertTrue(candidates["Envelope_Limitation"].str.contains("not proof").all())


class ArtifactAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".split_test_", dir=HERE)
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture_dataset(self):
        root = self.base / "original"
        for oldsplit, times in {
            "train": ["2023-09-14 11:00:00", "2023-09-14 12:00:00", "2023-09-14 13:00:00"],
            "test": ["2023-09-14 18:00:00", "2023-09-14 19:00:00"],
        }.items():
            folder = root / "CSV" / oldsplit
            folder.mkdir(parents=True)
            raw_frame(times).to_csv(folder / f"Benign_{oldsplit}.pcap_Flow.csv", index=False)
        return root

    def test_full_scan_is_off_before_any_source_access(self):
        out = self.base / "not_created"
        with self.assertRaisesRegex(split.SplitError, "Full scan is disabled"):
            split.run_pipeline(self.base / "absent", out, mode="plan")
        self.assertFalse(out.exists())

    def test_output_inside_original_dataset_rejected(self):
        root = self.fixture_dataset()
        with self.assertRaises(split.SplitError):
            split.run_pipeline(root, root / "outputs")

    def test_unbounded_smoke_cap_rejected(self):
        with self.assertRaises(split.SplitError):
            split.run_pipeline(self.base, self.base / "out", max_rows=1_000_000)

    def test_written_parquet_is_reloaded_and_validated(self):
        result = manifest(all_blocks())
        path = self.base / "manifest.parquet"
        split.write_parquet(result, path)
        check = split.verify_written_manifest(path, {0: len(result)}, True)
        self.assertEqual(check["structural_status"], "PASS")
        damaged = pd.read_parquet(path)
        damaged.loc[1, "Original_Row_Index"] = 0
        split.write_parquet(damaged, path)
        self.assertEqual(split.verify_written_manifest(path, {0: len(result)}, True)["structural_status"], "FAIL")

    def test_smoke_pipeline_source_immutability_and_determinism(self):
        root = self.fixture_dataset()
        files = sorted(root.glob("CSV/*/*.csv"))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        a = split.run_pipeline(root, self.base / "run_a", max_rows=100)
        b = split.run_pipeline(root, self.base / "run_b", max_rows=100)
        self.assertEqual(a["structural_status"], "PASS")
        self.assertEqual(a["full_19891102_conservation"], "NOT_RUN_NOT_CLAIMED")
        self.assertFalse(a["ready_for_flow_csv_export"])
        self.assertEqual(a["rows_scanned"], 5)
        self.assertEqual(a["actual_scanned_counts"], {"train": 1, "val": 1, "test": 1, "embargo": 2})
        self.assertEqual(a["actual_scanned_counts"], b["actual_scanned_counts"])
        ma = pd.read_parquet(self.base / "run_a" / "manifest" / "G02.parquet")
        mb = pd.read_parquet(self.base / "run_b" / "manifest" / "G02.parquet")
        pd.testing.assert_frame_equal(ma, mb)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        self.assertEqual(ma["Original_Label"].unique().tolist(), ["No Label"])
        self.assertFalse((self.base / "run_a" / "train.csv").exists())
        report = json.loads((self.base / "run_a" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["scope"], "BOUNDED_REAL_CSV_PREFIX_SMOKE")

    def test_existing_output_is_not_overwritten(self):
        root = self.fixture_dataset()
        out = self.base / "existing"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(split.SplitError, "not empty"):
            split.run_pipeline(root, out)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_csv_prefix_reader_caps_original_rows(self):
        root = self.fixture_dataset()
        source = root / "CSV" / "train" / "Benign_train.pcap_Flow.csv"
        frame, record = split.read_metadata(source, 0, "microseconds", max_rows=2)
        self.assertEqual(frame["Original_Row_Index"].tolist(), [0, 1])
        self.assertEqual(record["Scanned_Rows"], 2)
        self.assertEqual(record["Read_Scope"], "BOUNDED_PREFIX")
        self.assertEqual(record["SHA256"], "")


class ExperimentalGuardReviewTests(unittest.TestCase):
    def fixture(self):
        # Duration from 03:25:41 reaches 03:25:44; its conservative end is 45.
        return manifest(
            ["2023-08-30 03:25:40", "2023-08-30 03:25:41",
             "2023-08-30 03:25:42", "2023-08-30 03:25:43",
             "2023-08-30 03:25:45", "2023-08-30 03:25:45",
             "2023-08-30 03:25:46", "2023-08-30 03:27:55",
             "2023-08-30 03:27:56", "2023-08-30 03:27:57"],
            [0, 3_000_000, 0, 0, 0, 0, 0, 0, 0, 0],
            scenario="MQTT-DDoS-Connect_Flood",
        )

    def test_right_prefix_preserves_whole_equal_timestamp_bucket(self):
        base = self.fixture()
        result, moves = split.simulate_activity_guard(base, "right_prefix")
        bucket = result.loc[result["Timestamp"].eq(pd.Timestamp("2023-08-30 03:25:45"))]
        self.assertEqual(len(bucket), 2)
        self.assertTrue(bucket["Is_Embargo"].all())
        self.assertEqual(moves["Proposed_Right_Start"].iloc[0], pd.Timestamp("2023-08-30 03:25:46"))
        self.assertTrue(split.observed_activity_crossings(result).empty)

    def test_left_suffix_removes_offending_bucket_and_later_not_earlier(self):
        base = self.fixture()
        result, _ = split.simulate_activity_guard(base, "left_suffix")
        self.assertEqual(result.loc[result["Original_Row_Index"].eq(0), "Split"].iloc[0], "train")
        self.assertEqual(result.loc[result["Original_Row_Index"].eq(1), "Split"].iloc[0], "embargo")
        self.assertTrue(split.observed_activity_crossings(result).empty)

    def test_simulation_does_not_mutate_baseline_or_frozen_policy(self):
        base = self.fixture()
        saved = base.copy(deep=True)
        clocks = dict(split.B2_CUTOFFS)
        split.simulate_activity_guard(base, "right_prefix")
        pd.testing.assert_frame_equal(base, saved)
        self.assertEqual(clocks, split.B2_CUTOFFS)

    def test_candidate_conserves_source_identity_metadata_and_segments(self):
        base = self.fixture()
        result, _ = split.simulate_activity_guard(base, "right_prefix")
        checks = split.validate_guard_simulation(base, result)
        self.assertNotIn("FAIL", checks.values())
        self.assertEqual(checks["capture_session_independence"], "NOT_ESTABLISHED")
        self.assertEqual(len(base), len(result))
        self.assertFalse(split.validate_segments(split.build_segments(result)))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(split.SplitError):
            split.simulate_activity_guard(self.fixture(), "invented")

    def test_invalid_duration_not_hidden_as_safe_guard(self):
        base = self.fixture()
        base.loc[0, "Duration_Valid"] = False
        with self.assertRaisesRegex(split.SplitError, "Invalid duration"):
            split.simulate_activity_guard(base, "right_prefix")

    def test_small_group_stays_train_only(self):
        base = manifest(["2023-09-12 22:30:00"], [120_000_000], scenario="Recon-Ping_Sweep")
        result, moves = split.simulate_activity_guard(base, "right_prefix")
        self.assertEqual(result["Split"].tolist(), ["train"])
        self.assertTrue(moves.empty)

    def test_validation_rejects_source_metadata_change(self):
        base = self.fixture()
        damaged = base.copy()
        damaged.loc[0, "Flow_ID"] = "altered"
        self.assertEqual(split.validate_guard_simulation(base, damaged)["source_metadata_preserved"], "FAIL")

    def test_validation_rejects_reallocation_to_another_retained_split(self):
        base = self.fixture()
        damaged = base.copy()
        damaged.loc[0, "Split"] = "test"
        self.assertEqual(split.validate_guard_simulation(base, damaged)["only_retained_to_embargo_no_reallocation"], "FAIL")

    def test_missing_later_sample_not_a_session_independence_claim(self):
        base = manifest(["2023-09-14 11:00:00"])
        result, _ = split.simulate_activity_guard(base, "right_prefix")
        checks = split.validate_guard_simulation(base, result)
        self.assertEqual(checks["capture_session_independence"], "NOT_ESTABLISHED")
        self.assertEqual(checks["minimum_511_each_group_boundary"], "NOT_ESTABLISHED_PREFIX_SAMPLE")

    def test_pairwise_uncertainty_is_not_estimated_overlap(self):
        base = manifest(
            ["2023-08-30 03:25:41", "2023-08-30 03:25:43"],
            [1_500_000, 1000], scenario="MQTT-DDoS-Connect_Flood",
        )
        result = split.pairwise_key_interval_audit(base)
        self.assertEqual(result["estimated_overlap_pairs"], 0)
        self.assertEqual(result["possible_overlap_pairs"], 1)

    def test_pair_cap_does_not_report_a_healthy_zero(self):
        result = split.pairwise_key_interval_audit(self.fixture(), cap=0)
        self.assertIn("NOT_ESTABLISHED", result["status"])
        self.assertIsNone(result["possible_overlap_pairs"])

    def test_review_rejects_population_input_before_source_access(self):
        with tempfile.TemporaryDirectory(prefix=".review_test_", dir=HERE) as folder:
            root = Path(folder)
            (root / "report.json").write_text(json.dumps({"scope": "FULL_POPULATION_MANIFEST"}), encoding="utf-8")
            with self.assertRaisesRegex(split.SplitError, "bounded smoke artifacts only"):
                split.run_duration_review(root, root / "out", root / "absent_dataset")
            self.assertFalse((root / "out").exists())

    def test_review_rejects_over_limit_prefix_before_source_access(self):
        with tempfile.TemporaryDirectory(prefix=".review_test_", dir=HERE) as folder:
            root = Path(folder)
            (root / "report.json").write_text(json.dumps({"scope": "BOUNDED_REAL_CSV_PREFIX_SMOKE"}), encoding="utf-8")
            pd.DataFrame({"Scanned_Rows": [3001]}).to_csv(root / "source_inventory.csv", index=False)
            with self.assertRaisesRegex(split.SplitError, "<=3000 rows each"):
                split.run_duration_review(root, root / "out", root / "absent_dataset")
            self.assertFalse((root / "out").exists())


class PopulationMetadataAuditTests(unittest.TestCase):
    def setUp(self):
        import duckdb
        self.con = duckdb.connect()
        self.con.execute("SET threads=1")
        self.con.execute("SET preserve_insertion_order=true")
        split.population_schema(self.con)
        self.tmp = tempfile.TemporaryDirectory(prefix=".population_test_",dir=HERE)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def source(self, raw, scenario="MQTT-DDoS-Connect_Flood", sid=0):
        path=self.root/"CSV"/"train"/f"{scenario}{sid or ''}_train.pcap_Flow.csv"
        path.parent.mkdir(parents=True,exist_ok=True)
        raw.to_csv(path,index=False)
        split.ingest_population_source(self.con,path,sid,raw.columns.tolist())
        return path

    def test_full_metadata_authorization_is_required_before_source_access(self):
        with self.assertRaisesRegex(split.SplitError,"authorize-full-metadata"):
            split.run_full_metadata_audit(self.root/"absent",self.root/"out")
        self.assertFalse((self.root/"out").exists())

    def test_streamed_original_rows_not_timestamp_ranks(self):
        raw=raw_frame(["2023-08-30 03:27:57","2023-08-30 03:25:41","2023-08-30 03:25:43"])
        self.source(raw)
        rows=self.con.execute("SELECT rid,ts FROM flows ORDER BY ts,rid").fetchall()
        self.assertEqual([r[0] for r in rows],[1,2,0])

    def test_duckdb_canonical_key_matches_original_helper(self):
        raw=raw_frame(["2023-08-30 03:25:41"])
        raw.loc[0,["Src IP","Src Port","Dst IP","Dst Port"]]=["192.168.1.2","443","192.168.1.1","50000"]
        self.source(raw)
        actual=self.con.execute("SELECT ck,flow_id_consistent FROM flows").fetchone()
        self.assertEqual(actual[0],split.connection_key("192.168.1.1",50000,"192.168.1.2",443,6))
        self.assertTrue(actual[1])

    def test_duration_microseconds_and_one_second_upper(self):
        self.source(raw_frame(["2023-08-30 03:25:41"],[1_500_000]))
        row=self.con.execute("SELECT ts,end_estimate,end_upper FROM activity").fetchone()
        self.assertEqual((row[1]-row[0]).total_seconds(),1.5)
        self.assertEqual((row[2]-row[1]).total_seconds(),1)

    def test_negative_duration_is_retained_and_flagged(self):
        self.source(raw_frame(["2023-08-30 03:25:41"],[-1]))
        n,valid,issue=self.con.execute("SELECT count(*),count(duration_us),max(metadata_issue) FROM flows").fetchone()
        self.assertEqual(n,1)
        self.assertEqual(valid,0)
        self.assertIsNotNone(issue)

    def test_zero_duration_is_retained(self):
        self.source(raw_frame(["2023-08-30 03:25:41"],[0]))
        self.assertEqual(self.con.execute("SELECT duration_us,metadata_issue FROM flows").fetchone(),(0,None))

    def test_invalid_timestamp_not_silently_dropped(self):
        raw=raw_frame(["2023-08-30 03:25:41"])
        raw.loc[0,"Timestamp"]="bad-clock"
        self.source(raw)
        self.assertEqual(self.con.execute("SELECT count(*) FROM flows WHERE ts IS NULL AND metadata_issue IS NOT NULL").fetchone()[0],1)

    def test_all_original_fields_participate_in_fingerprint(self):
        raw=raw_frame(["2023-08-30 03:25:41"]*2)
        raw["extra_feature"]=["1","2"]
        self.source(raw)
        self.assertEqual(self.con.execute("SELECT count(DISTINCT record_sha) FROM flows").fetchone()[0],2)

    def test_identical_records_have_distinct_source_rows_and_exact_verification(self):
        self.source(raw_frame(["2023-08-30 03:25:41"]*2))
        result=split.verify_full_record_duplicates(self.con)
        self.assertEqual(result["exact_duplicate_groups"],1)
        self.assertEqual(result["exact_duplicate_extra_copies"],1)
        self.assertEqual(self.con.execute("SELECT count(DISTINCT rid) FROM flows").fetchone()[0],2)

    def test_population_B_C_match_bounded_simulator_and_preserve_A(self):
        times=["2023-08-30 03:25:40","2023-08-30 03:25:41","2023-08-30 03:25:42",
               "2023-08-30 03:25:43","2023-08-30 03:25:45","2023-08-30 03:25:46",
               "2023-08-30 03:27:55","2023-08-30 03:27:57"]
        durations=[0,3_000_000,0,0,0,0,0,0]
        raw=raw_frame(times,durations)
        path=self.source(raw)
        baseline=split.assign_b2(split.metadata_from_raw(raw,path,0,"MQTT-DDoS-Connect_Flood"))
        split.populate_policies(self.con)
        for design,strategy in [("A",None),("B","right_prefix"),("C","left_suffix")]:
            split.activate_population_design(self.con,design)
            actual=dict(self.con.execute("SELECT rid,split FROM proposed ORDER BY rid").fetchall())
            frame=baseline if strategy is None else split.simulate_activity_guard(baseline,strategy)[0]
            expected=dict(zip(frame.Original_Row_Index,frame.Split))
            self.assertEqual(actual,expected)
            if design!="A":
                self.assertFalse(split.population_crossing_counts(self.con))

    def test_window_formula_matches_boundary_examples(self):
        for n,fw,fs,expected in [(255,256,8,0),(256,256,8,1),(263,256,8,1),(264,256,8,2),(511,512,16,0),(528,512,16,2)]:
            self.assertEqual(split.window_count_only(n,fw,fs),expected)

    def test_cplus_candidate_selection_honors_ratio_and_retention(self):
        candidates=[(600,200,200),(700,100,200),(590,190,220),(500,200,100)]
        selected=split._best_ratio_candidate(candidates,minimum_retained=1000)
        self.assertIsNotNone(selected)
        index,counts,_=selected
        self.assertEqual(index,0)
        self.assertEqual(counts.tolist(),[600,200,200])
        self.assertEqual(split._ratio_objective(counts)[0],0)

    def test_window_counts_never_join_short_sources(self):
        self.assertEqual(split.window_count_only(200,256,8)+split.window_count_only(200,256,8),0)

    def test_frozen_cplus_membership_keeps_inclusive_cutoffs_and_embargoes_bad_metadata(self):
        times=["2023-08-30 03:25:41","2023-08-30 03:25:42","2023-08-30 03:25:43",
               "2023-08-30 03:27:55","2023-08-30 03:27:56","2023-08-30 03:27:57"]
        self.source(raw_frame(times), scenario="MQTT-DDoS-Connect_Flood", sid=0)
        self.con.execute("""CREATE TEMP TABLE cplus_policies(gid UTINYINT,train_end TIMESTAMP,
          val_start TIMESTAMP,val_end TIMESTAMP,test_start TIMESTAMP)""")
        self.con.execute("""INSERT INTO cplus_policies VALUES
          (3,'2023-08-30 03:25:41','2023-08-30 03:25:43','2023-08-30 03:27:55','2023-08-30 03:27:57')""")
        split.create_cplus_membership_view(self.con, "flows")
        actual=self.con.execute("SELECT rid,split FROM cplus_membership ORDER BY rid").fetchall()
        self.assertEqual(actual,[(0,"train"),(1,"embargo"),(2,"val"),
                                 (3,"val"),(4,"embargo"),(5,"test")])

    def test_full_export_segment_sorts_timestamp_stably_and_roundtrips_fields(self):
        raw=raw_frame(["2023-08-30 03:25:43","2023-08-30 03:25:41","2023-08-30 03:25:41"])
        path=self.source(raw, scenario="MQTT-DDoS-Connect_Flood", sid=0)
        self.con.execute("""CREATE TEMP TABLE cplus_membership AS
          SELECT 0::SMALLINT sid,range::BIGINT rid,3::UTINYINT gid,'train'::VARCHAR split
          FROM range(3)""")
        output=self.root/"sorted_segment.csv"
        columns=raw.columns.tolist()
        split.export_cplus_source_segment(self.con,0,path,columns,"train",output)
        result=split._append_output_for_audit(self.con,output,0,path,"train",columns,3,"Attack")
        self.assertEqual(result,{"rows":3,"sort_inversions":0,"bad_provenance_or_labels":0})
        self.assertEqual(self.con.execute("SELECT rid FROM export_audit ORDER BY ts,rid").fetchall(),[(1,),(2,),(0,)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
