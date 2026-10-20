"""B2 source-row manifests and leakage audits; no windows, models or flow export.

Default smoke scans are bounded prefixes. A population manifest requires the
explicit --authorize-full-scan option. Frozen B2 cutoffs remain provisional:
duration/session findings are reported, never silently purged or reallocated.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import csv
import os
import shutil
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_TOTAL = 19_891_102
EXPECTED_FILES = 71
EXPECTED_CPLUS = {"train": 10_798_603, "val": 3_599_535,
                  "test": 3_599_534, "embargo": 1_893_430}
MIN_EMBARGO = 511
LABEL_ORIGIN = "SCENARIO_PROXY_NOT_PER_FLOW_GROUND_TRUTH"
DEFAULT_ROOT = Path(r"D:\COMPACT-MI-RFE\CICIoMT24")
GROUP_NAMES = (
    "ARP_Spoofing", "Benign", "MQTT-DDoS-Connect_Flood",
    "MQTT-DDoS-Publish_Flood", "MQTT-DoS-Connect_Flood",
    "MQTT-DoS-Publish_Flood", "MQTT-Malformed_Data", "Recon-OS_Scan",
    "Recon-Ping_Sweep", "Recon-Port_Scan", "Recon-VulScan",
    "TCP_IP-DDoS-ICMP", "TCP_IP-DDoS-SYN", "TCP_IP-DDoS-TCP",
    "TCP_IP-DDoS-UDP", "TCP_IP-DoS-ICMP", "TCP_IP-DoS-SYN",
    "TCP_IP-DoS-TCP", "TCP_IP-DoS-UDP",
)
GROUP_IDS = {name: f"G{i:02d}" for i, name in enumerate(GROUP_NAMES, 1)}
SMALL_GROUPS = frozenset({
    "ARP_Spoofing", "MQTT-DoS-Publish_Flood", "MQTT-Malformed_Data",
    "Recon-Ping_Sweep", "TCP_IP-DDoS-ICMP", "TCP_IP-DoS-ICMP",
})
# Train end, validation start/end, test start: inclusive start-time cutoffs.
# All clocks are source-naive; the CSVs do not establish a timezone.
B2_CUTOFFS = {
    "Benign": ("2023-09-14 11:54:58", "2023-09-14 12:15:27", "2023-09-14 17:52:56", "2023-09-14 18:13:37"),
    "MQTT-DDoS-Connect_Flood": ("2023-08-30 03:25:41", "2023-08-30 03:25:43", "2023-08-30 03:27:55", "2023-08-30 03:27:57"),
    "MQTT-DDoS-Publish_Flood": ("2023-08-30 03:02:35", "2023-08-30 03:02:39", "2023-08-30 03:04:19", "2023-08-30 03:04:24"),
    "MQTT-DoS-Connect_Flood": ("2023-08-29 01:29:22", "2023-08-29 01:29:30", "2023-08-29 01:32:47", "2023-08-29 01:32:54"),
    "Recon-OS_Scan": ("2023-09-07 23:31:48", "2023-09-07 23:31:51", "2023-09-07 23:33:49", "2023-09-07 23:33:57"),
    "Recon-Port_Scan": ("2023-09-07 23:03:02", "2023-09-07 23:03:05", "2023-09-07 23:03:31", "2023-09-07 23:03:33"),
    "Recon-VulScan": ("2023-09-12 22:32:09", "2023-09-12 22:32:11", "2023-09-12 22:34:00", "2023-09-12 22:35:16"),
    "TCP_IP-DDoS-SYN": ("2023-09-05 22:14:30", "2023-09-05 22:14:32", "2023-09-05 22:41:31", "2023-09-05 22:41:36"),
    "TCP_IP-DDoS-TCP": ("2023-09-05 20:34:31", "2023-09-05 20:34:33", "2023-09-05 21:00:38", "2023-09-05 21:00:40"),
    "TCP_IP-DDoS-UDP": ("2023-09-06 00:28:57", "2023-09-06 00:29:07", "2023-09-06 01:06:42", "2023-09-06 01:06:52"),
    "TCP_IP-DoS-SYN": ("2023-08-31 21:46:19", "2023-08-31 21:46:21", "2023-08-31 22:00:15", "2023-08-31 22:00:17"),
    "TCP_IP-DoS-TCP": ("2023-08-31 00:49:09", "2023-08-31 00:49:14", "2023-08-31 00:57:43", "2023-08-31 00:57:45"),
    "TCP_IP-DoS-UDP": ("2023-09-01 02:09:32", "2023-09-01 02:10:16", "2023-09-01 21:14:49", "2023-09-01 21:15:13"),
}
EXPECTED_B2 = {"train": 11_843_431, "val": 3_947_810, "test": 3_947_817, "embargo": 152_044}
REQUIRED = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp", "Flow Duration", "Label",
]
SPLITS = ("train", "val", "test", "embargo")
SMOKE_STEMS = (
    "Benign", "MQTT-DDoS-Connect_Flood", "ARP_Spoofing",
    "Recon-VulScan", "Recon-Ping_Sweep", "MQTT-Malformed_Data",
)
UNIT_CODES = {"microseconds": "us", "milliseconds": "ms", "seconds": "s", "nanoseconds": "ns"}


class SplitError(ValueError):
    """A failed invariant: do not publish this manifest as an approved split."""


def scenario_from_path(path: Path) -> str:
    suffix = f"_{path.parent.name}.pcap_Flow.csv"
    if path.parent.name not in {"train", "test"} or not path.name.endswith(suffix):
        raise SplitError(f"Unexpected source name/layout: {path}")
    scenario = path.name[:-len(suffix)].rstrip("0123456789")
    if scenario not in GROUP_IDS:
        raise SplitError(f"Unknown scenario; explicit mapping required: {path}")
    return scenario


@lru_cache(maxsize=65536)
def normalize_ip(value: str) -> str:
    return str(ipaddress.ip_address(value.strip()))


def connection_key(src_ip, src_port, dst_ip, dst_port, protocol) -> str:
    a = (normalize_ip(str(src_ip)), int(src_port))
    b = (normalize_ip(str(dst_ip)), int(dst_port))
    proto = int(protocol)
    if not (0 <= a[1] <= 65535 and 0 <= b[1] <= 65535 and 0 <= proto <= 255):
        raise ValueError("Invalid port/protocol")
    a, b = sorted((a, b))
    return f"{proto}|{a[0]}:{a[1]}|{b[0]}:{b[1]}"


def flow_id_key(flow_id: str) -> str | None:
    try:
        a, b, p, q, proto = str(flow_id).rsplit("-", 4)
        return connection_key(a, p, b, q, proto)
    except (ValueError, TypeError):
        return None


def metadata_from_raw(
    raw: pd.DataFrame, source: Path, source_id: int, scenario: str,
    offset: int = 0, duration_unit: str = "microseconds",
    duration_evidence: str = "",
) -> pd.DataFrame:
    """Preserve source identities; estimate activity only under declared units."""
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise SplitError(f"Missing columns in {source}: {sorted(missing)}")
    if duration_unit not in UNIT_CODES or scenario not in GROUP_IDS:
        raise SplitError("Unknown duration unit or scenario")
    raw = raw.reset_index(drop=True)
    ts = pd.to_datetime(raw["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if ts.isna().any():
        bad = (np.flatnonzero(ts.isna()) + offset)[:10].tolist()
        raise SplitError(f"Invalid Timestamp in {source}, original rows {bad}; nothing dropped")
    duration = pd.to_numeric(raw["Flow Duration"], errors="coerce")
    valid_duration = np.isfinite(duration) & (duration >= 0)
    delta = pd.to_timedelta(duration.where(valid_duration), unit=UNIT_CODES[duration_unit], errors="coerce")
    # Avoid int64 timestamp overflow, retaining bad-duration rows for review.
    ns_start = ts.astype("int64").to_numpy()
    ns_delta = delta.astype("int64").to_numpy()
    valid_duration = valid_duration & delta.notna()
    can_add = valid_duration.to_numpy() & (ns_delta <= np.iinfo(np.int64).max - ns_start - 1_000_000_000)
    end_ns = np.full(len(raw), pd.NaT.value, dtype=np.int64)
    end_ns[can_add] = ns_start[can_add] + ns_delta[can_add]
    end = pd.Series(pd.to_datetime(end_ns))
    keys, consistent = [], []
    for flow_id, a, p, b, q, proto in raw[["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]].itertuples(index=False, name=None):
        try:
            key = connection_key(a, p, b, q, proto)
        except (ValueError, TypeError):
            key = ""
        keys.append(key)
        consistent.append(bool(key) and flow_id_key(flow_id) == key)
    n = len(raw)
    frame = pd.DataFrame({
        "Source_ID": np.full(n, source_id, dtype=np.int32),
        "Source_CSV": str(source.resolve()),
        "Original_Split": source.parent.name,
        "Original_Row_Index": np.arange(offset, offset + n, dtype=np.int64),
        "Scenario": scenario,
        "Capture_Group": GROUP_IDS[scenario],
        "Capture_Verification": "INFERRED_DEPENDENCY_GROUP_NOT_VERIFIED_CAPTURE",
        "Timestamp_Raw": raw["Timestamp"].astype(str),
        "Timestamp": ts,
        "Timestamp_Timezone": "UNSPECIFIED_IN_SOURCE",
        "Original_Label": raw["Label"].astype(str),
        "Proxy_Label": np.full(n, int(scenario != "Benign"), dtype=np.int8),
        "Label_Origin": LABEL_ORIGIN,
        "Flow_ID": raw["Flow ID"].astype(str),
        "Connection_Key": pd.Series(keys, dtype="string[pyarrow]"),
        "Connection_Key_Valid": np.array([bool(k) for k in keys]),
        "Flow_ID_Consistent_With_5Tuple": consistent,
        "Flow_Duration_Raw": raw["Flow Duration"].astype(str),
        "Duration_Unit": duration_unit,
        "Duration_Unit_Status": "EVIDENCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED" if duration_evidence else "ASSUMED_NOT_VERIFIED",
        "Duration_Valid": can_add,
        "Activity_End_Estimate": end,
        # The source clock is rounded/truncated to seconds: conservative uncertainty.
        "Activity_End_Upper": end + pd.Timedelta(seconds=1),
    })
    return frame


def assign_b2(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign start-time buckets to frozen B2 blocks. Never purge session risks."""
    out = frame.sort_values(
        ["Capture_Group", "Timestamp", "Source_CSV", "Original_Row_Index"],
        kind="stable",
    ).reset_index(drop=True).copy()
    if out["Timestamp"].isna().any():
        raise SplitError("Missing Timestamp cannot be assigned")
    out["Split"] = "embargo"
    out["Embargo_Reason"] = ""
    for scenario, ids in out.groupby("Scenario", sort=False, observed=True).groups.items():
        if scenario in SMALL_GROUPS:
            out.loc[ids, "Split"] = "train"
            continue
        if scenario not in B2_CUTOFFS:
            raise SplitError(f"No approved B2 cutoffs for {scenario}")
        a, b, c, d = map(pd.Timestamp, B2_CUTOFFS[scenario])
        if not a < b <= c < d:
            raise SplitError("Invalid frozen B2 cutoff order")
        t = out.loc[ids, "Timestamp"]
        out.loc[t.index[t <= a], "Split"] = "train"
        out.loc[t.index[(t >= b) & (t <= c)], "Split"] = "val"
        out.loc[t.index[t >= d], "Split"] = "test"
        out.loc[t.index[(t > a) & (t < b)], "Embargo_Reason"] = "TRAIN_VAL_GUARD_WHOLE_TIMESTAMP"
        out.loc[t.index[(t > c) & (t < d)], "Embargo_Reason"] = "VAL_TEST_GUARD_WHOLE_TIMESTAMP"
    out["Is_Embargo"] = out["Split"].eq("embargo")
    out["Boundary_Status"] = "PROVISIONAL_PENDING_POPULATION_ACTIVITY_AND_SESSION_REVIEW"
    out["Group_Sorted_Rank"] = out.groupby("Capture_Group", sort=False).cumcount()
    out["Source_Sorted_Rank"] = out.groupby("Source_ID", sort=False).cumcount()
    # Ranks refer to sorted positions, not a contiguous interval of raw CSV rows.
    return out


def build_segments(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for source_id, part in manifest.groupby("Source_ID", sort=True):
        part = part.sort_values("Source_Sorted_Rank", kind="stable")
        run = part["Split"].ne(part["Split"].shift()).cumsum()
        for number, (_, seg) in enumerate(part.groupby(run, sort=False)):
            records.append({
                "Segment_ID": f"S{int(source_id):05d}_{number:03d}",
                "Source_ID": int(source_id), "Source_CSV": seg["Source_CSV"].iloc[0],
                "Capture_Group": seg["Capture_Group"].iloc[0], "Split": seg["Split"].iloc[0],
                "Start_Sorted_Rank": int(seg["Source_Sorted_Rank"].min()),
                "End_Sorted_Rank_Exclusive": int(seg["Source_Sorted_Rank"].max()) + 1,
                "Rows": len(seg), "Timestamp_Min": seg["Timestamp"].min(),
                "Timestamp_Max": seg["Timestamp"].max(),
            })
    return pd.DataFrame(records)


def validate_segments(segments: pd.DataFrame) -> list[str]:
    errors = []
    for source, part in segments.groupby("Source_ID", sort=False):
        part = part.sort_values("Start_Sorted_Rank")
        previous = 0
        for row in part.itertuples(index=False):
            if row.Start_Sorted_Rank != previous or row.End_Sorted_Rank_Exclusive <= row.Start_Sorted_Rank:
                errors.append(f"Overlapping/gapped/invalid segments for Source_ID={source}")
            if row.Rows != row.End_Sorted_Rank_Exclusive - row.Start_Sorted_Rank:
                errors.append(f"Segment length mismatch for Source_ID={source}")
            previous = row.End_Sorted_Rank_Exclusive
    return errors


def validate_manifest(
    manifest: pd.DataFrame, expected_by_source: dict[int, int],
    complete_population: bool = False,
) -> dict:
    checks, errors = {}, []
    wanted = int(sum(expected_by_source.values()))
    checks["row_count_conservation"] = "PASS" if len(manifest) == wanted else "FAIL"
    checks["source_identity_unique"] = "PASS" if not manifest.duplicated(["Source_ID", "Original_Row_Index"]).any() and not manifest.duplicated(["Source_CSV", "Original_Row_Index"]).any() else "FAIL"
    identity_ok = set(manifest["Source_ID"].unique()) == set(expected_by_source)
    for source, count in expected_by_source.items():
        part = manifest.loc[manifest["Source_ID"].eq(source)]
        raw_rows = np.sort(part["Original_Row_Index"].to_numpy(dtype=np.int64))
        ranks = np.sort(part["Source_Sorted_Rank"].to_numpy(dtype=np.int64))
        identity_ok &= np.array_equal(raw_rows, np.arange(count)) and np.array_equal(ranks, np.arange(count))
        identity_ok &= part["Source_CSV"].nunique() == 1
        identity_ok &= part["Capture_Group"].nunique() == 1
    checks["every_scanned_source_row_accounted_once"] = "PASS" if identity_ok else "FAIL"
    mapping_ok = manifest.groupby("Source_CSV", observed=True)["Source_ID"].nunique().le(1).all()
    checks["source_path_id_mapping"] = "PASS" if mapping_ok else "FAIL"
    checks["timestamp_bucket_not_cut"] = "PASS" if manifest.groupby(["Capture_Group", "Timestamp"], observed=True)["Split"].nunique().le(1).all() else "FAIL"
    segments = build_segments(manifest)
    errors.extend(validate_segments(segments))
    checks["segments_nonoverlapping_and_contiguous"] = "PASS" if not errors else "FAIL"
    checks["embargo_flag_matches_split"] = "PASS" if manifest["Is_Embargo"].eq(manifest["Split"].eq("embargo")).all() else "FAIL"
    checks["split_values_valid"] = "PASS" if manifest["Split"].isin(SPLITS).all() else "FAIL"
    checks["proxy_mapping_and_origin"] = "PASS" if manifest["Proxy_Label"].eq(manifest["Scenario"].ne("Benign").astype(int)).all() and manifest["Label_Origin"].eq(LABEL_ORIGIN).all() else "FAIL"
    valid_b2 = assign_b2(manifest)
    a = manifest.set_index(["Source_ID", "Original_Row_Index"])["Split"].sort_index()
    b = valid_b2.set_index(["Source_ID", "Original_Row_Index"])["Split"].sort_index()
    checks["frozen_b2_membership"] = "PASS" if a.equals(b) else "FAIL"
    small = manifest.loc[manifest["Scenario"].isin(SMALL_GROUPS)]
    checks["six_small_groups_train_only"] = "PASS" if small["Split"].eq("train").all() else "FAIL"
    if manifest["Scenario"].isin(SMALL_GROUPS).all():
        checks["minimum_511_each_group_boundary"] = "NOT_APPLICABLE_TRAIN_ONLY"
    elif complete_population:
        guards_ok = True
        for scenario, part in manifest.groupby("Scenario", observed=True):
            if scenario in SMALL_GROUPS:
                continue
            guards_ok &= all(
                part["Embargo_Reason"].eq(reason).sum() >= MIN_EMBARGO
                for reason in ("TRAIN_VAL_GUARD_WHOLE_TIMESTAMP", "VAL_TEST_GUARD_WHOLE_TIMESTAMP")
            )
            guards_ok &= all(part["Split"].eq(s).any() for s in ("train", "val", "test"))
        checks["minimum_511_each_group_boundary"] = "PASS" if guards_ok else "FAIL"
    else:
        checks["minimum_511_each_group_boundary"] = "NOT_ESTABLISHED_PREFIX_SAMPLE"
    errors.extend(name for name, status in checks.items() if status == "FAIL")
    return {"structural_status": "FAIL" if errors else "PASS", "checks": checks, "errors": errors, "rows": len(manifest)}


def activity_crossings(manifest: pd.DataFrame) -> pd.DataFrame:
    """A single flow's activity may reach a later split, independent of key reuse."""
    frames = []
    for scenario, part in manifest.groupby("Scenario", observed=True):
        if scenario in SMALL_GROUPS:
            continue
        _, val_start, _, test_start = map(pd.Timestamp, B2_CUTOFFS[scenario])
        for source_split, target, boundary in (
            ("train", "val", val_start), ("train", "test", test_start), ("val", "test", test_start)
        ):
            selected = part.loc[
                part["Split"].eq(source_split)
                & part["Duration_Valid"]
                & part["Activity_End_Upper"].ge(boundary)
            ].copy()
            if selected.empty:
                continue
            selected["Boundary"] = f"{source_split}_to_{target}"
            selected["Next_Split_Start"] = boundary
            selected["Crossing_Evidence"] = np.where(
                selected["Activity_End_Estimate"].ge(boundary),
                "ESTIMATED_ACTIVITY_REACHES_NEXT_SPLIT",
                "POSSIBLE_WITH_ONE_SECOND_TIMESTAMP_UNCERTAINTY",
            )
            selected["Session_Conclusion"] = "POTENTIAL_RISK_NOT_VERIFIED_SESSION"
            frames.append(selected)
    columns = list(manifest.columns) + ["Boundary", "Next_Split_Start", "Crossing_Evidence", "Session_Conclusion"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def key_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    part = manifest.loc[~manifest["Is_Embargo"] & manifest["Connection_Key_Valid"]]
    return part.groupby(["Connection_Key", "Split"], observed=True, sort=True).agg(
        Start_Min=("Timestamp", "min"), Start_Max=("Timestamp", "max"),
        End_Estimate_Max=("Activity_End_Estimate", "max"),
        End_Upper_Max=("Activity_End_Upper", "max"),
        Rows=("Original_Row_Index", "size"),
        Flow_ID_Example=("Flow_ID", "min"),
        Source_CSV_Example=("Source_CSV", "min"),
    ).reset_index()


def session_candidates(summaries: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Shared keys are candidates only; envelopes can span unrelated sessions."""
    summaries = [p for p in summaries if not p.empty]
    cols = ["Connection_Key", "Left_Split", "Right_Split", "Risk", "Session_Conclusion"]
    if not summaries:
        return pd.DataFrame(columns=cols)
    merged = pd.concat(summaries, ignore_index=True)
    merged = merged.groupby(["Connection_Key", "Split"], observed=True, sort=True).agg(
        Start_Min=("Start_Min", "min"), Start_Max=("Start_Max", "max"),
        End_Estimate_Max=("End_Estimate_Max", "max"),
        End_Upper_Max=("End_Upper_Max", "max"), Rows=("Rows", "sum"),
        Flow_ID_Example=("Flow_ID_Example", "min"),
        Source_CSV_Example=("Source_CSV_Example", "min"),
    ).reset_index()
    findings = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        l = merged.loc[merged["Split"].eq(left)].drop(columns="Split")
        r = merged.loc[merged["Split"].eq(right)].drop(columns="Split")
        common = l.merge(r, on="Connection_Key", suffixes=("_Left", "_Right"))
        if common.empty:
            continue
        overlap = (
            common["Start_Min_Left"].le(common["End_Upper_Max_Right"])
            & common["Start_Min_Right"].le(common["End_Upper_Max_Left"])
        )
        common["Left_Split"], common["Right_Split"] = left, right
        common["Risk"] = np.where(overlap, "POSSIBLE_ACTIVITY_ENVELOPE_OVERLAP", "KEY_REUSED_NO_ENVELOPE_OVERLAP_OBSERVED")
        common["Session_Conclusion"] = "NOT_A_VERIFIED_SESSION_ID"
        common["Envelope_Limitation"] = "Min/max envelopes may bridge idle periods; not proof of pairwise flow overlap"
        findings.append(common)
    if not findings:
        return pd.DataFrame(columns=cols)
    result = pd.concat(findings, ignore_index=True)
    # Make the short human preview show overlap risks before harmless key reuse.
    result["_priority"] = result["Risk"].ne("POSSIBLE_ACTIVITY_ENVELOPE_OVERLAP").astype(int)
    return result.sort_values(["_priority", "Left_Split", "Right_Split", "Connection_Key"], kind="stable").drop(columns="_priority").reset_index(drop=True)


def file_signature(path: Path, full_hash: bool = False) -> dict:
    stat = path.stat()
    value = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if full_hash:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        value["sha256"] = h.hexdigest()
    return value


def read_metadata(
    source: Path, source_id: int, duration_unit: str,
    duration_evidence: str = "", max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    scenario = scenario_from_path(source)
    before = file_signature(source, full_hash=max_rows is None)
    frames, offset = [], 0
    for raw in pd.read_csv(
        source, usecols=REQUIRED, dtype=str, keep_default_na=False,
        nrows=max_rows, chunksize=100_000,
    ):
        frames.append(metadata_from_raw(raw, source, source_id, scenario, offset, duration_unit, duration_evidence))
        offset += len(raw)
    if not frames:
        raise SplitError(f"Empty source requires explicit review: {source}")
    after = file_signature(source)
    if any(before[k] != after[k] for k in ("bytes", "mtime_ns")):
        raise SplitError(f"Source changed during reading: {source}")
    record = {
        "Source_ID": source_id, "Source_CSV": str(source.resolve()),
        "Scenario": scenario, "Capture_Group": GROUP_IDS[scenario],
        "Original_Split": source.parent.name, "Scanned_Rows": offset,
        "Read_Scope": "BOUNDED_PREFIX" if max_rows is not None else "FULL_SOURCE",
        "Byte_Size": before["bytes"], "Mtime_NS": before["mtime_ns"],
        "SHA256": before.get("sha256", ""), "Source_Stat_Unchanged": True,
    }
    return pd.concat(frames, ignore_index=True), record


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)


def verify_written_manifest(path: Path, expected: dict[int, int], complete: bool) -> dict:
    # Reload the persisted artifact, not merely the in-memory frame.
    return validate_manifest(pd.read_parquet(path), expected, complete)


def _output_folder(out: Path, dataset: Path) -> Path:
    out, dataset = out.resolve(), dataset.resolve()
    code = Path(__file__).resolve().parent
    if out == dataset or dataset in out.parents or out in dataset.parents:
        raise SplitError("Output must be separate from the original dataset")
    if out == code or code not in out.parents:
        raise SplitError(f"Output must be a dedicated subdirectory of {code}")
    if out.exists() and any(out.iterdir()):
        raise SplitError(f"Output is not empty; choose a new run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(
    dataset: Path, out: Path, mode: str = "smoke", max_rows: int = 3000,
    duration_unit: str = "microseconds", duration_evidence: str = "",
    authorize_full_scan: bool = False,
) -> dict:
    if mode not in {"smoke", "plan"}:
        raise SplitError("Only smoke and manifest-plan modes exist; no flow CSV export")
    if mode == "plan" and not authorize_full_scan:
        raise SplitError("Full scan is disabled; requires explicit --authorize-full-scan")
    if mode == "smoke" and not 1 <= max_rows <= 10_000:
        raise SplitError("Smoke cap must be 1..10000 rows per file")
    dataset = dataset.resolve()
    all_paths = sorted((dataset / "CSV").glob("*/*.csv"))
    if not all_paths:
        raise SplitError(f"No source CSVs in {dataset / 'CSV'}")
    named = [(i, p, scenario_from_path(p)) for i, p in enumerate(all_paths)]
    if mode == "plan" and len(named) != EXPECTED_FILES:
        raise SplitError(f"Population file count differs: {len(named)} != {EXPECTED_FILES}")
    if mode == "smoke":
        named = [(i, p, s) for i, p, s in named if s in SMOKE_STEMS]
    out = _output_folder(out, dataset)
    manifest_dir = out / "manifest"
    manifest_dir.mkdir()
    inventory, validation, segments_all, key_parts = [], [], [], []
    counts = {s: 0 for s in SPLITS}
    label_counts = {s: {"benign": 0, "attack": 0} for s in SPLITS}
    invalid_duration = invalid_keys = inconsistent_ids = crossing_count = crossing_unique = 0
    original_labels = set()
    crossing_writer = None
    preview, crossing_preview = [], []
    complete = mode == "plan"
    try:
        for scenario in GROUP_NAMES:
            selected = [(i, p) for i, p, s in named if s == scenario]
            if not selected:
                continue
            parts, expected = [], {}
            for source_id, path in selected:
                meta, record = read_metadata(path, source_id, duration_unit, duration_evidence, None if complete else max_rows)
                inventory.append(record)
                expected[source_id] = len(meta)
                parts.append(meta)
                print(f"[{mode}] {path.name}: {len(meta):,} source rows", flush=True)
            manifest = assign_b2(pd.concat(parts, ignore_index=True))
            del parts
            manifest["Manifest_Scope"] = "FULL_POPULATION_PROVISIONAL" if complete else "BOUNDED_PREFIX_ONLY"
            path = manifest_dir / f"{GROUP_IDS[scenario]}.parquet"
            # Provisional manifest is persisted BEFORE validation/final audit.
            write_parquet(manifest, path)
            check = verify_written_manifest(path, expected, complete)
            validation.append({"scenario": scenario, **check})
            segments_all.append(build_segments(manifest))
            key_parts.append(key_summary(manifest))
            crossings = activity_crossings(manifest)
            if not crossings.empty:
                table = pa.Table.from_pandas(crossings, preserve_index=False)
                if crossing_writer is None:
                    crossing_writer = pq.ParquetWriter(out / "activity_crossings.parquet", table.schema, compression="zstd")
                crossing_writer.write_table(table)
                crossing_count += len(crossings)
                crossing_unique += len(crossings.drop_duplicates(["Source_ID", "Original_Row_Index"]))
                crossing_preview.append(crossings.groupby("Boundary", sort=False).head(5))
            invalid_duration += int((~manifest["Duration_Valid"]).sum())
            invalid_keys += int((~manifest["Connection_Key_Valid"]).sum())
            inconsistent_ids += int((~manifest["Flow_ID_Consistent_With_5Tuple"]).sum())
            original_labels.update(manifest["Original_Label"].unique())
            for split in SPLITS:
                part = manifest.loc[manifest["Split"].eq(split)]
                counts[split] += len(part)
                label_counts[split]["benign"] += int(part["Proxy_Label"].eq(0).sum())
                label_counts[split]["attack"] += int(part["Proxy_Label"].eq(1).sum())
            if sum(len(x) for x in preview) < 100:
                preview.append(manifest.head(20))
            del manifest
        if crossing_writer is not None:
            crossing_writer.close()
            crossing_writer = None
        candidates = session_candidates(key_parts)
        write_parquet(candidates, out / "session_candidates.parquet")
        candidates.head(100).to_csv(out / "session_candidates_preview.csv", index=False)
        if crossing_preview:
            pd.concat(crossing_preview, ignore_index=True).head(100).to_csv(out / "activity_crossings_preview.csv", index=False)
        pd.concat(segments_all, ignore_index=True).to_csv(out / "segments.csv", index=False)
        pd.concat(preview, ignore_index=True).head(100).to_csv(out / "manifest_preview.csv", index=False)
        pd.DataFrame(inventory).to_csv(out / "source_inventory.csv", index=False)
        # Recheck every source after ALL groups, not just immediately after reading.
        for record in inventory:
            current = file_signature(Path(record["Source_CSV"]))
            record["Source_Stat_Unchanged"] &= current["bytes"] == record["Byte_Size"] and current["mtime_ns"] == record["Mtime_NS"]
        pd.DataFrame(inventory).to_csv(out / "source_inventory.csv", index=False)
        scanned = sum(r["Scanned_Rows"] for r in inventory)
        conservation = scanned == sum(counts.values())
        full_total_ok = scanned == EXPECTED_TOTAL if complete else None
        frozen_counts_ok = counts == EXPECTED_B2 if complete else None
        statuses = [c["structural_status"] for c in validation]
        source_unchanged = all(r["Source_Stat_Unchanged"] for r in inventory)
        structural_pass = bool(statuses) and all(s == "PASS" for s in statuses) and conservation and source_unchanged
        if complete:
            structural_pass &= bool(full_total_ok and frozen_counts_ok)
        report = {
            "mode": mode, "scope": "FULL_POPULATION_MANIFEST" if complete else "BOUNDED_REAL_CSV_PREFIX_SMOKE",
            "policy": "FROZEN_B2_WITH_PROVISIONAL_ACTIVITY_SESSION_REVIEW",
            "structural_status": "PASS" if structural_pass else "FAIL",
            "manifest_status": "PROVISIONAL_NOT_APPROVED_FOR_FLOW_CSV_EXPORT",
            "sources_scanned": len(inventory), "rows_scanned": scanned,
            "sample_rows_per_file_cap": None if complete else max_rows,
            "actual_scanned_counts": counts, "actual_scanned_proxy_counts": label_counts,
            "scanned_row_conservation": "PASS" if conservation else "FAIL",
            "full_19891102_conservation": ("PASS" if full_total_ok else "FAIL") if complete else "NOT_RUN_NOT_CLAIMED",
            "full_frozen_b2_count_match": ("PASS" if frozen_counts_ok else "FAIL") if complete else "NOT_RUN_NOT_CLAIMED",
            "expected_full_b2_from_approved_stage2_NOT_REMEASURED_IN_SMOKE": EXPECTED_B2,
            "expected_full_retained": EXPECTED_TOTAL - EXPECTED_B2["embargo"],
            "duration_unit": duration_unit,
            "duration_unit_evidence_declared": duration_evidence or None,
            "duration_unit_independently_verified": False,
            "timestamp_resolution_uncertainty_seconds": 1,
            "invalid_duration_rows": invalid_duration, "invalid_connection_key_rows": invalid_keys,
            "flow_id_inconsistent_or_unparseable_rows": inconsistent_ids,
            "potential_activity_crossing_records": crossing_count,
            "potential_activity_crossing_unique_source_rows": crossing_unique,
            "activity_crossing_examples": json.loads(pd.concat(crossing_preview, ignore_index=True).groupby("Boundary", sort=False).head(1).to_json(orient="records", date_format="iso", date_unit="us")) if crossing_preview else [],
            "shared_connection_key_split_pair_candidates": len(candidates),
            "possible_activity_envelope_overlap_candidates": int(candidates["Risk"].eq("POSSIBLE_ACTIVITY_ENVELOPE_OVERLAP").sum()) if "Risk" in candidates else 0,
            "session_identity_verified": False, "session_conclusion": "UNPROVEN; shared keys and envelopes are risk indicators only",
            "duration_and_session_review_required": True,
            "ready_for_flow_csv_export": False,
            "flow_csv_export_implemented": False,
            "source_immutability_check": "SHA256_SNAPSHOT_AND_STAT" if complete else "SIZE_MTIME_ONLY_FOR_LARGE_SOURCE_PREFIXES",
            "raw_sources_unchanged_by_stat": source_unchanged,
            "original_labels_observed": sorted(original_labels),
            "quality_and_risk_checks": {
                "duration_values_finite_nonnegative_representable": "PASS" if invalid_duration == 0 else "FAIL_REVIEW_REQUIRED_ROWS_RETAINED",
                "valid_connection_keys": "PASS" if invalid_keys == 0 else "FAIL_REVIEW_REQUIRED_ROWS_RETAINED",
                "flow_id_matches_bidirectional_5tuple": "PASS" if inconsistent_ids == 0 else "REVIEW_REQUIRED",
                "original_source_size_mtime_unchanged": "PASS" if source_unchanged else "FAIL",
                "duration_unit_independently_verified": "NOT_ESTABLISHED",
                "capture_independence": "NOT_ESTABLISHED",
                "activity_session_boundary_safety": "REVIEW_REQUIRED" if crossing_count or len(candidates) or invalid_duration else "NOT_ESTABLISHED",
            },
            "six_small_groups_train_only": sorted(SMALL_GROUPS),
            "coverage_warning": "Six small groups are Train-only; Recon-Ping_Sweep has 114 total flows and cannot form FW>=256.",
            "capture_status": "INFERRED_DEPENDENCY_GROUPS; NOT_CAPTURE_DISJOINT",
            "duplicate_full_raw_record_scan": "NOT_RERUN_IN_THIS_STAGE; stage2 fingerprints are not a session proof",
            "checks_by_group": validation,
            "limitations": [
                "No Label is preserved; binary labels are scenario proxies.",
                "A prefix sample cannot validate full population counts, all embargo sizes, or all sessions.",
                "Duration units need exporter evidence; all activity estimates are conditional on the declared unit.",
                "One-second timestamp uncertainty does not reconstruct packet timing.",
                "Min/max key envelopes can span idle periods and unrelated reuse of the same tuple.",
                "Frozen B2 guard boundaries are not silently changed when activity risks are found.",
                "No sliding windows or classifiers are executed and no source flow CSV is overwritten.",
            ],
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report_markdown(report, out / "report.md")
        print(json.dumps({k: report[k] for k in ("scope", "structural_status", "rows_scanned", "actual_scanned_counts", "potential_activity_crossing_records", "shared_connection_key_split_pair_candidates", "full_19891102_conservation", "ready_for_flow_csv_export")}, ensure_ascii=False, indent=2), flush=True)
        if not structural_pass:
            raise SplitError(f"Manifest validation failed; inspect {out / 'report.json'}")
        return report
    finally:
        if crossing_writer is not None:
            crossing_writer.close()


def write_report_markdown(report: dict, path: Path) -> None:
    rows = [
        "# Báo cáo bộ chia CICIoMT2024 — Giai đoạn 3",
        "",
        f"Phạm vi: {report['scope']}. Kiểm định cấu trúc: {report['structural_status']}.",
        "Chỉ xuất manifest và audit. Chưa xuất CSV flow, tạo sliding window hoặc huấn luyện.",
        "",
        f"Số nguồn đã đọc: {report['sources_scanned']}; số dòng đã đọc: {report['rows_scanned']:,}.",
        f"Bảo toàn 19.891.102 dòng toàn bộ: {report['full_19891102_conservation']}.",
        "",
        "| Tập | Dòng thực đọc | Flow toàn bộ dự kiến từ B2 |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        rows.append(f"| {split} | {report['actual_scanned_counts'][split]:,} | {EXPECTED_B2[split]:,} |")
    rows += [
        "",
        "Các số toàn bộ trong bảng là thiết kế Giai đoạn 2, chưa được smoke test đo lại.",
        f"Đơn vị Flow Duration: {report['duration_unit']}; chưa được xác minh độc lập.",
        "Activity_End_Estimate = Timestamp + Flow Duration theo đơn vị đã khai báo.",
        "Activity_End_Upper cộng thêm 1 giây vì timestamp nguồn chỉ tới giây.",
        f"Bản ghi hoạt động có thể đi qua ranh giới: {report['potential_activity_crossing_records']:,}.",
        f"Ứng viên khóa kết nối chung giữa cặp tập: {report['shared_connection_key_split_pair_candidates']:,}.",
        "Trùng 5-tuple, Flow ID hoặc min/max envelope không chứng minh cùng session.",
        "",
        "Ranh giới được giữ PROVISIONAL; cần xem xét mở rộng embargo hoặc chấp nhận rủi ro có giải thích.",
        "Module chưa có chức năng xuất CSV flow. Kiểm định toàn bộ và phê duyệt vẫn là điều kiện trước bước đó.",
        "",
        "## Kiểm định theo nhóm",
        "",
        "| Nhóm kịch bản | Kiểm tra | Trạng thái |",
        "|---|---|---|",
    ]
    for group in report["checks_by_group"]:
        for name, status in group["checks"].items():
            rows.append(f"| {group['scenario']} | {name} | {status} |")
    rows += ["", "## Chất lượng và rủi ro", "", "| Kiểm tra | Trạng thái |", "|---|---|"]
    for name, status in report["quality_and_risk_checks"].items():
        rows.append(f"| {name} | {status} |")
    rows += ["", "## Ví dụ flow có thể đi qua ranh giới", ""]
    for example in report["activity_crossing_examples"]:
        rows += [
            f"- Nguồn: {example['Source_CSV']}; Original_Row_Index={example['Original_Row_Index']}.",
            f"  Timestamp={example['Timestamp']}, Duration raw={example['Flow_Duration_Raw']} ({example['Duration_Unit']}, chưa xác minh).",
            f"  Activity end ước tính={example['Activity_End_Estimate']}; tập kế tiếp bắt đầu={example['Next_Split_Start']}.",
            f"  Ranh giới={example['Boundary']}; đây là rủi ro, chưa xác minh session.",
        ]
    rows += ["", "## Hạn chế", ""] + [f"- {item}" for item in report["limitations"]]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def simulate_activity_guard(manifest: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bounded, experimental ONLY. Never changes B2_CUTOFFS or run_pipeline.

    right_prefix delays the later block; left_suffix shortens the earlier block.
    bucket_purge is a diagnostic comparator: it can fragment retained blocks.
    All removals are whole scenario/timestamp buckets and remain in the ledger.
    """
    if strategy not in {"right_prefix", "left_suffix", "bucket_purge"}:
        raise SplitError("Unknown experimental guard strategy")
    if not 0 < len(manifest) <= 120_000:
        raise SplitError("Experimental review is limited to 120000 cached sample rows")
    if (~manifest.loc[~manifest["Is_Embargo"], "Duration_Valid"]).any():
        raise SplitError("Invalid duration: cannot certify experimental activity guard")
    out = manifest.copy(deep=True)
    changes = []
    for scenario in out["Scenario"].unique():
        if scenario in SMALL_GROUPS:
            continue
        for left, right, clock in (
            ("train", "val", B2_CUTOFFS[scenario][1]),
            ("val", "test", B2_CUTOFFS[scenario][3]),
        ):
            boundary = pd.Timestamp(clock)
            same = out["Scenario"].eq(scenario)
            earlier = same & out["Split"].eq(left)
            bad = out.loc[earlier & out["Activity_End_Upper"].ge(boundary)]
            if bad.empty:
                continue
            guard_until = out.loc[earlier, "Activity_End_Upper"].max()
            if strategy == "right_prefix":
                mask = same & out["Split"].eq(right) & out["Timestamp"].le(guard_until)
            elif strategy == "left_suffix":
                mask = earlier & out["Timestamp"].ge(bad["Timestamp"].min())
            else:
                mask = earlier & out["Timestamp"].isin(bad["Timestamp"])
            changes.append({
                "Strategy": strategy, "Scenario": scenario,
                "Boundary": f"{left}_to_{right}", "Old_Right_Start": boundary,
                "First_Offending_Start": bad["Timestamp"].min(),
                "Earlier_Max_End_Upper": guard_until,
                "Proposed_Right_Start": guard_until.floor("s") + pd.Timedelta(seconds=1)
                if strategy == "right_prefix" else boundary,
                "Left_End_Exclusive": bad["Timestamp"].min()
                if strategy == "left_suffix" else pd.NaT,
                "Additional_Embargo_Rows_Observed": int(mask.sum()),
            })
            out.loc[mask, "Split"] = "embargo"
            out.loc[mask, "Is_Embargo"] = True
            out.loc[mask, "Embargo_Reason"] = f"EXPERIMENTAL_{strategy.upper()}_{left.upper()}_{right.upper()}"
    out["Boundary_Status"] = "EXPERIMENTAL_SAMPLE_ONLY_NOT_APPROVED"
    return out, pd.DataFrame(changes)


def observed_activity_crossings(manifest: pd.DataFrame) -> pd.DataFrame:
    """Use the actual retained target start, not the frozen B2 start after delay.

    Missing target blocks cannot establish safety. The bounded review reports
    missing coverage separately; this function only enumerates observed risks.
    """
    frames = []
    for scenario, part in manifest.groupby("Scenario", observed=True):
        if scenario in SMALL_GROUPS:
            continue
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            target = part.loc[part["Split"].eq(right), "Timestamp"]
            if target.empty:
                continue
            selected = part.loc[
                part["Split"].eq(left) & part["Duration_Valid"]
                & part["Activity_End_Upper"].ge(target.min())
            ].copy()
            if not selected.empty:
                selected["Boundary"] = f"{left}_to_{right}"
                selected["Next_Split_Start"] = target.min()
                frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_guard_simulation(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Audit the candidate without misrepresenting it as frozen B2 membership."""
    expected = baseline.groupby("Source_ID").size().to_dict()
    checks = validate_manifest(candidate, expected, False)["checks"]
    checks.pop("frozen_b2_membership")  # Candidate changes are deliberately NOT B2 approval.
    identity = ["Source_ID", "Original_Row_Index"]
    a = baseline.set_index(identity).sort_index()
    b = candidate.set_index(identity).sort_index()
    immutable = [c for c in baseline.columns if c not in {
        "Source_ID", "Original_Row_Index", "Split", "Is_Embargo", "Embargo_Reason", "Boundary_Status",
    }]
    same_source = a.index.equals(b.index) and a[immutable].equals(b[immutable])
    checks["source_metadata_preserved"] = "PASS" if same_source else "FAIL"
    if a.index.equals(b.index):
        allowed = b["Split"].eq(a["Split"]) | (a["Split"].ne("embargo") & b["Split"].eq("embargo"))
        checks["only_retained_to_embargo_no_reallocation"] = "PASS" if allowed.all() else "FAIL"
    else:
        checks["only_retained_to_embargo_no_reallocation"] = "FAIL"
    contiguous = True
    for _, part in candidate.groupby("Scenario", observed=True):
        part = part.sort_values("Group_Sorted_Rank")
        for role in ("train", "val", "test"):
            positions = np.flatnonzero(part["Split"].eq(role))
            if len(positions):
                contiguous &= positions[-1] - positions[0] + 1 == len(positions)
    checks["retained_group_blocks_contiguous"] = "PASS" if contiguous else "FAIL"
    checks["no_observed_activity_crossings"] = "PASS" if observed_activity_crossings(candidate).empty else "FAIL"
    checks["capture_session_independence"] = "NOT_ESTABLISHED"
    return checks


def pairwise_key_interval_audit(manifest: pd.DataFrame, cap: int = 1_000_000) -> dict:
    """Check actual row pairs, not min/max envelopes, across ALL sample scenarios.

    A reused key is not a session identifier. An upper-bound intersection is
    only possible overlap under one-second uncertainty, not observed packets.
    """
    examined = estimated = possible = 0
    roles = {}
    valid = manifest.loc[~manifest["Is_Embargo"] & manifest["Connection_Key_Valid"]]
    for role in ("train", "val", "test"):
        roles[role] = {k: p for k, p in valid.loc[valid["Split"].eq(role)].groupby("Connection_Key", observed=True)}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in roles[left].keys() & roles[right].keys():
            a, b = roles[left][key], roles[right][key]
            count = len(a) * len(b)
            if examined + count > cap:
                return {"status": "NOT_ESTABLISHED_PAIR_CAP_REACHED", "pairs_examined": examined,
                        "estimated_overlap_pairs": None, "possible_overlap_pairs": None}
            examined += count
            for row in a.itertuples(index=False):
                estimated += int((b["Timestamp"].le(row.Activity_End_Estimate) & b["Activity_End_Estimate"].ge(row.Timestamp)).sum())
                possible += int((b["Timestamp"].le(row.Activity_End_Upper) & b["Activity_End_Upper"].ge(row.Timestamp)).sum())
    return {"status": "COMPLETED_BOUNDED_SAMPLE_NOT_SESSION_PROOF", "pairs_examined": examined,
            "estimated_overlap_pairs": estimated, "possible_overlap_pairs": possible}


def run_duration_review(input_run: Path, out: Path, dataset: Path = DEFAULT_ROOT) -> dict:
    """Read only a previous bounded smoke and the SAME <=3000-row prefixes.

    Produces audit intermediates, not split flow CSVs or an approved manifest.
    No full-scan option exists for this branch.
    """
    input_run = input_run.resolve()
    old_report = json.loads((input_run / "report.json").read_text(encoding="utf-8"))
    if old_report.get("scope") != "BOUNDED_REAL_CSV_PREFIX_SMOKE":
        raise SplitError("Duration review accepts bounded smoke artifacts only")
    inventory = pd.read_csv(input_run / "source_inventory.csv")
    if not 0 < len(inventory) <= 12 or not inventory["Scanned_Rows"].between(1, 3000).all():
        raise SplitError("Duration review cap: 12 sources, <=3000 rows each")
    paths = sorted((input_run / "manifest").glob("*.parquet"))
    if not paths or len(paths) > 19:
        raise SplitError("Missing or unexpected sample manifests")
    # Inspect Parquet footer counts BEFORE loading; a misleading report cannot bypass the cap.
    if sum(pq.ParquetFile(p).metadata.num_rows for p in paths) > 36_000:
        raise SplitError("Cached review manifest exceeds bounded input cap")
    baseline = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    expected = {int(r.Source_ID): int(r.Scanned_Rows) for r in inventory.itertuples(index=False)}
    if validate_manifest(baseline, expected)["structural_status"] != "PASS":
        raise SplitError("Input smoke manifest failed structural checks")
    if not baseline["Duration_Unit"].eq("microseconds").all():
        raise SplitError("Review requires the explicitly declared microsecond sample")
    allowed_paths = {str(p.resolve()) for p in (dataset / "CSV").glob("*/*.csv")}
    raw_parts, signatures = [], {}
    for row in inventory.itertuples(index=False):
        source = Path(row.Source_CSV).resolve()
        if str(source) not in allowed_paths or scenario_from_path(source) != row.Scenario:
            raise SplitError("Cached inventory source is outside the specified dataset")
        before = file_signature(source)
        if before != {"bytes": int(row.Byte_Size), "mtime_ns": int(row.Mtime_NS)}:
            raise SplitError("Source signature differs from the original smoke")
        raw = pd.read_csv(source, nrows=int(row.Scanned_Rows))
        raw["Source_ID"] = int(row.Source_ID)
        raw["Original_Row_Index"] = np.arange(len(raw), dtype=np.int64)
        raw_parts.append(raw)
        signatures[str(source)] = before
    raw = pd.concat(raw_parts, ignore_index=True)
    identity = ["Source_ID", "Original_Row_Index"]
    a = baseline.set_index(identity).sort_index()
    b = raw.set_index(identity).sort_index()
    if not a.index.equals(b.index):
        raise SplitError("Raw prefix identity mismatch")
    for original, cached in (("Flow ID", "Flow_ID"), ("Timestamp", "Timestamp_Raw"), ("Label", "Original_Label")):
        if not b[original].astype(str).equals(a[cached].astype(str)):
            raise SplitError(f"Raw/cached mismatch: {original}")
    if not np.array_equal(b["Flow Duration"].to_numpy(), pd.to_numeric(a["Flow_Duration_Raw"]).to_numpy()):
        raise SplitError("Raw/cached duration mismatch")
    d = raw["Flow Duration"]
    packets = raw["Tot Fwd Pkts"] + raw["Tot Bwd Pkts"]
    byte_count = raw["TotLen Fwd Pkts"] + raw["TotLen Bwd Pkts"]
    rate_checks, hypotheses = {}, []
    for rate, number in (
        ("Flow Pkts/s", packets), ("Flow Byts/s", byte_count),
        ("Fwd Pkts/s", raw["Tot Fwd Pkts"]), ("Bwd Pkts/s", raw["Tot Bwd Pkts"]),
    ):
        valid = d.gt(0) & number.gt(0) & np.isfinite(raw[rate])
        observed = raw.loc[valid, rate]
        predicted = number[valid] * 1e6 / d[valid]
        rate_checks[rate] = {
            "eligible": int(valid.sum()),
            "matching": int(np.isclose(observed, predicted, rtol=1e-5, atol=1e-8).sum()),
            "max_relative_error": float(((observed - predicted).abs() / predicted).max()),
            "median_inferred_scale": float((observed * d[valid] / number[valid]).median()),
        }
        if rate == "Flow Pkts/s":
            for unit, scale in (("seconds", 1), ("milliseconds", 1e3), ("microseconds", 1e6), ("nanoseconds", 1e9)):
                hypotheses.append({"Unit": unit, "Eligible": int(valid.sum()), "Matching": int(np.isclose(observed, number[valid] * scale / d[valid], rtol=1e-5, atol=1e-8).sum())})
    iat_valid = packets.gt(1) & np.isfinite(raw["Flow IAT Mean"])
    iat_ok = np.isclose(raw["Flow IAT Mean"] * (packets - 1), d, rtol=1e-5, atol=1e-3)
    time_checks = {}
    for feature in ("Flow IAT Max", "Fwd IAT Tot", "Bwd IAT Tot", "Active Max", "Idle Max"):
        eligible = np.isfinite(raw[feature]) & d.ge(0)
        time_checks[feature] = {"eligible": int(eligible.sum()), "not_exceeding_duration": int((raw.loc[eligible, feature] <= d[eligible] + 1e-3).sum())}
    raw_by_id = raw.set_index(identity)
    crossings = activity_crossings(baseline)
    saved = pd.read_parquet(input_run / "activity_crossings.parquet")
    keys = identity + ["Boundary"]
    if set(map(tuple, crossings[keys].to_numpy())) != set(map(tuple, saved[keys].to_numpy())):
        raise SplitError("Saved crossings do not reproduce from cached B2")
    preview = pd.read_csv(input_run / "activity_crossings_preview.csv")
    if not set(map(tuple, preview[keys].to_numpy())).issubset(set(map(tuple, crossings[keys].to_numpy()))):
        raise SplitError("Preview contains identities absent from full crossings")
    details, pair_rows = [], []
    for row in crossings.itertuples(index=False):
        target = row.Boundary.split("_to_")[1]
        matching = baseline.loc[
            baseline["Scenario"].eq(row.Scenario) & baseline["Split"].eq(target)
            & baseline["Connection_Key"].eq(row.Connection_Key)
        ]
        possible = matching["Timestamp"].le(row.Activity_End_Upper) & matching["Activity_End_Upper"].ge(row.Timestamp)
        estimate = matching["Timestamp"].le(row.Activity_End_Estimate) & matching["Activity_End_Estimate"].ge(row.Timestamp)
        source_raw = raw_by_id.loc[(row.Source_ID, row.Original_Row_Index)]
        detail = {
            "Source_ID": row.Source_ID, "Source_CSV": row.Source_CSV,
            "Original_Row_Index": row.Original_Row_Index, "Scenario": row.Scenario,
            "Start_Split": row.Split, "Next_Split": target, "Boundary": row.Boundary,
            "Timestamp": row.Timestamp, "Flow_Duration_us": int(row.Flow_Duration_Raw),
            "Flow_End_Time": row.Activity_End_Estimate, "Flow_End_Upper": row.Activity_End_Upper,
            "Next_Split_Start": row.Next_Split_Start,
            "Overrun_Estimate_s": (row.Activity_End_Estimate - row.Next_Split_Start).total_seconds(),
            "Overrun_Upper_s": (row.Activity_End_Upper - row.Next_Split_Start).total_seconds(),
            "Flow_ID": row.Flow_ID, "Connection_Key": row.Connection_Key,
            "Matching_Key_Rows": len(matching), "Matching_Flow_ID_Rows": int(matching["Flow_ID"].eq(row.Flow_ID).sum()),
            "Estimated_Overlap_Pairs": int(estimate.sum()), "Possible_Overlap_Pairs": int(possible.sum()),
            "Session_Conclusion": "SAME_KEY_AND_INTERVAL_OVERLAP_SESSION_UNPROVEN" if possible.any() else "KEY_REUSE_NO_MATCHING_INTERVAL_OVERLAP_OBSERVED" if len(matching) else "TIME_CROSSING_NO_MATCHING_KEY_OBSERVED",
        }
        for field in ("Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "SYN Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt"):
            detail[field] = source_raw[field]
        detail["Transport_Key_Interpretation"] = "TCP_UDP_CONNECTION_CANDIDATE_NOT_SESSION_ID" if int(source_raw["Protocol"]) in {6, 17} else "NON_TCP_UDP_KEY_REQUIRES_SPECIAL_REVIEW_NOT_TRANSPORT_SESSION_ID"
        details.append(detail)
        for candidate in matching.itertuples(index=False):
            peer_raw = raw_by_id.loc[(candidate.Source_ID, candidate.Original_Row_Index)]
            overlap = candidate.Timestamp <= row.Activity_End_Upper and candidate.Activity_End_Upper >= row.Timestamp
            pair_rows.append({
                "Boundary": row.Boundary, "Source_ID_Left": row.Source_ID,
                "Original_Row_Index_Left": row.Original_Row_Index,
                "Source_CSV_Left": row.Source_CSV, "Flow_ID_Left": row.Flow_ID,
                "Connection_Key": row.Connection_Key,
                "Start_Left": row.Timestamp, "End_Left_Estimate": row.Activity_End_Estimate,
                "Source_ID_Right": candidate.Source_ID, "Original_Row_Index_Right": candidate.Original_Row_Index,
                "Source_CSV_Right": candidate.Source_CSV, "Flow_ID_Right": candidate.Flow_ID,
                "Start_Right": candidate.Timestamp, "End_Right_Estimate": candidate.Activity_End_Estimate,
                "Gap_From_Left_End_s": (candidate.Timestamp - row.Activity_End_Estimate).total_seconds(),
                "Estimated_Overlap": candidate.Timestamp <= row.Activity_End_Estimate and candidate.Activity_End_Estimate >= row.Timestamp,
                "Possible_Overlap": overlap,
                "Right_SYN": peer_raw["SYN Flag Cnt"], "Right_FIN": peer_raw["FIN Flag Cnt"], "Right_RST": peer_raw["RST Flag Cnt"],
                "Session_Verified": False,
            })
    details, pairs = pd.DataFrame(details), pd.DataFrame(pair_rows)
    comparison, boundaries, ledgers, validations = [], [], [], {}
    for strategy in ("baseline", "right_prefix", "left_suffix", "bucket_purge"):
        candidate, moves = (baseline.copy(), pd.DataFrame()) if strategy == "baseline" else simulate_activity_guard(baseline, strategy)
        counts = {role: int(candidate["Split"].eq(role).sum()) for role in SPLITS}
        retained = len(candidate) - counts["embargo"]
        checks = validate_guard_simulation(baseline, candidate)
        validations[strategy] = checks
        shared = session_candidates([key_summary(candidate)])
        pairwise = pairwise_key_interval_audit(candidate)
        record = {"Strategy": strategy, **counts, "Retained": retained,
                  "Additional_Embargo": counts["embargo"] - int(baseline["Is_Embargo"].sum()),
                  "Observed_Activity_Crossings": len(observed_activity_crossings(candidate)),
                  "Shared_Key_Split_Pair_Candidates": len(shared),
                  "Possible_Envelope_Overlap_Candidates": int(shared["Risk"].eq("POSSIBLE_ACTIVITY_ENVELOPE_OVERLAP").sum()),
                  "Pairwise_Key_Interval_Audit": pairwise}
        record.update({f"{role}_pct_retained": counts[role] / retained * 100 for role in ("train", "val", "test")})
        comparison.append(record)
        if not moves.empty:
            boundaries.append(moves)
        changed = candidate.loc[candidate["Split"].ne(baseline["Split"]), identity + ["Source_CSV", "Scenario", "Timestamp", "Embargo_Reason"]].copy()
        changed["Strategy"] = strategy
        changed["Original_B2_Split"] = baseline.loc[changed.index, "Split"]
        ledgers.append(changed)
    unchanged = all(file_signature(Path(path)) == signature for path, signature in signatures.items())
    if not unchanged:
        raise SplitError("Source changed during review")
    rates_ok = all(v["eligible"] > 0 and v["eligible"] == v["matching"] for v in rate_checks.values())
    report = {
        "scope": "BOUNDED_CACHED_SMOKE_PLUS_SAME_RAW_PREFIXES", "rows": len(baseline),
        "source_files": len(inventory), "prefix_cap": 3000,
        "unit_status": "MICROSECOND_STANDARD_SOURCE_VERIFIED_SAMPLE_CORROBORATED_LOCAL_EXPORTER_BUILD_UNVERIFIED"
        if rates_ok else "STANDARD_VERIFIED_BUT_LOCAL_SAMPLE_INCONSISTENT_REVIEW_REQUIRED",
        "all_sample_endpoints_ipv4": bool(not raw["Src IP"].str.contains(":").any() and not raw["Dst IP"].str.contains(":").any()),
        "sources": [
            "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/PacketReader.java",
            "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java",
        ],
        "rate_checks": rate_checks, "unit_hypothesis_checks": hypotheses,
        "duration_zero_rows": int(d.eq(0).sum()), "duration_min_us": int(d.min()), "duration_max_us": int(d.max()),
        "iat_mean_duration": {"eligible": int(iat_valid.sum()), "matching": int(iat_ok[iat_valid].sum())},
        "other_time_feature_checks": time_checks,
        "crossing_rows": len(details), "preview_rows": len(preview),
        "crossing_summary": details.groupby("Boundary").agg(
            Rows=("Source_ID", "size"), Matching_Key_Rows=("Matching_Key_Rows", lambda v: int(v.gt(0).sum())),
            Rows_With_Estimated_Overlap=("Estimated_Overlap_Pairs", lambda v: int(v.gt(0).sum())),
            Rows_With_Possible_Overlap=("Possible_Overlap_Pairs", lambda v: int(v.gt(0).sum())),
            Overrun_Min_s=("Overrun_Estimate_s", "min"), Overrun_Max_s=("Overrun_Estimate_s", "max"),
        ).reset_index().to_dict(orient="records"),
        "same_key_pairs": len(pairs), "estimated_overlap_pairs": int(pairs["Estimated_Overlap"].sum()),
        "possible_overlap_pairs": int(pairs["Possible_Overlap"].sum()),
        "all_41_no_fin_rst_observed": bool(details["FIN Flag Cnt"].eq(0).all() and details["RST Flag Cnt"].eq(0).all()),
        "crossing_protocol_counts": {str(k): int(v) for k, v in details["Protocol"].value_counts().items()},
        "sample_scenario_coverage": {
            "baseline": baseline.groupby(["Scenario", "Split"], observed=True).size().unstack(fill_value=0).to_dict(orient="index"),
            "right_prefix": simulate_activity_guard(baseline, "right_prefix")[0].groupby(["Scenario", "Split"], observed=True).size().unstack(fill_value=0).to_dict(orient="index"),
        },
        "comparisons": comparison, "checks": validations,
        "raw_source_stats_unchanged": unchanged,
        "frozen_b2_default_changed": False, "full_split_executed": False,
        "session_independence_proven": False, "ready_for_full_split": False,
        "full_population_candidate_counts": "NOT_MEASURED_NO_EXTRAPOLATION_FROM_PREFIX",
        "expected_full_original_B2_NOT_REMEASURED": EXPECTED_B2,
    }
    out = _output_folder(out, dataset)
    details.to_csv(out / "crossing_details_all41.csv", index=False)
    pairs.to_csv(out / "matching_flow_pairs.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
    pd.concat(boundaries, ignore_index=True).to_csv(out / "proposed_boundaries_SAMPLE_ONLY.csv", index=False)
    write_parquet(pd.concat(ledgers, ignore_index=True), out / "experimental_changes_SAMPLE_ONLY.parquet")
    (out / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scope", "rows", "crossing_summary", "comparisons", "ready_for_full_split")}, ensure_ascii=False, indent=2), flush=True)
    return report


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def window_count_only(rows: int, fw: int, fs: int) -> int:
    """Count the original range(0, n-fw+1, fs); never constructs a window."""
    if rows < 0 or fw <= 0 or fs <= 0:
        raise SplitError("Invalid window count inputs")
    return max(0, (rows - fw) // fs + 1)


def population_schema(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS flows (
        sid SMALLINT, rid BIGINT, gid UTINYINT, ts TIMESTAMP, duration_us BIGINT,
        ck VARCHAR, proto SMALLINT, syn INTEGER, fin INTEGER, rst INTEGER,
        original_label VARCHAR, record_sha BLOB, metadata_issue VARCHAR,
        flow_id_consistent BOOLEAN, rate_eligible BOOLEAN, rate_matches BOOLEAN
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS sources (
        sid SMALLINT PRIMARY KEY, path VARCHAR, gid UTINYINT, scenario VARCHAR,
        rows BIGINT, bytes BIGINT, mtime_ns BIGINT, sha256 VARCHAR
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS policies (
        design VARCHAR, gid UTINYINT, train_end TIMESTAMP, val_start TIMESTAMP,
        val_end TIMESTAMP, test_start TIMESTAMP, PRIMARY KEY(design,gid)
    )""")
    con.execute("""CREATE OR REPLACE VIEW activity AS SELECT *,
        ts + to_microseconds(duration_us) AS end_estimate,
        ts + to_microseconds(duration_us) + INTERVAL 1 SECOND AS end_upper
        FROM flows""")


def ingest_population_source(con, path: Path, source_id: int, columns: list[str]) -> dict:
    """CSV scan is ordered and single-reader; only compact metadata is persisted."""
    scenario = scenario_from_path(path)
    gid = int(GROUP_IDS[scenario][1:])
    before = file_signature(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        actual_columns = next(csv.reader(stream))
    if actual_columns != columns or not set(REQUIRED).issubset(columns):
        raise SplitError(f"CSV schema mismatch: {path}")
    reader = f"read_csv({sql_literal(path)}, header=true, all_varchar=true, parallel=false, nullstr='__AUDIT_NULL_SENTINEL__', strict_mode=true, ignore_errors=false)"
    all_fields = ",".join(sql_identifier(c) for c in columns)
    number = lambda c: f"try_cast({sql_identifier(c)} AS DOUBLE)" if c in columns else "NULL::DOUBLE"
    flag = lambda c: f"coalesce(try_cast({sql_identifier(c)} AS INTEGER),0)" if c in columns else "0"
    n = f"({number('Tot Fwd Pkts')}+{number('Tot Bwd Pkts')})"
    rate = number("Flow Pkts/s")
    # Match the existing canonical key: lexicographic IP, numeric port, bidirectional.
    ck = """CASE WHEN si < di OR (si=di AND sp<=dp)
        THEN pr::VARCHAR || '|' || si || ':' || sp::VARCHAR || '|' || di || ':' || dp::VARCHAR
        ELSE pr::VARCHAR || '|' || di || ':' || dp::VARCHAR || '|' || si || ':' || sp::VARCHAR END"""
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"""INSERT INTO flows
          WITH raw AS (SELECT row_number() OVER ()-1 AS original_row, * FROM {reader}),
          typed AS (SELECT *, try_strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS clock,
             try_cast("Flow Duration" AS DOUBLE) AS d,
             trim("Src IP") AS si, trim("Dst IP") AS di,
             try_cast("Src Port" AS INTEGER) AS sp, try_cast("Dst Port" AS INTEGER) AS dp,
             try_cast("Protocol" AS INTEGER) AS pr FROM raw),
          keyed AS (SELECT *, {ck} AS connection,
             clock IS NOT NULL AND isfinite(d) AND d>=0 AND d=floor(d)
               AND d<=315576000000000 AS valid_time,
             sp BETWEEN 0 AND 65535 AND dp BETWEEN 0 AND 65535 AND pr BETWEEN 0 AND 255
               AND length(si)>0 AND length(di)>0 AS valid_key FROM typed)
          SELECT {source_id}, original_row, {gid}, clock,
             CASE WHEN valid_time THEN d::BIGINT ELSE NULL END,
             CASE WHEN valid_key THEN connection ELSE NULL END,
             pr, {flag('SYN Flag Cnt')}, {flag('FIN Flag Cnt')}, {flag('RST Flag Cnt')},
             "Label", from_hex(sha256(to_json(list_value({all_fields})))),
             CASE WHEN NOT coalesce(valid_time,false) OR NOT coalesce(valid_key,false)
               THEN to_json(list_value("Timestamp","Flow Duration","Src IP","Src Port","Dst IP","Dst Port","Protocol")) ELSE NULL END,
             "Flow ID" IN (si||'-'||di||'-'||sp::VARCHAR||'-'||dp::VARCHAR||'-'||pr::VARCHAR,
                            di||'-'||si||'-'||dp::VARCHAR||'-'||sp::VARCHAR||'-'||pr::VARCHAR),
             coalesce(d>0 AND {n}>0 AND isfinite({rate}),false),
             coalesce(abs({rate} - {n}*1000000.0/d) <= 1e-8 + 1e-5*abs({n}*1000000.0/d),false)
          FROM keyed""")
        count = int(con.execute("SELECT count(*) FROM flows WHERE sid=?", [source_id]).fetchone()[0])
        # Full byte SHA256 binds all downstream metadata decisions to this input version.
        after = file_signature(path, full_hash=True)
        if before != {k: after[k] for k in before}:
            raise SplitError(f"Source changed during audit: {path}")
        con.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?)", [source_id, str(path.resolve()), gid, scenario, count, before["bytes"], before["mtime_ns"], after["sha256"]])
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return {"source_id": source_id, "rows": count, "path": str(path)}


def populate_policies(con) -> None:
    """A=frozen B2; B=delay later prefixes; C=shorten earlier suffixes."""
    con.execute("DELETE FROM policies")
    for scenario in GROUP_NAMES:
        gid = int(GROUP_IDS[scenario][1:])
        if scenario in SMALL_GROUPS:
            for design in "ABC":
                con.execute("INSERT INTO policies VALUES (?,?,NULL,NULL,NULL,NULL)", [design, gid])
            continue
        a, b, c, d = map(pd.Timestamp, B2_CUTOFFS[scenario])
        con.execute("INSERT INTO policies VALUES ('A',?,?,?,?,?)", [gid, a.to_pydatetime(), b.to_pydatetime(), c.to_pydatetime(), d.to_pydatetime()])
        latest_train = con.execute("SELECT max(end_upper) FROM activity WHERE gid=? AND ts<=?", [gid, a.to_pydatetime()]).fetchone()[0]
        new_b = max(b, pd.Timestamp(latest_train).floor("s") + pd.Timedelta(seconds=1)) if latest_train else b
        latest_val = con.execute("SELECT max(end_upper) FROM activity WHERE gid=? AND ts>=? AND ts<=?", [gid, new_b.to_pydatetime(), c.to_pydatetime()]).fetchone()[0]
        new_d = max([d] + [pd.Timestamp(t).floor("s") + pd.Timedelta(seconds=1) for t in (latest_train, latest_val) if t is not None])
        con.execute("INSERT INTO policies VALUES ('B',?,?,?,?,?)", [gid, a.to_pydatetime(), new_b.to_pydatetime(), c.to_pydatetime(), new_d.to_pydatetime()])
        first_bad_train = con.execute("SELECT min(ts) FROM activity WHERE gid=? AND ts<=? AND end_upper>=?", [gid, a.to_pydatetime(), b.to_pydatetime()]).fetchone()[0]
        first_bad_val = con.execute("SELECT min(ts) FROM activity WHERE gid=? AND ts>=? AND ts<=? AND end_upper>=?", [gid, b.to_pydatetime(), c.to_pydatetime(), d.to_pydatetime()]).fetchone()[0]
        new_a = min(a, pd.Timestamp(first_bad_train) - pd.Timedelta(seconds=1)) if first_bad_train else a
        new_c = min(c, pd.Timestamp(first_bad_val) - pd.Timedelta(seconds=1)) if first_bad_val else c
        con.execute("INSERT INTO policies VALUES ('C',?,?,?,?,?)", [gid, new_a.to_pydatetime(), b.to_pydatetime(), new_c.to_pydatetime(), d.to_pydatetime()])


def activate_population_design(con, design: str) -> None:
    if design not in {"A", "B", "C", "C+"}:
        raise SplitError("Only metadata designs A, B, C and C+ exist")
    con.execute(f"""CREATE OR REPLACE TEMP VIEW proposed AS
        SELECT f.*, p.train_end,p.val_start,p.val_end,p.test_start,
          CASE WHEN metadata_issue IS NOT NULL THEN 'embargo'
            WHEN p.train_end IS NULL THEN 'train'
            WHEN ts<=p.train_end THEN 'train'
            WHEN ts>=p.val_start AND ts<=p.val_end THEN 'val'
            WHEN ts>=p.test_start THEN 'test' ELSE 'embargo' END AS split
        FROM activity f JOIN policies p ON f.gid=p.gid AND p.design={sql_literal(design)}""")


def population_crossing_counts(con) -> list[dict]:
    return con.execute("""SELECT boundary, count(*) AS crossing_rows,
       count(DISTINCT (sid,rid)) AS unique_source_rows,
       count(*) FILTER(WHERE end_estimate>=boundary_time) AS estimated_crossing_rows,
       max(epoch_us(end_estimate)-epoch_us(boundary_time))/1000000.0 AS max_overrun_s
       FROM (
         SELECT *, 'train_to_val' AS boundary,val_start AS boundary_time FROM proposed WHERE split='train' AND val_start IS NOT NULL
         UNION ALL SELECT *, 'train_to_test',test_start FROM proposed WHERE split='train' AND test_start IS NOT NULL
         UNION ALL SELECT *, 'val_to_test',test_start FROM proposed WHERE split='val' AND test_start IS NOT NULL
       ) WHERE end_upper>=boundary_time GROUP BY boundary ORDER BY boundary""").fetchdf().to_dict(orient="records")


def verify_full_record_duplicates(con) -> dict:
    """Find SHA256 candidates, then compare ALL original string fields exactly."""
    con.execute("""CREATE OR REPLACE TEMP TABLE duplicate_hashes AS
       SELECT record_sha, count(*) AS copies FROM flows GROUP BY record_sha HAVING count(*)>1""")
    candidates = con.execute("""SELECT f.sid,f.rid,hex(f.record_sha) AS hash,s.path
       FROM flows f JOIN duplicate_hashes d USING(record_sha) JOIN sources s USING(sid)
       ORDER BY f.sid,f.rid""").fetchdf()
    if len(candidates) > 100_000:
        raise SplitError("Unexpectedly many duplicate candidates; stop before unbounded exact-row materialization")
    actual = {}
    for path, part in candidates.groupby("path", sort=True):
        wanted = set(map(int, part["rid"]))
        offset = 0
        with pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=50_000) as reader:
            for chunk in reader:
                ids = np.array(sorted(i-offset for i in wanted if offset <= i < offset+len(chunk)), dtype=int)
                for pos in ids:
                    actual[(int(part["sid"].iloc[0]), offset+int(pos))] = tuple(chunk.iloc[int(pos)].tolist())
                offset += len(chunk)
                if offset > max(wanted):
                    break
    groups, verified_rows = {}, []
    for row in candidates.itertuples(index=False):
        value = actual[(row.sid,row.rid)]
        group = groups.setdefault(value, len(groups))
        verified_rows.append((row.sid,row.rid,group))
    counts = {}
    for _,_,group in verified_rows:
        counts[group] = counts.get(group,0)+1
    verified_rows = [r for r in verified_rows if counts[r[2]]>1]
    con.execute("CREATE OR REPLACE TABLE exact_duplicate_members(sid SMALLINT,rid BIGINT,duplicate_group INTEGER)")
    if verified_rows:
        con.executemany("INSERT INTO exact_duplicate_members VALUES (?,?,?)",verified_rows)
    return {"sha256_candidate_rows":len(candidates),"exact_duplicate_groups":sum(n>1 for n in counts.values()),
            "exact_duplicate_extra_copies":sum(n-1 for n in counts.values() if n>1),
            "exact_fields_verified":True,"definition":"All decoded original CSV fields as strings, fixed column order; not approximate numeric duplicates"}


def audit_population_design(con, design: str, out: Path) -> dict:
    activate_population_design(con, design)
    counts = con.execute("SELECT split,count(*) AS flows,count(*) FILTER(WHERE gid=2) AS benign,count(*) FILTER(WHERE gid<>2) AS attack FROM proposed GROUP BY split ORDER BY split").fetchdf()
    by_group = con.execute("SELECT gid,split,count(*) AS flows,min(ts) AS start,max(ts) AS last_start,max(end_upper) AS max_end_upper FROM proposed GROUP BY gid,split ORDER BY gid,split").fetchdf()
    # A/B/C are single contiguous start-time blocks; each source/split is a segment.
    segments = con.execute("""SELECT sid,gid,split,count(*) AS flows,min(ts) AS start,max(ts) AS last_start
         FROM proposed GROUP BY sid,gid,split ORDER BY sid,split""").fetchdf()
    windows = []
    for row in segments.itertuples(index=False):
        if row.split == "embargo":
            continue
        for fw in (256,512):
            for fs in (8,16):
                windows.append({"design":design,"sid":row.sid,"gid":row.gid,"split":row.split,
                   "fw":fw,"fs":fs,"flows":row.flows,"windows":window_count_only(row.flows,fw,fs),
                   "proxy_label":"Benign" if row.gid==2 else "Attack"})
    window_df = pd.DataFrame(windows)
    window_df.to_csv(out/f"{design}_window_counts_by_source.csv",index=False)
    segments.to_csv(out/f"{design}_segments_summary.csv",index=False)
    by_group.to_csv(out/f"{design}_scenario_counts.csv",index=False)
    counts.to_csv(out/f"{design}_flow_label_counts.csv",index=False)
    # Exact memberships are deterministic functions of (gid,ts), not copied row lists.
    guards = []
    for row in con.execute("SELECT * FROM policies WHERE design=? ORDER BY gid",[design]).fetchdf().itertuples(index=False):
        if pd.isna(row.train_end):
            continue
        for name,lo,hi in (("train_to_val",row.train_end,row.val_start),("val_to_test",row.val_end,row.test_start)):
            guard = con.execute("SELECT count(*) FROM proposed WHERE gid=? AND split='embargo' AND ts>? AND ts<?",[row.gid,lo.to_pydatetime(),hi.to_pydatetime()]).fetchone()[0]
            guards.append({"gid":row.gid,"boundary":name,"embargo_flows":guard,"at_least_511":guard>=MIN_EMBARGO})
    pd.DataFrame(guards).to_csv(out/f"{design}_embargo_guards.csv",index=False)
    crossings = population_crossing_counts(con)
    con.execute("""CREATE OR REPLACE TEMP TABLE key_split AS SELECT ck,split,
        bit_or(1::UTINYINT) AS present FROM proposed
        WHERE split<>'embargo' AND ck IS NOT NULL GROUP BY ck,split""")
    con.execute("""CREATE OR REPLACE TEMP TABLE key_masks AS SELECT ck,
        sum(CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 4 END)::INTEGER AS mask
        FROM key_split GROUP BY ck""")
    global_keys = con.execute("""SELECT count(*) FILTER(WHERE bit_count(mask)>1) AS any_shared_key,
        count(*) FILTER(WHERE (mask&3)=3) AS train_val,
        count(*) FILTER(WHERE (mask&5)=5) AS train_test,
        count(*) FILTER(WHERE (mask&6)=6) AS val_test FROM key_masks""").fetchdf().to_dict(orient="records")[0]
    con.execute("""CREATE OR REPLACE TEMP TABLE group_keys AS SELECT gid,ck,split,
        min(ts) AS first_start,max(end_estimate) AS last_end,max(end_upper) AS last_end_upper,
        max(proto) AS proto,count(*) AS flows,
        arg_max(fin,struct_pack(e:=end_estimate,s:=sid,r:=rid)) AS fin_latest,
        arg_max(rst,struct_pack(e:=end_estimate,s:=sid,r:=rid)) AS rst_latest
        FROM proposed WHERE split<>'embargo' AND ck IS NOT NULL GROUP BY gid,ck,split""")
    session_rows = []
    for left,right in (("train","val"),("train","test"),("val","test")):
        con.execute(f"""CREATE OR REPLACE TEMP VIEW paired_keys AS SELECT l.gid,l.ck,l.proto,
          l.last_end,l.last_end_upper,r.first_start,l.fin_latest,l.rst_latest,
          (epoch_us(r.first_start)-epoch_us(l.last_end))/1000000.0 AS gap_estimate_s
          FROM group_keys l JOIN group_keys r USING(gid,ck)
          WHERE l.split={sql_literal(left)} AND r.split={sql_literal(right)}""")
        summary = con.execute("""SELECT count(*) AS shared_group_keys,
          count(*) FILTER(WHERE last_end>=first_start) AS keys_estimated_interval_overlap,
          count(*) FILTER(WHERE last_end_upper>=first_start) AS keys_possible_interval_overlap,
          count(*) FILTER(WHERE proto IN(6,17) AND first_start<=last_end_upper+INTERVAL 1 SECOND) AS session_continuity_candidates_1s,
          count(*) FILTER(WHERE proto=6 AND first_start<=last_end_upper+INTERVAL 1 SECOND AND fin_latest=0 AND rst_latest=0) AS tcp_candidates_no_fin_rst,
          count(*) FILTER(WHERE proto NOT IN(6,17)) AS non_transport_shared_keys
          FROM paired_keys""").fetchdf().to_dict(orient="records")[0]
        summary.update({"boundary":left+"_to_"+right})
        session_rows.append(summary)
        con.execute(f"""COPY (SELECT *,{sql_literal(left+'_to_'+right)} AS boundary
          FROM paired_keys WHERE first_start<=last_end_upper+INTERVAL 1 SECOND
          ORDER BY gid,ck LIMIT 100) TO {sql_literal(out/f'{design}_{left}_{right}_session_examples.csv')} (HEADER,FORMAT CSV)""")
    duplicates = con.execute("""SELECT count(*) AS groups_crossing_retained_splits FROM (
        SELECT d.duplicate_group,count(DISTINCT p.split) AS splits
        FROM exact_duplicate_members d JOIN proposed p USING(sid,rid)
        WHERE p.split<>'embargo' GROUP BY d.duplicate_group HAVING splits>1)""").fetchone()[0]
    n = int(counts["flows"].sum())
    counts_map = {role:int(counts.loc[counts.split.eq(role),"flows"].sum()) for role in SPLITS}
    retained = n-counts_map["embargo"]
    invariant = {
        "each_source_row_one_role":"PASS" if n==con.execute("SELECT count(*) FROM flows").fetchone()[0] else "FAIL",
        "timestamp_buckets_not_cut":"PASS_BY_VERIFIED_PURE_GID_TIMESTAMP_MEMBERSHIP",
        "sorted_segments_nonoverlap":"PASS_BY_ORDERED_DISJOINT_BLOCK_CUTOFFS",
        "minimum_511_all_boundaries":"PASS" if guards and all(g["at_least_511"] for g in guards) else "FAIL",
        "no_estimated_activity_crossings":"PASS" if not crossings else "FAIL",
        "no_cross_split_exact_record_duplicates":"PASS" if duplicates==0 else "FAIL",
        "capture_session_independence":"NOT_ESTABLISHED",
        "window_overlap_across_splits":"NOT_CREATED_COUNT_ONLY_SOURCE_DISJOINT_DESIGN",
    }
    # Independently validate every actual timestamp bucket, not just a claim about SQL.
    bucket_bad = con.execute("SELECT count(*) FROM (SELECT gid,ts,count(DISTINCT split) AS roles FROM proposed GROUP BY gid,ts HAVING roles>1)").fetchone()[0]
    invariant["timestamp_buckets_not_cut"] = "PASS" if bucket_bad==0 else "FAIL"
    # Stable sorted source rank ranges imply no overlap and no split reappearing later.
    order_bad = con.execute("""SELECT count(*) FROM (SELECT sid,
       max(ts) FILTER(WHERE split='train') AS a,min(ts) FILTER(WHERE split='val') AS b,
       max(ts) FILTER(WHERE split='val') AS c,min(ts) FILTER(WHERE split='test') AS d
       FROM proposed GROUP BY sid) WHERE a>=b OR c>=d OR a>=d""").fetchone()[0]
    invariant["sorted_segments_nonoverlap"] = "PASS" if order_bad==0 else "FAIL"
    aggregate_windows = window_df.groupby(["fw","fs","split","proxy_label"],observed=True)["windows"].sum().reset_index()
    aggregate_windows.to_csv(out/f"{design}_window_counts_totals.csv",index=False)
    missing = []
    for fw in (256,512):
        for fs in (8,16):
            for gid,name in enumerate(GROUP_NAMES,1):
                for role in ("val","test"):
                    subset=window_df.loc[window_df.gid.eq(gid)&window_df.split.eq(role)&window_df.fw.eq(fw)&window_df.fs.eq(fs)]
                    if subset["windows"].sum()==0:
                        missing.append({"fw":fw,"fs":fs,"gid":gid,"scenario":name,"split":role,"proxy_label":"Benign" if gid==2 else "Attack"})
    pd.DataFrame(missing).to_csv(out/f"{design}_scenarios_without_eval_windows.csv",index=False)
    result = {"design":design,"flows":counts_map,"retained":retained,
        "percent_retained":{role:counts_map[role]/retained*100 for role in ("train","val","test")},
        "labels":counts.to_dict(orient="records"),"activity_crossings":crossings,
        "activity_crossing_records":sum(r["crossing_rows"] for r in crossings),
        "shared_keys_global":global_keys,"session_candidates_by_boundary":session_rows,
        "session_continuity_candidate_group_key_boundaries":sum(r["session_continuity_candidates_1s"] for r in session_rows),
        "cross_split_exact_record_duplicate_groups":duplicates,
        "windows":aggregate_windows.to_dict(orient="records"),"scenarios_without_eval_windows":missing,
        "checks":invariant}
    (out/f"{design}_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return result


def _ratio_objective(counts: np.ndarray) -> tuple:
    targets = np.array([60.0, 20.0, 20.0])
    retained = int(counts.sum())
    if retained <= 0:
        return (1, float("inf"), float("inf"), 0)
    deviations = np.abs(counts / retained * 100.0 - targets)
    maximum = float(deviations.max())
    total = float(deviations.sum())
    return (int(maximum > 0.5000001), maximum, total, -retained)


def _best_ratio_candidate(candidates: list[tuple[int, int, int]], minimum_retained: int):
    if not candidates:
        return None
    array = np.asarray(candidates, dtype=np.int64)
    retained = array.sum(axis=1)
    eligible = np.flatnonzero(retained >= minimum_retained)
    if not len(eligible):
        return None
    target = np.array([60.0, 20.0, 20.0])
    percentages = array[eligible] / retained[eligible, None] * 100.0
    deviations = np.abs(percentages - target)
    maximum = deviations.max(axis=1)
    total = deviations.sum(axis=1)
    outside = maximum > 0.5000001
    # Deterministic: first meet the 0.5 pp goal, then minimize ratio error,
    # then prefer more retained flow.
    order = np.lexsort((-retained[eligible], total, maximum, outside.astype(np.int8)))
    index = int(eligible[order[0]])
    return index, array[index], _ratio_objective(array[index])


def optimize_cplus_from_metadata(metadata_dir: Path, out: Path,
                                 search_radius_seconds: int = 300) -> dict:
    """One bounded C+ cutoff search using an existing DuckDB index only."""
    import duckdb

    metadata_dir = metadata_dir.resolve()
    out = out.resolve()
    database = metadata_dir / "metadata.duckdb"
    full_audit_path = metadata_dir / "audit_full.json"
    baseline_path = metadata_dir / "C_audit.json"
    if not database.is_file() or not full_audit_path.is_file() or not baseline_path.is_file():
        raise SplitError("C+ requires the existing stage32 DuckDB and full/C audit artifacts")
    full_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
    if (full_audit.get("scope") != "FULL_POPULATION_METADATA_AUDIT_NOT_FULL_SPLIT"
            or full_audit.get("rows") != EXPECTED_TOTAL
            or full_audit.get("source_row_conservation") != "PASS"
            or full_audit.get("windows_created")
            or full_audit.get("flow_csv_exported")):
        raise SplitError("Existing stage32 audit is incomplete or not metadata-only")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("design") != "C" or baseline.get("checks", {}).get("no_estimated_activity_crossings") != "PASS":
        raise SplitError("The saved C baseline does not pass its activity guard")
    if search_radius_seconds <= 0 or search_radius_seconds > 300:
        raise SplitError("This optimizer is bounded to 1..300 seconds around C cutoffs")
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=2")
        con.execute(f"ATTACH {sql_literal(database)} AS stage2 (READ_ONLY)")
        con.execute("CREATE VIEW flows AS SELECT * FROM stage2.main.flows")
        con.execute("CREATE VIEW activity AS SELECT * FROM stage2.main.activity")
        con.execute("CREATE VIEW exact_duplicate_members AS SELECT * FROM stage2.main.exact_duplicate_members")
        con.execute("CREATE TABLE policies AS SELECT * FROM stage2.main.policies WHERE design <> 'C+'")
        rows = int(full_audit["rows"])
        if rows != EXPECTED_TOTAL:
            raise SplitError(f"Existing metadata audit row count changed: {rows}")
        # One timestamp-bucket aggregation from compact DuckDB metadata; no CSVs are opened.
        buckets = con.execute("""SELECT gid,ts,count(*)::BIGINT AS n,max(end_upper) AS max_end
          FROM activity GROUP BY gid,ts ORDER BY gid,ts""").fetchdf()
        if int(buckets["n"].sum()) != rows:
            raise SplitError("Timestamp-bucket aggregation does not conserve all source flows")
        policies = con.execute("SELECT gid,train_end,val_start,val_end,test_start FROM policies WHERE design='C' ORDER BY gid").fetchdf().set_index("gid")
        if len(policies) != len(GROUP_NAMES):
            raise SplitError("C policy table is incomplete")

        group_data = {}
        for gid, frame in buckets.groupby("gid", sort=True):
            gid = int(gid)
            timestamps = frame["ts"].to_numpy(dtype="datetime64[s]")
            counts = frame["n"].to_numpy(dtype=np.int64)
            cumulative = np.cumsum(counts)
            end_ns = frame["max_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
            group_data[gid] = (timestamps, cumulative, end_ns,
                               np.maximum.accumulate(end_ns), int(cumulative[-1]))
        optimized = policies.copy(deep=True)

        def le_index(timestamps, value) -> int:
            return int(np.searchsorted(timestamps, np.datetime64(pd.Timestamp(value), "s"), side="right") - 1)

        def ge_index(timestamps, value) -> int:
            return int(np.searchsorted(timestamps, np.datetime64(pd.Timestamp(value), "s"), side="left"))

        def cumulative_at(cumulative, index: int) -> int:
            return int(cumulative[index]) if index >= 0 else 0

        def group_counts(gid: int, row) -> np.ndarray:
            timestamps, cumulative, _, _, total = group_data[gid]
            if pd.isna(row.train_end):
                return np.array([total, 0, 0], dtype=np.int64)
            ti = le_index(timestamps, row.train_end)
            vi = ge_index(timestamps, row.val_start)
            ve = le_index(timestamps, row.val_end)
            xi = ge_index(timestamps, row.test_start)
            return np.array([cumulative_at(cumulative, ti),
                cumulative_at(cumulative, ve) - cumulative_at(cumulative, vi - 1),
                total - cumulative_at(cumulative, xi - 1)], dtype=np.int64)

        per_group = {int(gid): group_counts(int(gid), optimized.loc[gid]) for gid in optimized.index}
        total_counts = sum(per_group.values(), np.zeros(3, dtype=np.int64))
        baseline_counts = np.array([baseline["flows"][k] for k in ("train", "val", "test")], dtype=np.int64)
        if not np.array_equal(total_counts, baseline_counts):
            raise SplitError("Reconstructed C counts do not match the saved full audit")
        minimum_retained = int(baseline_counts.sum())
        radius = np.timedelta64(search_radius_seconds, "s")
        candidates_examined = 0
        accepted_moves = []

        def near_indices(timestamps, center):
            center = np.datetime64(pd.Timestamp(center), "s")
            return np.flatnonzero((timestamps >= center - radius) & (timestamps <= center + radius))

        # Freeze Benign membership and the six B2 Train-only groups.
        group_order = sorted(group_data, key=lambda gid: (-group_data[gid][4], gid))
        for gid in group_order:
            row = optimized.loc[gid]
            if gid == 2 or pd.isna(row.train_end):
                continue
            timestamps, cumulative, end_ns, prefix_end, group_total = group_data[gid]
            train_i = le_index(timestamps, row.train_end)
            val_j = ge_index(timestamps, row.val_start)
            val_end_i = le_index(timestamps, row.val_end)
            test_j = ge_index(timestamps, row.test_start)
            if min(train_i, val_j, val_end_i, test_j) < 0 or max(train_i, val_j, val_end_i, test_j) >= len(timestamps):
                continue

            # Search a safe Train/Validation cutoff pair within +/- 300 seconds.
            train_options = near_indices(timestamps, row.train_end)
            val_start_options = near_indices(timestamps, row.val_start)
            val_end_suffix = np.maximum.accumulate(end_ns[:val_end_i + 1][::-1])[::-1]
            local_global = sum(per_group.values(), np.zeros(3, dtype=np.int64))
            left_candidates, left_params = [], []
            test_start_ns = int(timestamps[test_j].astype("datetime64[ns]").astype("int64"))
            for vj in val_start_options:
                if vj > val_end_i or val_end_suffix[vj] >= test_start_ns:
                    continue
                for ti in train_options:
                    if ti >= vj:
                        continue
                    candidates_examined += 1
                    gap = cumulative_at(cumulative, int(vj) - 1) - cumulative_at(cumulative, int(ti))
                    start_ns = int(timestamps[vj].astype("datetime64[ns]").astype("int64"))
                    if gap < MIN_EMBARGO or prefix_end[ti] >= start_ns:
                        continue
                    train_n = cumulative_at(cumulative, int(ti))
                    val_n = cumulative_at(cumulative, val_end_i) - cumulative_at(cumulative, int(vj) - 1)
                    test_n = group_total - cumulative_at(cumulative, test_j - 1)
                    if min(train_n, val_n, test_n) < 512:
                        continue
                    proposed = local_global - per_group[gid] + np.array([train_n, val_n, test_n], dtype=np.int64)
                    left_candidates.append(tuple(proposed.tolist()))
                    left_params.append((int(ti), int(vj)))
            selected = _best_ratio_candidate(left_candidates, minimum_retained)
            if selected is not None:
                option_i, best, _ = selected
                if _ratio_objective(best) < _ratio_objective(local_global):
                    old_group = per_group[gid].copy()
                    per_group[gid] = best - (local_global - old_group)
                    ti, vj = left_params[option_i]
                    optimized.at[gid, "train_end"] = pd.Timestamp(timestamps[ti])
                    optimized.at[gid, "val_start"] = pd.Timestamp(timestamps[vj])
                    accepted_moves.append({"gid": gid, "boundary": "train_to_val",
                        "train_end": str(optimized.at[gid, "train_end"]),
                        "val_start": str(optimized.at[gid, "val_start"])})

            # Search Validation/Test only after the left pair is fixed; recheck
            # the maximum activity end of every included validation bucket.
            row = optimized.loc[gid]
            val_start_i = ge_index(timestamps, row.val_start)
            val_end_options = near_indices(timestamps, policies.loc[gid, "val_end"])
            test_options = near_indices(timestamps, policies.loc[gid, "test_start"])
            local_global = sum(per_group.values(), np.zeros(3, dtype=np.int64))
            right_candidates, right_params = [], []
            for ve in val_end_options:
                if ve < val_start_i:
                    continue
                max_val_end = int(np.max(end_ns[val_start_i:ve + 1]))
                val_n = cumulative_at(cumulative, int(ve)) - cumulative_at(cumulative, val_start_i - 1)
                if val_n < 512:
                    continue
                for xj in test_options:
                    if xj <= ve:
                        continue
                    candidates_examined += 1
                    gap = cumulative_at(cumulative, int(xj) - 1) - cumulative_at(cumulative, int(ve))
                    start_ns = int(timestamps[xj].astype("datetime64[ns]").astype("int64"))
                    if gap < MIN_EMBARGO or max_val_end >= start_ns:
                        continue
                    train_n = per_group[gid][0]
                    test_n = group_total - cumulative_at(cumulative, int(xj) - 1)
                    if test_n < 512:
                        continue
                    proposed = local_global - per_group[gid] + np.array([train_n, val_n, test_n], dtype=np.int64)
                    right_candidates.append(tuple(proposed.tolist()))
                    right_params.append((int(ve), int(xj)))
            selected = _best_ratio_candidate(right_candidates, minimum_retained)
            if selected is not None:
                option_i, best, _ = selected
                if _ratio_objective(best) < _ratio_objective(local_global):
                    old_group = per_group[gid].copy()
                    per_group[gid] = best - (local_global - old_group)
                    ve, xj = right_params[option_i]
                    optimized.at[gid, "val_end"] = pd.Timestamp(timestamps[ve])
                    optimized.at[gid, "test_start"] = pd.Timestamp(timestamps[xj])
                    accepted_moves.append({"gid": gid, "boundary": "val_to_test",
                        "val_end": str(optimized.at[gid, "val_end"]),
                        "test_start": str(optimized.at[gid, "test_start"])})

        # Add C+ to this ephemeral audit connection only; stage32 DuckDB stays read-only.
        policy_rows = []
        for gid, row in optimized.sort_index().iterrows():
            values = [None if pd.isna(row[name]) else pd.Timestamp(row[name]).to_pydatetime()
                      for name in ("train_end", "val_start", "val_end", "test_start")]
            policy_rows.append(["C+", int(gid), *values])
        con.executemany("INSERT INTO policies VALUES (?,?,?,?,?,?)", policy_rows)
        result = audit_population_design(con, "C+", out)
        con.execute(f"COPY (SELECT * FROM policies WHERE design='C+' ORDER BY gid) TO {sql_literal(out/'Cplus_cutoffs.csv')} (HEADER,FORMAT CSV)")

        required = ("each_source_row_one_role", "timestamp_buckets_not_cut",
                    "sorted_segments_nonoverlap", "minimum_511_all_boundaries",
                    "no_estimated_activity_crossings", "no_cross_split_exact_record_duplicates")
        failed = [name for name in required if result["checks"].get(name) != "PASS"]
        if failed:
            raise SplitError(f"C+ failed required post-optimization checks: {failed}")
        if result["retained"] < minimum_retained:
            raise SplitError("C+ retained fewer rows than the C baseline")

        comparison = []
        for design, item in (("C", baseline), ("C+", result)):
            comparison.append({"design": design, "train": item["flows"]["train"],
                "val": item["flows"]["val"], "test": item["flows"]["test"],
                "embargo": item["flows"]["embargo"], "retained": item["retained"],
                "train_pct": item["percent_retained"]["train"],
                "val_pct": item["percent_retained"]["val"],
                "test_pct": item["percent_retained"]["test"],
                "activity_crossings": item["activity_crossing_records"],
                "shared_keys": item["shared_keys_global"]["any_shared_key"],
                "session_candidates": item["session_continuity_candidate_group_key_boundaries"]})
        pd.DataFrame(comparison).to_csv(out/"comparison_C_Cplus.csv", index=False)
        labels = [{"design": design, **entry} for design, item in (("C", baseline), ("C+", result)) for entry in item["labels"]]
        pd.DataFrame(labels).to_csv(out/"proxy_label_counts_C_Cplus.csv", index=False)
        optimization = {"scope": "ONE_BOUNDED_CUTOFF_SEARCH_ON_EXISTING_DUCKDB_ONLY",
            "source_csvs_opened": False, "rows_in_duckdb": rows,
            "timestamp_buckets_aggregated": len(buckets),
            "search_radius_seconds_per_cutoff": search_radius_seconds,
            "benign_gid_2_cutoffs_frozen": True,
            "minimum_retained_rows": minimum_retained,
            "candidate_pairs_examined": candidates_examined,
            "accepted_cutoff_moves": accepted_moves,
            "windows_created": False, "flow_csv_exported": False, "models_trained": False,
            "baseline_C": baseline, "optimized_Cplus": result}
        (out/"optimization_Cplus.json").write_text(json.dumps(optimization, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return optimization
    finally:
        con.close()


def create_cplus_membership_view(con, flow_relation: str = "stage2.main.flows") -> None:
    """Apply the frozen C+ cutoff table to each source row exactly once."""
    con.execute(f"""CREATE OR REPLACE TEMP VIEW cplus_membership AS
        SELECT f.sid, f.rid, f.gid, f.ts, f.duration_us, f.record_sha,
               f.metadata_issue, p.train_end, p.val_start, p.val_end, p.test_start,
          CASE WHEN f.metadata_issue IS NOT NULL THEN 'embargo'
            WHEN p.train_end IS NULL THEN 'train'
            WHEN f.ts<=p.train_end THEN 'train'
            WHEN f.ts>=p.val_start AND f.ts<=p.val_end THEN 'val'
            WHEN f.ts>=p.test_start THEN 'test' ELSE 'embargo' END AS split
        FROM {flow_relation} f JOIN cplus_policies p ON f.gid=p.gid""")


def _source_csv_reader(path: Path) -> str:
    return (f"read_csv({sql_literal(path)}, header=true, all_varchar=true, parallel=false, "
            "nullstr='__AUDIT_NULL_SENTINEL__', strict_mode=true, ignore_errors=false)")


def export_cplus_source_segment(con, source_id: int, source_path: Path,
                                columns: list[str], split_name: str,
                                output_path: Path) -> None:
    """Export one source/split as a timestamp-sorted, source-isolated CSV."""
    if split_name not in {"train", "val", "test"}:
        raise SplitError(f"Refusing to export non-retained role: {split_name}")
    raw_columns = ",".join(f"r.{sql_identifier(name)}" for name in columns)
    select_columns = ",".join([
        raw_columns,
        f"{sql_literal(str(source_path.resolve()))} AS {sql_identifier('Source_CSV')}",
        f"m.sid AS {sql_identifier('Source_CSV_ID')}",
        f"m.rid AS {sql_identifier('Original_Row_Index')}",
        f"m.split AS {sql_identifier('Split')}",
        f"CASE WHEN m.gid=2 THEN 'Benign' ELSE 'Attack' END AS {sql_identifier('Proxy_Label')}",
        f"'SCENARIO_PROXY' AS {sql_identifier('Proxy_Label_Source')}",
    ])
    raw = _source_csv_reader(source_path)
    query = f"""WITH raw_rows AS (
        SELECT row_number() OVER ()-1 AS __source_row, * FROM {raw}
      )
      SELECT {select_columns}
      FROM raw_rows r JOIN cplus_membership m
        ON m.sid={int(source_id)} AND m.rid=r.__source_row
      WHERE m.split={sql_literal(split_name)}
      ORDER BY try_strptime(r.{sql_identifier('Timestamp')}, '%d/%m/%Y %I:%M:%S %p'),
               m.sid, m.rid"""
    con.execute(f"""COPY ({query}) TO {sql_literal(output_path)}
      (HEADER, DELIMITER ',', QUOTE '"', ESCAPE '"')""")


def _append_output_for_audit(con, output_path: Path, source_id: int,
                             source_path: Path, split_name: str,
                             columns: list[str], expected_rows: int,
                             expected_proxy: str) -> dict:
    """Read a completed CSV back, verify order/schema/provenance, and hash raw fields."""
    with output_path.open("r", encoding="utf-8-sig", newline="") as stream:
        output_header = next(csv.reader(stream))
    expected_header = columns + ["Source_CSV", "Source_CSV_ID", "Original_Row_Index",
                                 "Split", "Proxy_Label", "Proxy_Label_Source"]
    if output_header != expected_header:
        raise SplitError(f"Output schema mismatch: {output_path}")
    original_fields = ",".join(sql_identifier(name) for name in columns)
    reader = _source_csv_reader(output_path)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE one_export AS
      SELECT row_number() OVER ()::BIGINT AS output_position,
        try_strptime({sql_identifier('Timestamp')}, '%d/%m/%Y %I:%M:%S %p') AS ts,
        try_cast({sql_identifier('Source_CSV_ID')} AS SMALLINT) AS sid,
        try_cast({sql_identifier('Original_Row_Index')} AS BIGINT) AS rid,
        {sql_identifier('Split')} AS split,
        coalesce({sql_identifier('Source_CSV')}={sql_literal(str(source_path.resolve()))},false) AS source_path_ok,
        {sql_identifier('Proxy_Label')} AS proxy_label,
        coalesce({sql_identifier('Proxy_Label_Source')}='SCENARIO_PROXY',false) AS proxy_source_ok,
        lower(sha256(to_json(list_value({original_fields})))) AS record_hash
      FROM {reader}""")
    total = int(con.execute("SELECT count(*) FROM one_export").fetchone()[0])
    if total != expected_rows:
        raise SplitError(f"Readback row count mismatch for {output_path}: {total} != {expected_rows}")
    bad_rows = con.execute("""SELECT count(*) FROM one_export
      WHERE sid IS NULL OR rid IS NULL OR ts IS NULL OR NOT source_path_ok
        OR NOT proxy_source_ok OR proxy_label<>? OR split<>? OR sid<>?""",
      [expected_proxy, split_name, source_id]).fetchone()[0]
    order_bad = con.execute("""SELECT count(*) FROM (
      SELECT ts,rid,lag(ts) OVER(ORDER BY output_position) AS prev_ts,
             lag(rid) OVER(ORDER BY output_position) AS prev_rid
      FROM one_export) q
      WHERE ts IS NULL OR (prev_ts IS NOT NULL AND
        (ts<prev_ts OR (ts=prev_ts AND rid<=prev_rid)))""").fetchone()[0]
    if bad_rows or order_bad:
        raise SplitError(f"Output readback failed for {output_path}: bad rows={bad_rows}, sort inversions={order_bad}")
    con.execute("""CREATE TEMP TABLE IF NOT EXISTS export_audit(
      sid SMALLINT,rid BIGINT,split VARCHAR,source_path_ok BOOLEAN,
      proxy_label VARCHAR,proxy_source_ok BOOLEAN,record_hash VARCHAR,ts TIMESTAMP)""")
    con.execute("""INSERT INTO export_audit
      SELECT sid,rid,split,source_path_ok,proxy_label,proxy_source_ok,record_hash,ts
      FROM one_export""")
    con.execute("DROP TABLE one_export")
    return {"rows": total, "sort_inversions": int(order_bad), "bad_provenance_or_labels": int(bad_rows)}


def _read_cplus_cutoffs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"design", "gid", "train_end", "val_start", "val_end", "test_start"}
    if set(frame.columns) != required or len(frame) != len(GROUP_NAMES):
        raise SplitError("Frozen C+ cutoff file has an unexpected schema or row count")
    if set(frame["design"]) != {"C+"} or frame["gid"].duplicated().any():
        raise SplitError("Cutoff file must contain exactly one C+ row per group")
    frame["gid"] = pd.to_numeric(frame["gid"], errors="raise").astype("int16")
    if sorted(frame["gid"].tolist()) != list(range(1, len(GROUP_NAMES) + 1)):
        raise SplitError("C+ cutoff file does not cover group IDs 1..19")
    for name in ("train_end", "val_start", "val_end", "test_start"):
        values = frame[name].replace("", pd.NA)
        parsed = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if (values.notna() & parsed.isna()).any():
            raise SplitError(f"Malformed C+ timestamp in {name}")
        frame[name] = parsed
    return frame.sort_values("gid", kind="stable").reset_index(drop=True)


def _cplus_preflight(con, expected: dict[str, int]) -> dict:
    create_cplus_membership_view(con)
    total_rows, unique_rows = con.execute("""SELECT count(*),count(DISTINCT (sid,rid))
      FROM stage2.main.flows""").fetchone()
    if int(total_rows) != EXPECTED_TOTAL or int(unique_rows) != int(total_rows):
        raise SplitError(f"Stage32 source-row identity/conservation failed: {total_rows}, {unique_rows}")
    role_rows = con.execute("SELECT split,count(*) FROM cplus_membership GROUP BY split").fetchall()
    counts = {name: 0 for name in SPLITS}
    counts.update({str(name): int(n) for name, n in role_rows})
    if counts != expected:
        raise SplitError(f"Frozen C+ role counts differ from approved counts: {counts} != {expected}")
    if sum(counts.values()) != EXPECTED_TOTAL:
        raise SplitError("C+ roles do not conserve the full source population")

    bucket_cuts = int(con.execute("""SELECT count(*) FROM (
      SELECT gid,ts,count(DISTINCT split) roles FROM cplus_membership
      GROUP BY gid,ts HAVING roles>1)""").fetchone()[0])
    segment_cuts = int(con.execute("""WITH r AS (
      SELECT sid,max(ts) FILTER(WHERE split='train') AS train_last,
        min(ts) FILTER(WHERE split='val') AS val_first,
        max(ts) FILTER(WHERE split='val') AS val_last,
        min(ts) FILTER(WHERE split='test') AS test_first
      FROM cplus_membership GROUP BY sid)
      SELECT count(*) FROM r WHERE train_last>=val_first OR val_last>=test_first""").fetchone()[0])
    guards = con.execute("""SELECT p.gid,
      (SELECT count(*) FROM cplus_membership m WHERE m.gid=p.gid AND m.split='embargo'
       AND m.ts>p.train_end AND m.ts<p.val_start) AS train_val_embargo,
      (SELECT count(*) FROM cplus_membership m WHERE m.gid=p.gid AND m.split='embargo'
       AND m.ts>p.val_end AND m.ts<p.test_start) AS val_test_embargo
      FROM cplus_policies p WHERE p.train_end IS NOT NULL ORDER BY p.gid""").fetchdf()
    if bucket_cuts or segment_cuts:
        raise SplitError(f"C+ timestamp/segment invariant failed: bucket cuts={bucket_cuts}, segment cuts={segment_cuts}")
    if guards.empty or (guards[["train_val_embargo", "val_test_embargo"]] < MIN_EMBARGO).any().any():
        raise SplitError("At least one frozen C+ embargo guard is below 511 flows")
    crossings = int(con.execute("""SELECT count(*) FROM (
      SELECT m.sid,m.rid FROM cplus_membership m JOIN stage2.main.activity a USING(sid,rid)
        WHERE m.split='train' AND m.val_start IS NOT NULL AND a.end_upper>=m.val_start
      UNION ALL
      SELECT m.sid,m.rid FROM cplus_membership m JOIN stage2.main.activity a USING(sid,rid)
        WHERE m.split='train' AND m.test_start IS NOT NULL AND a.end_upper>=m.test_start
      UNION ALL
      SELECT m.sid,m.rid FROM cplus_membership m JOIN stage2.main.activity a USING(sid,rid)
        WHERE m.split='val' AND m.test_start IS NOT NULL AND a.end_upper>=m.test_start
      )""").fetchone()[0])
    exact_duplicate_crossings = int(con.execute("""SELECT count(*) FROM (
      SELECT d.duplicate_group,count(DISTINCT m.split) roles
      FROM stage2.main.exact_duplicate_members d JOIN cplus_membership m USING(sid,rid)
      WHERE m.split<>'embargo' GROUP BY d.duplicate_group HAVING roles>1)""").fetchone()[0])
    source_counts = con.execute("""SELECT s.sid,s.path,s.gid,s.scenario,s.rows,s.bytes,s.mtime_ns,s.sha256,
      count(f.rid) AS indexed_rows
      FROM stage2.main.sources s LEFT JOIN stage2.main.flows f USING(sid)
      GROUP BY s.sid,s.path,s.gid,s.scenario,s.rows,s.bytes,s.mtime_ns,s.sha256 ORDER BY s.sid""").fetchdf()
    if len(source_counts) != EXPECTED_FILES or int(source_counts["rows"].sum()) != EXPECTED_TOTAL:
        raise SplitError("Stage32 source inventory does not cover the approved 71 CSVs")
    if not (source_counts["rows"].astype("int64") == source_counts["indexed_rows"].astype("int64")).all():
        raise SplitError("Per-source DuckDB row counts differ from its source inventory")
    labels = con.execute("SELECT count(DISTINCT original_label),min(original_label),max(original_label) FROM stage2.main.flows").fetchone()
    if labels != (1, "No Label", "No Label"):
        raise SplitError(f"Unexpected original Label values in metadata index: {labels}")
    scenario_counts = con.execute("""SELECT gid,split,count(*)::BIGINT AS flows,
      min(ts) AS first_timestamp,max(ts) AS last_timestamp
      FROM cplus_membership GROUP BY gid,split ORDER BY gid,split""").fetchdf()
    proxy_counts = con.execute("""SELECT split,count(*)::BIGINT AS flows,
      count(*) FILTER(WHERE gid=2)::BIGINT AS benign,
      count(*) FILTER(WHERE gid<>2)::BIGINT AS attack
      FROM cplus_membership GROUP BY split ORDER BY split""").fetchdf()
    return {"counts": counts, "source_counts": source_counts,
            "scenario_counts": scenario_counts, "proxy_counts": proxy_counts,
            "guards": guards, "activity_crossings": crossings,
            "exact_duplicate_crossings": exact_duplicate_crossings,
            "timestamp_bucket_cuts": bucket_cuts, "segment_cuts": segment_cuts}


def run_full_cplus_split(dataset_root: Path, metadata_dir: Path, cutoff_file: Path,
                         out: Path, report_path: Path) -> dict:
    """Stream the approved C+ membership to isolated timestamp-sorted CSV segments."""
    import duckdb

    started = time.time()
    dataset_root, metadata_dir = dataset_root.resolve(), metadata_dir.resolve()
    cutoff_file, out, report_path = cutoff_file.resolve(), out.resolve(), report_path.resolve()
    database = metadata_dir / "metadata.duckdb"
    stage33_dir = metadata_dir.parent / "stage33_cutoff_optimization"
    frozen_audit_path = stage33_dir / "C+_audit.json"
    if not dataset_root.is_dir() or not database.is_file() or not cutoff_file.is_file():
        raise SplitError("Dataset CSV, stage32 DuckDB, or approved C+ cutoff file is missing")
    if not frozen_audit_path.is_file():
        raise SplitError("The frozen stage33 C+ audit is missing")
    if out.exists():
        raise SplitError(f"Output already exists; refusing to overwrite: {out}")
    if report_path.exists():
        raise SplitError(f"Stage34 report already exists; refusing to overwrite: {report_path}")
    if not out.parent.is_dir() or not report_path.parent.is_dir():
        raise SplitError("Output/report parent directory must already exist")
    if dataset_root == out or dataset_root in out.parents or out in dataset_root.parents:
        raise SplitError("Split output must be separate from the original CSV source")
    frozen_audit = json.loads(frozen_audit_path.read_text(encoding="utf-8"))
    if frozen_audit.get("design") != "C+" or frozen_audit.get("flows") != EXPECTED_CPLUS:
        raise SplitError("Stage33 C+ audit does not match the explicitly approved flow counts")
    for check in ("each_source_row_one_role", "timestamp_buckets_not_cut",
                  "sorted_segments_nonoverlap", "minimum_511_all_boundaries",
                  "no_estimated_activity_crossings", "no_cross_split_exact_record_duplicates"):
        if frozen_audit.get("checks", {}).get(check) != "PASS":
            raise SplitError(f"Frozen C+ audit check is not PASS: {check}")

    cutoffs = _read_cplus_cutoffs(cutoff_file)
    cutoff_signature = file_signature(cutoff_file, full_hash=True)
    stage_id = uuid.uuid4().hex
    staging = out.parent / f".{out.name}.partial-{stage_id}"
    report_tmp = report_path.with_name(f".{report_path.name}.partial-{stage_id}")
    source_hashes = {}
    source_initial_signatures = {}
    con = None
    try:
        con = duckdb.connect(":memory:")
        con.execute("SET memory_limit='4GB'")
        con.execute("SET threads=2")
        con.execute("SET preserve_insertion_order=true")
        con.execute(f"ATTACH {sql_literal(database)} AS stage2 (READ_ONLY)")
        con.register("_cplus_policy_frame", cutoffs[["gid", "train_end", "val_start", "val_end", "test_start"]])
        con.execute("""CREATE TEMP TABLE cplus_policies AS
          SELECT gid::UTINYINT AS gid,train_end::TIMESTAMP AS train_end,
            val_start::TIMESTAMP AS val_start,val_end::TIMESTAMP AS val_end,
            test_start::TIMESTAMP AS test_start FROM _cplus_policy_frame""")
        con.unregister("_cplus_policy_frame")
        preflight = _cplus_preflight(con, EXPECTED_CPLUS)

        source_rows = preflight["source_counts"].to_dict(orient="records")
        expected_columns = None
        reserved = {"Source_CSV", "Source_CSV_ID", "Original_Row_Index", "Split",
                    "Proxy_Label", "Proxy_Label_Source"}
        for row in source_rows:
            source_path = Path(row["path"]).resolve()
            if source_path.parent.parent != dataset_root or source_path.parent.name not in {"train", "test"}:
                raise SplitError(f"Source path is outside the approved train/test CSV folders: {source_path}")
            if scenario_from_path(source_path) != row["scenario"]:
                raise SplitError(f"Source scenario/path mismatch: {source_path}")
            with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader(stream))
            if expected_columns is None:
                expected_columns = columns
            elif columns != expected_columns:
                raise SplitError(f"Source CSV schema differs from the other files: {source_path}")
            if not set(REQUIRED).issubset(columns) or reserved.intersection(columns):
                raise SplitError(f"Missing required source fields or reserved output columns: {source_path}")
        if expected_columns is None:
            raise SplitError("No CSV header was found")

        source_bytes = int(preflight["source_counts"]["bytes"].sum())
        retained = EXPECTED_CPLUS["train"] + EXPECTED_CPLUS["val"] + EXPECTED_CPLUS["test"]
        output_estimate = source_bytes + retained * 128
        temp_allowance = 3 * source_bytes
        safety_reserve = 5 * 1024**3
        estimated_required = output_estimate + temp_allowance + safety_reserve
        disk_free = shutil.disk_usage(out.parent).free
        if disk_free < estimated_required:
            raise SplitError(f"Insufficient free disk: {disk_free} bytes available, at least {estimated_required} estimated")
        print(json.dumps({"phase": "preflight_passed", "source_rows": EXPECTED_TOTAL,
            "expected_flows": EXPECTED_CPLUS, "output_upper_estimate_bytes": output_estimate,
            "sort_temp_allowance_bytes": temp_allowance, "reserve_bytes": safety_reserve,
            "drive_free_bytes": disk_free}, ensure_ascii=False), flush=True)

        # Bind this exact input version to the stage32 index before any output is written.
        for index, row in enumerate(source_rows, 1):
            path = Path(row["path"]).resolve()
            current = file_signature(path)
            if current["bytes"] != int(row["bytes"]) or current["mtime_ns"] != int(row["mtime_ns"]):
                raise SplitError(f"Source stat differs from the audited stage32 version: {path}")
            hashed = file_signature(path, full_hash=True)
            after_hash = file_signature(path)
            if (hashed.get("sha256") != row["sha256"] or
                    any(hashed[k] != after_hash[k] for k in ("bytes", "mtime_ns"))):
                raise SplitError(f"Source SHA256 differs from stage32 or changed while hashing: {path}")
            source_hashes[int(row["sid"])] = hashed["sha256"]
            source_initial_signatures[int(row["sid"])] = after_hash
            if index % 10 == 0 or index == len(source_rows):
                print(f"Input SHA256 verified: {index}/{len(source_rows)} CSV", flush=True)

        staging.mkdir(parents=False, exist_ok=False)
        (staging / "INCOMPLETE.txt").write_text(
            "This directory is staging only. Do not use as a completed split unless it is atomically published.\n",
            encoding="utf-8")
        scratch = staging / "_duckdb_temp"
        scratch.mkdir()
        con.execute(f"SET temp_directory={sql_literal(scratch)}")
        for role in ("train", "val", "test"):
            (staging / role).mkdir()
        shutil.copyfile(cutoff_file, staging / "Cplus_cutoffs_used.csv")

        segments = con.execute("""SELECT sid,gid,split,count(*)::BIGINT AS flows,
            min(ts) AS first_timestamp,max(ts) AS last_timestamp,
            min(rid) AS min_original_row_index,max(rid) AS max_original_row_index
          FROM cplus_membership GROUP BY sid,gid,split ORDER BY sid,split""").fetchdf()
        segment_lookup = {(int(row.sid), str(row.split)): row
                          for row in segments.itertuples(index=False)}
        out_files = []
        output_audit_rows = 0
        for source_index, row in enumerate(source_rows, 1):
            sid = int(row["sid"])
            source_path = Path(row["path"]).resolve()
            scenario = str(row["scenario"])
            proxy = "Benign" if int(row["gid"]) == 2 else "Attack"
            for role in ("train", "val", "test"):
                segment = segment_lookup.get((sid, role))
                expected_rows = 0 if segment is None else int(segment.flows)
                if not expected_rows:
                    continue
                folder = staging / role / scenario
                folder.mkdir(parents=True, exist_ok=True)
                final_name = f"sid{sid:03d}__{source_path.stem}__{role}.csv"
                final_path = folder / final_name
                partial_path = folder / (final_name + ".partial")
                export_cplus_source_segment(con, sid, source_path, expected_columns, role, partial_path)
                partial_path.replace(final_path)
                result = _append_output_for_audit(con, final_path, sid, source_path,
                    role, expected_columns, expected_rows, proxy)
                output_audit_rows += result["rows"]
                out_files.append({"sid": sid, "scenario": scenario, "split": role,
                    "path": final_path, "relative_path": str(final_path.relative_to(staging)),
                    "rows": expected_rows, "sort_inversions": result["sort_inversions"]})
            if source_index % 10 == 0 or source_index == len(source_rows):
                print(f"Source CSVs exported and read-back checked: {source_index}/{len(source_rows)}",
                      flush=True)

        actual_output_rows = int(con.execute("SELECT count(*) FROM export_audit").fetchone()[0])
        unique_output_rows = int(con.execute("SELECT count(DISTINCT (sid,rid)) FROM export_audit").fetchone()[0])
        split_mismatch = int(con.execute("""SELECT count(*) FROM export_audit e
          LEFT JOIN cplus_membership m USING(sid,rid)
          WHERE m.rid IS NULL OR e.split<>m.split""").fetchone()[0])
        row_hash_mismatch = int(con.execute("""SELECT count(*) FROM export_audit e
          JOIN stage2.main.flows f USING(sid,rid)
          WHERE e.record_hash<>lower(hex(f.record_sha))""").fetchone()[0])
        output_role_counts = {str(role): int(n) for role, n in
            con.execute("SELECT split,count(*) FROM export_audit GROUP BY split").fetchall()}
        output_embargo_rows = output_role_counts.get("embargo", 0)
        expected_retained = {key: EXPECTED_CPLUS[key] for key in ("train", "val", "test")}
        output_role_counts = {key: output_role_counts.get(key, 0) for key in expected_retained}
        per_source_mismatch = int(con.execute("""WITH expected AS (
            SELECT sid,split,count(*) AS n FROM cplus_membership WHERE split<>'embargo' GROUP BY sid,split),
          actual AS (SELECT sid,split,count(*) AS n FROM export_audit GROUP BY sid,split)
          SELECT count(*) FROM expected e FULL OUTER JOIN actual a USING(sid,split)
          WHERE coalesce(e.n,0)<>coalesce(a.n,0)""").fetchone()[0])
        export_bad_flags = int(con.execute("""SELECT count(*) FROM export_audit
          WHERE NOT source_path_ok OR NOT proxy_source_ok OR proxy_label NOT IN ('Benign','Attack')""").fetchone()[0])
        exact_duplicate_crossings = int(con.execute("""SELECT count(*) FROM (
          SELECT d.duplicate_group,count(DISTINCT e.split) roles
          FROM stage2.main.exact_duplicate_members d JOIN export_audit e USING(sid,rid)
          GROUP BY d.duplicate_group HAVING roles>1)""").fetchone()[0])

        # Re-hash original inputs after export; neither bytes nor timestamps may have changed.
        source_hashes_after = {}
        for index, row in enumerate(source_rows, 1):
            sid = int(row["sid"])
            path = Path(row["path"]).resolve()
            hashed = file_signature(path, full_hash=True)
            after_hash = file_signature(path)
            if (hashed.get("sha256") != source_hashes[sid] or
                    hashed.get("sha256") != row["sha256"] or
                    after_hash != source_initial_signatures[sid]):
                raise SplitError(f"Source changed during full split: {path}")
            source_hashes_after[sid] = hashed["sha256"]
            if index % 10 == 0 or index == len(source_rows):
                print(f"Post-export source SHA256 verified: {index}/{len(source_rows)} CSV", flush=True)

        output_rows_ok = output_audit_rows == retained == sum(output_role_counts.values())
        checks = {
            "source_rows_total_19891102": "PASS" if int(preflight["counts"]["train"] + preflight["counts"]["val"] + preflight["counts"]["test"] + preflight["counts"]["embargo"]) == EXPECTED_TOTAL else "FAIL",
            "cplus_role_counts_exact": "PASS" if preflight["counts"] == EXPECTED_CPLUS and output_role_counts == expected_retained else "FAIL",
            "source_row_unique_across_output_splits": "PASS" if actual_output_rows == unique_output_rows == retained and split_mismatch == 0 and per_source_mismatch == 0 else "FAIL",
            "timestamp_buckets_not_cut": "PASS" if preflight["timestamp_bucket_cuts"] == 0 else "FAIL",
            "source_segments_nonoverlap": "PASS" if preflight["segment_cuts"] == 0 and all(f["sort_inversions"] == 0 for f in out_files) else "FAIL",
            "embargo_at_least_511_each_guard": "PASS" if bool((preflight["guards"][["train_val_embargo", "val_test_embargo"]] >= MIN_EMBARGO).all().all()) else "FAIL",
            "activity_crossing_duration_plus_1s": "PASS" if preflight["activity_crossings"] == 0 else "FAIL",
            "no_exact_duplicate_across_splits": "PASS" if exact_duplicate_crossings == 0 and preflight["exact_duplicate_crossings"] == 0 else "FAIL",
            "scenario_and_proxy_counts": "PASS" if output_rows_ok and export_bad_flags == 0 else "FAIL",
            "csv_readback_schema_sort_and_raw_fields": "PASS" if row_hash_mismatch == 0 and all(f["sort_inversions"] == 0 for f in out_files) else "FAIL",
            "source_csv_and_original_row_traceability": "PASS" if split_mismatch == 0 and export_bad_flags == 0 else "FAIL",
            "raw_source_sha256_unchanged": "PASS" if source_hashes_after == source_hashes else "FAIL",
            "no_embargo_rows_exported": "PASS" if output_embargo_rows == 0 and actual_output_rows == retained else "FAIL",
            "future_windows_must_stay_within_each_source_file": "PASS_BY_SOURCE_SPLIT_SEGMENTS",
            "capture_session_independence": "NOT_ESTABLISHED",
            "actual_sliding_window_overlap": "NOT_CREATED_NOT_TESTED",
        }
        failed = [key for key, value in checks.items() if value == "FAIL"]
        if failed:
            raise SplitError(f"Full C+ output validation failed: {failed}")

        actual_scenarios = con.execute("""SELECT s.gid,s.scenario,e.split,count(*)::BIGINT AS flows,
          count(*) FILTER(WHERE s.gid=2)::BIGINT AS benign,
          count(*) FILTER(WHERE s.gid<>2)::BIGINT AS attack
          FROM export_audit e JOIN stage2.main.sources s USING(sid)
          GROUP BY s.gid,s.scenario,e.split ORDER BY s.gid,e.split""").fetchdf()
        actual_labels = con.execute("""SELECT split,count(*)::BIGINT AS flows,
          count(*) FILTER(WHERE gid=2)::BIGINT AS benign,
          count(*) FILTER(WHERE gid<>2)::BIGINT AS attack
          FROM cplus_membership GROUP BY split ORDER BY split""").fetchdf()
        source_manifest = preflight["source_counts"].rename(columns={
            "sid": "Source_CSV_ID", "path": "Source_CSV", "scenario": "Scenario",
            "gid": "Capture_Group_ID", "rows": "Source_Rows", "bytes": "Source_Bytes",
            "sha256": "Source_SHA256", "mtime_ns": "Source_Mtime_NS"})
        source_manifest.to_csv(staging / "source_manifest.csv", index=False)

        manifest_rows = []
        source_by_sid = {int(row["sid"]): row for row in source_rows}
        segment_lookup = {(int(row.sid), str(row.split)): row for row in segments.itertuples(index=False)}
        file_lookup = {(item["sid"], item["split"]): item["relative_path"] for item in out_files}
        for source in source_rows:
            sid = int(source["sid"])
            for role in SPLITS:
                segment = segment_lookup.get((sid, role))
                manifest_rows.append({
                    "Source_CSV_ID": sid, "Source_CSV": source["path"],
                    "Scenario": source["scenario"], "Capture_Group_ID": int(source["gid"]),
                    "Split": role, "Flow_Count": 0 if segment is None else int(segment.flows),
                    "First_Timestamp": "" if segment is None or pd.isna(segment.first_timestamp) else segment.first_timestamp,
                    "Last_Timestamp": "" if segment is None or pd.isna(segment.last_timestamp) else segment.last_timestamp,
                    "Min_Original_Row_Index": "" if segment is None else int(segment.min_original_row_index),
                    "Max_Original_Row_Index": "" if segment is None else int(segment.max_original_row_index),
                    "Output_Relative_Path": file_lookup.get((sid, role), ""),
                    "Proxy_Label": "Benign" if int(source["gid"]) == 2 else "Attack",
                    "Proxy_Label_Source": "SCENARIO_PROXY",
                })
        pd.DataFrame(manifest_rows).to_csv(staging / "split_manifest.csv", index=False)
        scenario_table = preflight["scenario_counts"].copy()
        scenario_names = {int(i): name for i, name in enumerate(GROUP_NAMES, 1)}
        scenario_table["scenario"] = scenario_table["gid"].map(scenario_names)
        scenario_table["proxy_label"] = np.where(scenario_table["gid"].eq(2), "Benign", "Attack")
        scenario_table.to_csv(staging / "scenario_counts_all_roles.csv", index=False)
        actual_labels.to_csv(staging / "proxy_label_counts_all_roles.csv", index=False)
        pd.DataFrame([{"Scenario": name, "Capture_Group_ID": i,
            "Proxy_Label": "Benign" if i == 2 else "Attack",
            "Label_Origin": LABEL_ORIGIN, "Original_Label_Preserved": "No Label"}
            for i, name in enumerate(GROUP_NAMES, 1)]).to_csv(staging / "label_mapping.csv", index=False)

        role_output_bytes = {role: sum(item["path"].stat().st_size for item in out_files if item["split"] == role)
                             for role in ("train", "val", "test")}
        output_bytes = sum(role_output_bytes.values())
        output_file_count = len(out_files)
        validation = {
            "design": "C+", "cutoff_file": str(cutoff_file), "cutoff_sha256": cutoff_signature["sha256"],
            "source_rows": EXPECTED_TOTAL, "flows_by_role": preflight["counts"],
            "retained_rows_read_back": actual_output_rows, "unique_output_source_rows": unique_output_rows,
            "output_flow_counts": output_role_counts,
            "proxy_label_counts": actual_labels.to_dict(orient="records"),
            "scenario_flow_counts": actual_scenarios.to_dict(orient="records"),
            "timestamp_bucket_cuts": preflight["timestamp_bucket_cuts"],
            "segment_cuts": preflight["segment_cuts"],
            "activity_crossings_duration_plus_1s": preflight["activity_crossings"],
            "embargo_rows_found_in_output": output_embargo_rows,
            "exact_duplicate_cross_split_groups": exact_duplicate_crossings,
            "row_fingerprint_mismatches": row_hash_mismatch,
            "source_row_split_mismatches": split_mismatch,
            "per_source_role_count_mismatches": per_source_mismatch,
            "output_file_count": output_file_count,
            "output_bytes_by_split": role_output_bytes, "output_bytes_total": output_bytes,
            "minimum_embargo_flows_by_boundary": preflight["guards"].to_dict(orient="records"),
            "checks": checks, "session_audit_from_stage33": frozen_audit.get("session_candidates_by_boundary"),
            "shared_5tuple_keys": frozen_audit.get("shared_keys_global"),
            "capture_session_independence": "NOT_ESTABLISHED",
            "windows_created": False, "scaler_or_feature_selection_fit": False,
            "models_trained": False,
        }
        (staging / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        shutil.rmtree(scratch)
        (staging / "INCOMPLETE.txt").unlink()
        directory_bytes_total = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        elapsed = time.time() - started
        report_data = {**validation, "elapsed_seconds": elapsed,
            "output_root": str(out), "dataset_root": str(dataset_root),
            "directory_bytes_total": directory_bytes_total,
            "estimated_output_bytes_before_run": output_estimate,
            "estimated_temp_allowance_bytes": temp_allowance,
            "free_disk_bytes_before_run": disk_free,
            "source_sha256_verified_before_and_after": True,
            "original_label_values": ["No Label"],
            "proxy_counts_table": actual_labels.to_dict(orient="records"),
            "scenario_table": scenario_table.to_dict(orient="records"),
            "scenario_proxy_counts": actual_scenarios.to_dict(orient="records"),
            "ctu13_scripts_modified": False}
        report_tmp.write_text(_render_stage34_report(report_data), encoding="utf-8")
        con.close()
        con = None
        staging.rename(out)
        report_tmp.replace(report_path)
        report_data["elapsed_seconds"] = time.time() - started
        print(json.dumps({"status": "FULL_SPLIT_VALIDATED_AND_PUBLISHED",
            "output": str(out), "report": str(report_path),
            "flows": validation["flows_by_role"], "checks": checks,
            "output_bytes": output_bytes, "output_files": output_file_count,
            "elapsed_seconds": report_data["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)
        return report_data
    except BaseException as exc:
        if staging.exists():
            marker = staging / "INCOMPLETE.txt"
            try:
                marker.write_text(f"Split failed; this is not a valid output.\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
            except OSError:
                pass
        raise
    finally:
        if con is not None:
            con.close()


def _render_stage34_report(data: dict) -> str:
    roles = data["flows_by_role"]
    retained = sum(roles[name] for name in ("train", "val", "test"))
    lines = [
        "# Giai đoạn 3.4 — Full split CICIoMT2024 theo C+", "",
        "## Kết quả", "",
        f"Đã xuất và kiểm định toàn bộ **{retained:,}** flow được giữ lại từ **{roles['train'] + roles['val'] + roles['test'] + roles['embargo']:,}** dòng nguồn. Thiết kế cutoff C+ được đọc từ file đã phê duyệt; không tối ưu lại cutoff.", "",
        "| Role | Flow | Tỷ lệ trên toàn bộ nguồn | Tỷ lệ trên phần giữ lại |", "|---|---:|---:|---:|",
    ]
    total = sum(roles.values())
    for role in ("train", "val", "test", "embargo"):
        pct_all = 100 * roles[role] / total
        pct_ret = f"{100 * roles[role] / retained:.6f}%" if role != "embargo" else "—"
        lines.append(f"| {role} | {roles[role]:,} | {pct_all:.6f}% | {pct_ret} |")
    lines.extend(["", "Train/Validation/Test trên 17,997,672 flow giữ lại đạt xấp xỉ **60:20:20**. Embargo không được ghi ra các tập và không bị xóa khỏi CSV gốc.", "",
        "| Thư mục/nhóm output | Dung lượng (bytes) |", "|---|---:|"])
    for role in ("train", "val", "test"):
        lines.append(f"| CSV flow {role} | {int(data['output_bytes_by_split'][role]):,} |")
    lines.extend([
        f"| CSV flow Train/Validation/Test | {int(data['output_bytes_total']):,} |",
        f"| Toàn bộ thư mục sau publish (gồm manifest và validation) | {int(data.get('directory_bytes_total', data['output_bytes_total'])):,} |",
        "",
        "## Nhãn scenario proxy", "",
        "`Label` gốc trong CSV vẫn là `No Label`. Cột `Proxy_Label` được thêm riêng và `Proxy_Label_Source=SCENARIO_PROXY`; đây là nhãn theo kịch bản, không phải ground truth từng flow.", "",
        "| Split | Benign proxy | Attack proxy | Tổng |", "|---|---:|---:|---:|"])
    for row in data["proxy_counts_table"]:
        lines.append(f"| {row['split']} | {int(row['benign']):,} | {int(row['attack']):,} | {int(row['flows']):,} |")
    lines.extend(["", "### Flow theo scenario", "", "| Scenario | Proxy | Train | Validation | Test | Embargo |", "|---|---|---:|---:|---:|---:|"])
    scenario_map = {}
    for row in data["scenario_table"]:
        scenario_map.setdefault(row["scenario"], {})[str(row["split"])] = int(row["flows"])
    for scenario in GROUP_NAMES:
        proxy = "Benign" if scenario == "Benign" else "Attack"
        counts = scenario_map.get(scenario, {})
        lines.append(f"| {scenario} | {proxy} | {counts.get('train',0):,} | {counts.get('val',0):,} | {counts.get('test',0):,} | {counts.get('embargo',0):,} |")
    lines.extend(["", "## Cấu trúc và truy vết", "",
        f"- Thư mục đầu ra: `{data['output_root']}` (train/val/test, phân thư mục theo scenario).",
        "- Mỗi CSV là một segment của đúng một source CSV và một split; từng segment được sort ổn định theo `(Timestamp, Source_CSV_ID, Original_Row_Index)`.",
        "- Mỗi flow xuất ra có `Source_CSV`, `Source_CSV_ID`, `Original_Row_Index`, `Split`, `Proxy_Label`, `Proxy_Label_Source`; `split_manifest.csv` còn ghi cả số flow embargo theo source.",
        "- File kiểm định: `validation.json`; mapping nhãn: `label_mapping.csv`; cutoff snapshot: `Cplus_cutoffs_used.csv`; nguồn: `source_manifest.csv`.",
        "- Không nối các CSV/segment với nhau khi tạo window sau này; xử lý từng file độc lập.", "",
        "## Kiểm định toàn bộ", "", "| Kiểm tra | Kết quả |", "|---|---|"])
    for key, value in data["checks"].items():
        lines.append(f"| `{key}` | **{value}** |")
    lines.extend(["", f"Activity crossing theo `Timestamp + Flow Duration + 1 giây`: **{data['activity_crossings_duration_plus_1s']:,}**. Exact duplicate record đi qua nhiều split: **{data['exact_duplicate_cross_split_groups']:,} nhóm**. Output gồm **{data['output_file_count']:,} file CSV**, **{data['output_bytes_total']:,} bytes**; thời gian chạy khoảng **{data['elapsed_seconds']:.1f} giây**.", "",
        "SHA256 toàn bộ 71 CSV nguồn đã khớp index trước khi chạy và khớp lại sau khi xuất; size/mtime cũng không đổi. SHA256 của cả năm script CTU-13 cũng khớp snapshot Giai đoạn 3.2. Không sửa CSV gốc hoặc năm script CTU-13.", "",
        "## Leakage còn lại và giới hạn", "",
        f"Audit C+ ghi nhận {data.get('shared_5tuple_keys',{}).get('any_shared_key','?'):,} khóa 5-tuple dùng chung qua các split và {sum(int(x.get('session_continuity_candidates_1s',0)) for x in (data.get('session_audit_from_stage33') or [])):,} session candidate theo heuristic 1 giây (không đồng nghĩa session đã xác nhận). Capture/session độc lập chưa được chứng minh. Timestamp có độ phân giải giây và guard dùng Duration + 1 giây; điều đó không chứng minh độc lập capture.",
        "Không tạo sliding window trong giai đoạn này, nên overlap giữa các window chưa được đo. Việc không dùng chung source row được bảo đảm bởi membership duy nhất và segment tách theo source/split; ở bước tiếp theo phải tạo window độc lập trong từng file, không nối qua file, split hoặc capture.", "",
        "## Lệnh PowerShell kiểm tra manifest", "", "```powershell",
        f"$root = '{data['output_root']}'",
        "$m = Import-Csv (Join-Path $root 'split_manifest.csv')",
        "$m | Where-Object Split -ne 'embargo' | Group-Object Split | ForEach-Object {",
        "  [pscustomobject]@{ Split=$_.Name; Flows=(($_.Group | Measure-Object -Property Flow_Count -Sum).Sum) }",
        "} | Sort-Object Split | Format-Table -AutoSize",
        "(Get-Content (Join-Path $root 'validation.json') -Raw | ConvertFrom-Json).checks",
        "Get-ChildItem $root -Recurse -File -Filter '*.csv' | Measure-Object Length -Sum",
        "```", "",
        "## Có thể chuyển bước không?", "",
        "Có thể bắt đầu **thiết kế/triển khai sliding-window riêng theo từng file segment**, sau khi xác nhận proxy label phù hợp mục tiêu nghiên cứu. Chưa thể tuyên bố leakage-free hoàn toàn hoặc capture/session-independent; không chạy feature selection hay mô hình trong Stage 3.4.", ""])
    return "\n".join(lines)


def run_full_metadata_audit(dataset: Path, out: Path, authorize: bool=False, resume: bool=False,
                            expected_total: int=EXPECTED_TOTAL, expected_files: int=EXPECTED_FILES) -> dict:
    """Full metadata only: no raw split export, windows, selectors or classifiers."""
    if not authorize:
        raise SplitError("Full metadata audit requires explicit --authorize-full-metadata")
    import duckdb
    paths=sorted((dataset.resolve()/"CSV").glob("*/*.csv"))
    if len(paths)!=expected_files:
        raise SplitError(f"Expected {expected_files} CSVs, found {len(paths)}")
    if resume:
        code=Path(__file__).resolve().parent
        out=out.resolve()
        if code not in out.parents or not (out/"metadata.duckdb").is_file():
            raise SplitError("Resume requires this module's existing audit database")
    else:
        out=_output_folder(out,dataset)
    started=time.time()
    ctu_dir=Path(__file__).resolve().parent.parent/"CODE_ctu13"
    ctu_hashes={p.name:file_signature(p,True)["sha256"] for p in sorted(ctu_dir.glob("*.py"))}
    provenance=out/"provenance.json"
    if provenance.exists():
        existing=json.loads(provenance.read_text(encoding="utf-8"))
        if existing["ctu13_sha256_before"]!=ctu_hashes or existing["source_paths"]!=[str(p.resolve()) for p in paths]:
            raise SplitError("Resume source paths or CTU13 hashes changed")
    else:
        provenance.write_text(json.dumps({"source_paths":[str(p.resolve()) for p in paths],"ctu13_sha256_before":ctu_hashes,
          "memory_limit":"2GB","temporary_disk_limit":"8GB","format":"DuckDB columnar metadata only",
          "estimated_database_bytes":"1-3 GB; not three copies of flow features",
          "sort_rule":"Scenario group, Timestamp ASC, Source_CSV ASC, Original_Row_Index ASC",
          "fingerprint":"SHA256 of JSON array of all original string fields, fixed column order",
          "timestamp_uncertainty_seconds":1,"duration_unit":"microseconds; standard source and smoke corroboration; local exporter build unverified"},indent=2),encoding="utf-8")
    con=duckdb.connect(str(out/"metadata.duckdb"))
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=2")
        con.execute("SET preserve_insertion_order=true")
        con.execute(f"SET temp_directory={sql_literal(out/'duckdb_temp')}")
        con.execute("SET max_temp_directory_size='8GB'")
        population_schema(con)
        done={int(r[0]):r[1:] for r in con.execute("SELECT sid,path,bytes,mtime_ns FROM sources").fetchall()}
        with paths[0].open("r",encoding="utf-8-sig",newline="") as stream:
            columns=next(csv.reader(stream))
        for sid,path in enumerate(paths):
            if sid in done:
                old_path,old_bytes,old_mtime=done[sid]
                if str(path.resolve())!=old_path or file_signature(path)!={"bytes":old_bytes,"mtime_ns":old_mtime}:
                    raise SplitError("Previously audited source changed")
                continue
            t=time.time()
            record=ingest_population_source(con,path,sid,columns)
            scanned=con.execute("SELECT sum(rows) FROM sources").fetchone()[0]
            print(f"[metadata {sid+1}/{len(paths)}] {path.name}: {record['rows']:,}; total={scanned:,}; seconds={time.time()-t:.1f}",flush=True)
        con.execute("CHECKPOINT")
        total=int(con.execute("SELECT count(*) FROM flows").fetchone()[0])
        if total!=expected_total:
            raise SplitError(f"Full row conservation failed: {total} != {expected_total}")
        duplicate_identity=int(con.execute("SELECT count(*) FROM (SELECT sid,rid,count(*) n FROM flows GROUP BY sid,rid HAVING n<>1)").fetchone()[0])
        range_bad=int(con.execute("SELECT count(*) FROM (SELECT sid,min(rid) lo,max(rid) hi,count(*) n FROM flows GROUP BY sid) WHERE lo<>0 OR hi<>n-1").fetchone()[0])
        quality=con.execute("""SELECT count(*) AS rows,count(*) FILTER(WHERE ts IS NULL) AS invalid_timestamp,
            count(*) FILTER(WHERE duration_us IS NULL) AS invalid_duration,
            count(*) FILTER(WHERE try_cast(json_extract_string(metadata_issue,'$[1]') AS DOUBLE)<0) AS negative_duration,
            count(*) FILTER(WHERE duration_us=0) AS zero_duration,
            count(*) FILTER(WHERE duration_us>120000000) AS duration_above_120s_diagnostic,
            min(duration_us) AS min_duration_us,max(duration_us) AS max_duration_us,
            count(*) FILTER(WHERE ck IS NULL) AS invalid_key,
            count(*) FILTER(WHERE flow_id_consistent IS NOT TRUE) AS flow_id_inconsistent,
            count(*) FILTER(WHERE proto NOT IN(6,17)) AS non_tcp_udp_rows,
            count(*) FILTER(WHERE rate_eligible) AS packet_rate_eligible,
            count(*) FILTER(WHERE rate_eligible AND rate_matches) AS packet_rate_matches_microseconds,
            count(*) FILTER(WHERE metadata_issue IS NOT NULL) AS metadata_issues FROM flows""").fetchdf().to_dict(orient="records")[0]
        print(f"[quality] {json.dumps(quality,default=str)}",flush=True)
        if quality["metadata_issues"]:
            con.execute(f"COPY (SELECT * EXCLUDE(record_sha) FROM flows WHERE metadata_issue IS NOT NULL LIMIT 100) TO {sql_literal(out/'invalid_metadata_examples.csv')} (HEADER,FORMAT CSV)")
            raise SplitError("Invalid metadata found; counts retained in DB, review required before A/B/C decisions")
        # Check all unique endpoint strings with the same ipaddress normalization as smoke.
        ips=con.execute("SELECT DISTINCT unnest(string_split(ck,'|')[2:3]) AS endpoint FROM flows WHERE ck IS NOT NULL").fetchall()
        ip_bad=[]
        for (endpoint,) in ips:
            ip=endpoint.rsplit(':',1)[0]
            try:
                if normalize_ip(ip)!=ip:
                    ip_bad.append(ip)
            except ValueError:
                ip_bad.append(ip)
        if ip_bad:
            raise SplitError(f"Noncanonical/invalid IPs require explicit normalization: {ip_bad[:10]}")
        con.execute(f"""COPY (SELECT s.*,q.first_timestamp,q.last_timestamp,
           q.duration_zero,q.duration_max_us,'INFERRED_NOT_VERIFIED_CAPTURE' AS capture_status
           FROM sources s JOIN (SELECT sid,min(ts) first_timestamp,max(ts) last_timestamp,
           count(*) FILTER(WHERE duration_us=0) duration_zero,max(duration_us) duration_max_us
           FROM flows GROUP BY sid) q USING(sid) ORDER BY sid)
           TO {sql_literal(out/'source_inventory_71.csv')} (HEADER,FORMAT CSV)""")
        print("[duplicates] Hash candidate search, followed by exact original-field verification",flush=True)
        duplicates=verify_full_record_duplicates(con)
        print(f"[duplicates] {json.dumps(duplicates)}",flush=True)
        populate_policies(con)
        con.execute(f"COPY (SELECT * FROM policies ORDER BY design,gid) TO {sql_literal(out/'proposed_cutoffs_ABC.csv')} (HEADER,FORMAT CSV)")
        results=[]
        for design in "ABC":
            print(f"[design {design}] actual population counts, guards, keys and window COUNTS only",flush=True)
            result=audit_population_design(con,design,out)
            results.append(result)
            print(f"[design {design}] {json.dumps({k:result[k] for k in ['flows','percent_retained','activity_crossing_records','shared_keys_global','session_continuity_candidate_group_key_boundaries']},default=str)}",flush=True)
        if expected_total==EXPECTED_TOTAL and results[0]["flows"]!=EXPECTED_B2:
            raise SplitError("Full measured A counts differ from frozen B2 design")
        unchanged=all(file_signature(Path(path))=={"bytes":size,"mtime_ns":mtime} for path,size,mtime in con.execute("SELECT path,bytes,mtime_ns FROM sources").fetchall())
        ctu_after={p.name:file_signature(p,True)["sha256"] for p in sorted(ctu_dir.glob("*.py"))}
        if not unchanged or ctu_after!=ctu_hashes:
            raise SplitError("Source stat or CTU13 SHA256 changed")
        con.execute("CHECKPOINT")
        report={"scope":"FULL_POPULATION_METADATA_AUDIT_NOT_FULL_SPLIT","rows":total,"files":len(paths),
           "quality":quality,"duplicate_identity_groups":duplicate_identity,"original_row_range_errors":range_bad,
           "full_record_duplicates":duplicates,"results":results,
           "raw_sources_unchanged_by_stat":unchanged,"ctu13_sha256_unchanged":ctu_after==ctu_hashes,
           "source_row_conservation":"PASS" if duplicate_identity==0 and range_bad==0 else "FAIL",
           "session_capture_independence":"NOT_ESTABLISHED_SCENARIO_DEPENDENCY_GROUPS_ONLY",
           "session_candidate_definition":"Same scenario + bidirectional key + split-pair, earliest later start <= latest earlier end upper + 1s; transport protocols only. Heuristic, not session count.",
           "windows_created":False,"flow_csv_exported":False,"models_trained":False,
           "database_bytes":(out/"metadata.duckdb").stat().st_size,"elapsed_seconds_this_invocation":time.time()-started}
        (out/"audit_full.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        pd.DataFrame([{"design":r["design"],**r["flows"],**{f"{k}_pct":v for k,v in r["percent_retained"].items()},
           "activity_crossings":r["activity_crossing_records"],"shared_keys":r["shared_keys_global"]["any_shared_key"],
           "session_candidates":r["session_continuity_candidate_group_key_boundaries"]} for r in results]).to_csv(out/"comparison_ABC.csv",index=False)
        return report
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "plan", "review-smoke", "audit-full", "optimize-c", "split-cplus"))
    parser.add_argument("--authorize-full-metadata", action="store_true")
    parser.add_argument("--resume-audit", action="store_true")
    parser.add_argument("--input-run", type=Path, help="Existing bounded smoke; required by review-smoke only")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "stage32_full_metadata",
                        help="Existing metadata-only DuckDB directory for optimize-c/split-cplus")
    parser.add_argument("--cutoffs", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "stage33_cutoff_optimization" / "Cplus_cutoffs.csv")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_ROOT / "CSV",
                        help="Original CICIoMT24 CSV folder for split-cplus")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "STAGE34_REPORT.md")
    parser.add_argument("--max-rows-per-file", type=int, default=3000)
    parser.add_argument("--duration-unit", choices=tuple(UNIT_CODES), default="microseconds")
    parser.add_argument("--duration-evidence", default="", help="Exporter/unit evidence; not an automatic verification.")
    parser.add_argument("--authorize-full-scan", action="store_true", help="Future explicit authorization; never used by smoke.")
    args = parser.parse_args()
    try:
        if args.mode == "optimize-c":
            result=optimize_cplus_from_metadata(args.metadata_dir,args.out)
            print(json.dumps({"scope":result["scope"],"rows_in_duckdb":result["rows_in_duckdb"],
                "candidate_pairs_examined":result["candidate_pairs_examined"],
                "accepted_cutoff_moves":result["accepted_cutoff_moves"],
                "flows":result["optimized_Cplus"]["flows"],
                "percent_retained":result["optimized_Cplus"]["percent_retained"],
                "embargo":result["optimized_Cplus"]["flows"]["embargo"],
                "activity_crossings":result["optimized_Cplus"]["activity_crossing_records"],
                "session_candidates":result["optimized_Cplus"]["session_continuity_candidate_group_key_boundaries"]},
                ensure_ascii=False,indent=2,default=str),flush=True)
            return
        if args.mode == "split-cplus":
            run_full_cplus_split(args.csv_root, args.metadata_dir, args.cutoffs,
                                 args.out, args.report)
            return
        if args.mode == "audit-full":
            run_full_metadata_audit(args.dataset_root,args.out,args.authorize_full_metadata,args.resume_audit)
            return
        if args.mode == "review-smoke":
            if args.input_run is None or args.authorize_full_scan:
                raise SplitError("review-smoke requires --input-run and forbids full-scan authorization")
            run_duration_review(args.input_run, args.out, args.dataset_root)
            return
        run_pipeline(
            args.dataset_root, args.out, args.mode, args.max_rows_per_file,
            args.duration_unit, args.duration_evidence, args.authorize_full_scan,
        )
    except (SplitError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
