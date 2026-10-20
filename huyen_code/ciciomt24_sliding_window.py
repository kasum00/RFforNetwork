"""Memory-bounded CICIoMT2024 adaptation of CTU-13 script 3.

The module creates windows independently inside each validated C+ split CSV.
It deliberately does not modify the original CTU-13 scripts, split data, or
train any model.  A full run is guarded by an explicit --confirm-full flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skew


DEFAULT_INPUT_ROOT = Path(r"D:\COMPACT-MI-RFE\CICIoMT24\CSV_SPLIT_CPLUS")
DEFAULT_MANIFEST = "split_manifest.csv"
DEFAULT_VALIDATION = "validation.json"
SPLITS = ("train", "val", "test")
WINDOW_CONFIGS = ((256, 8), (256, 16), (512, 8), (512, 16))
STATS = ("mean", "std", "median", "skew", "kurtosis")
REFERENCE_RTOL = 1e-6
REFERENCE_ATOL = 1e-8
EXPECTED_SOURCE_ROWS = 19_891_102
EXPECTED_ROLE_COUNTS = {
    "train": 10_798_603,
    "val": 3_599_535,
    "test": 3_599_534,
    "embargo": 1_893_430,
}
EXPECTED_WINDOW_COUNTS = {
    (256, 8): {"train": 1_348_317, "val": 449_343, "test": 449_251},
    (256, 16): {"train": 674_169, "val": 224_677, "test": 224_631},
    (512, 8): {"train": 1_347_101, "val": 448_735, "test": 448_563},
    (512, 16): {"train": 673_559, "val": 224_373, "test": 224_287},
}
ORIGINAL_CT13_SCRIPT_HASHES = {
    "2_labeling.py": "EA8C3468609A99C69E9F2B5AE8B86B7B7B9B55F5C00147884B0AE4D3808D2305",
    "3_sliding_windown_and_aggregate_features.py": "9DF4BFD2BE6AD72DAA943C51265352CFDB92740B1C8429B03C098A4D0662562A",
    "7_senario1_svm.py": "E44BEF8B6A6F35D4160C06DB30691510601DDD1FD31BDB12F681E8831DFA3F0F",
    "[[CTU13]8_experiment2_svm.py": "AADF2D829DD91C67620124AB033E78636C7A9D63C0CCAE73BE799916F88B6664",
    "[[CTU13]9_ex2_rf.py": "6DB8448FC1FD5CD49416B7CE37AC5DA6E71E6DC1554557589D73F73C936E8CB4",
}

# 76 CICFlowMeter behavior fields.  Endpoint, address, time, label and
# provenance columns are intentionally absent from this tuple.
FEATURE_COLUMNS = (
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
    "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max",
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std",
    "Pkt Len Var", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count",
    "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg",
    "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min",
)
PORT_PROTOCOL_COLUMNS = ("Src Port", "Dst Port", "Protocol")
RAW_NUMERIC_COLUMNS = (*PORT_PROTOCOL_COLUMNS, *FEATURE_COLUMNS)

PROVENANCE_COLUMNS = (
    "Source_CSV", "Source_CSV_ID", "Original_Row_Index", "Split",
    "Proxy_Label", "Proxy_Label_Source",
)
EXPECTED_SOURCE_HEADER = (
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Timestamp", *FEATURE_COLUMNS, "Label", *PROVENANCE_COLUMNS,
)
INPUT_COLUMNS = (
    *FEATURE_COLUMNS, "Timestamp", "Label", *PROVENANCE_COLUMNS,
)
OUTPUT_METADATA_COLUMNS = (
    "Window_ID", "Split", "Scenario", "Source_CSV", "Source_CSV_ID",
    "Original_Row_Index_Start", "Original_Row_Index_End",
    "Window_Start_Timestamp", "Window_End_Timestamp", "Window_Index",
    "Num_Flows", "Proxy_Label", "Label_Origin", "Label",
)
OUTPUT_FEATURE_COLUMNS = tuple(
    f"{feature}_{stat}" for feature in FEATURE_COLUMNS for stat in STATS
)
REQUIRED_VALIDATION_CHECKS = (
    "source_rows_total_19891102",
    "cplus_role_counts_exact",
    "source_row_unique_across_output_splits",
    "timestamp_buckets_not_cut",
    "source_segments_nonoverlap",
    "activity_crossing_duration_plus_1s",
    "no_exact_duplicate_across_splits",
    "scenario_and_proxy_counts",
    "csv_readback_schema_sort_and_raw_fields",
    "source_csv_and_original_row_traceability",
    "no_embargo_rows_exported",
)


@dataclass(frozen=True)
class Segment:
    source_csv_id: int
    source_csv: str
    scenario: str
    split: str
    flow_count: int
    first_timestamp: str
    last_timestamp: str
    min_original_row_index: int
    max_original_row_index: int
    output_relative_path: str
    proxy_label: str
    proxy_label_source: str


def window_count(flow_count: int, fw: int, fs: int) -> int:
    """Match CTU-13 range(0, n-fw+1, fs), independently per source segment."""
    if fw <= 0 or fs <= 0:
        raise ValueError("FW and FS must be positive")
    return max(0, (int(flow_count) - fw) // fs + 1)


def output_feature_names(features: Sequence[str] = FEATURE_COLUMNS) -> tuple[str, ...]:
    return tuple(f"{feature}_{stat}" for feature in features for stat in STATS)


def validate_source_header(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise ValueError(f"Empty CSV: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column name in {path}")
    if tuple(header) != EXPECTED_SOURCE_HEADER:
        raise ValueError(
            f"Unexpected schema in {path}: expected {len(EXPECTED_SOURCE_HEADER)} "
            f"columns in canonical order, got {len(header)}"
        )
    if len(FEATURE_COLUMNS) != 76 or len(output_feature_names()) != 380:
        raise AssertionError("Internal CICIoMT feature schema is inconsistent")


def _int_field(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {field!r} is not an integer: {row}") from exc


def load_manifest(input_root: Path) -> list[Segment]:
    manifest_path = input_root / DEFAULT_MANIFEST
    segments: list[Segment] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Source_CSV_ID", "Source_CSV", "Scenario", "Split", "Flow_Count",
            "First_Timestamp", "Last_Timestamp", "Min_Original_Row_Index",
            "Max_Original_Row_Index", "Output_Relative_Path", "Proxy_Label",
            "Proxy_Label_Source",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing split manifest columns: {required - set(reader.fieldnames or [])}")
        for row in reader:
            role = row["Split"].strip().lower()
            count = _int_field(row, "Flow_Count")
            if role not in (*SPLITS, "embargo"):
                raise ValueError(f"Unexpected split role {role!r}")
            if count < 0:
                raise ValueError("Negative Flow_Count in manifest")
            if count == 0 or role == "embargo":
                continue
            if not row["Output_Relative_Path"].strip():
                raise ValueError(f"Missing output path for non-empty segment: {row}")
            segments.append(Segment(
                source_csv_id=_int_field(row, "Source_CSV_ID"),
                source_csv=row["Source_CSV"],
                scenario=row["Scenario"],
                split=role,
                flow_count=count,
                first_timestamp=row["First_Timestamp"],
                last_timestamp=row["Last_Timestamp"],
                min_original_row_index=_int_field(row, "Min_Original_Row_Index"),
                max_original_row_index=_int_field(row, "Max_Original_Row_Index"),
                output_relative_path=row["Output_Relative_Path"],
                proxy_label=row["Proxy_Label"],
                proxy_label_source=row["Proxy_Label_Source"],
            ))
    if len({(s.source_csv_id, s.split) for s in segments}) != len(segments):
        raise ValueError("Duplicate source-id/split segment in manifest")
    return segments


def verify_split_artifacts(input_root: Path, segments: Sequence[Segment]) -> dict:
    validation_path = input_root / DEFAULT_VALIDATION
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if validation.get("source_rows") != EXPECTED_SOURCE_ROWS:
        raise ValueError("Stage 3.4 validation does not report the expected source row total")
    if validation.get("flows_by_role") != EXPECTED_ROLE_COUNTS:
        raise ValueError("Stage 3.4 C+ role counts differ from the approved design")
    role_counts = {split: 0 for split in (*SPLITS, "embargo")}
    with (input_root / DEFAULT_MANIFEST).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role_counts[row["Split"].strip().lower()] += _int_field(row, "Flow_Count")
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"Manifest role totals differ from approved C+: {role_counts}")
    checks = validation.get("checks", {})
    failed = {name: checks.get(name) for name in REQUIRED_VALIDATION_CHECKS if checks.get(name) != "PASS"}
    if failed:
        raise ValueError(f"Stage 3.4 validation has missing/non-PASS safety checks: {failed}")
    if validation.get("windows_created") is not False:
        raise ValueError("Expected Stage 3.4 to have created no windows")
    if any(s.proxy_label_source != "SCENARIO_PROXY" for s in segments):
        raise ValueError("Unexpected proxy-label provenance in split manifest")
    if any(s.proxy_label not in {"Benign", "Attack"} for s in segments):
        raise ValueError("Unexpected Proxy_Label value in split manifest")
    if any(s.split not in SPLITS for s in segments):
        raise ValueError("Embargo or unknown role entered the window input list")
    expected_rows = {"train": EXPECTED_ROLE_COUNTS["train"], "val": EXPECTED_ROLE_COUNTS["val"], "test": EXPECTED_ROLE_COUNTS["test"]}
    actual_rows = {split: sum(s.flow_count for s in segments if s.split == split) for split in SPLITS}
    if actual_rows != expected_rows:
        raise ValueError(f"Non-empty split segment totals differ: {actual_rows}")
    return validation


def expected_window_table(
    segments: Sequence[Segment], *, verify_approved_totals: bool = True
) -> dict[str, dict[str, int]]:
    result = {f"FW{fw}_FS{fs}": {split: 0 for split in SPLITS} for fw, fs in WINDOW_CONFIGS}
    for seg in segments:
        for fw, fs in WINDOW_CONFIGS:
            result[f"FW{fw}_FS{fs}"][seg.split] += window_count(seg.flow_count, fw, fs)
    for fw, fs in WINDOW_CONFIGS:
        key = f"FW{fw}_FS{fs}"
        expected = EXPECTED_WINDOW_COUNTS[(fw, fs)]
        if verify_approved_totals and result[key] != expected:
            raise ValueError(f"Manifest-derived window counts for {key} differ: {result[key]} != {expected}")
    return result


def expected_windows_by_proxy_label(segments: Sequence[Segment]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for fw, fs in WINDOW_CONFIGS:
        key = f"FW{fw}_FS{fs}"
        result[key] = {split: {label: 0 for label in ("Benign", "Attack")} for split in SPLITS}
        for seg in segments:
            result[key][seg.split][seg.proxy_label] += window_count(seg.flow_count, fw, fs)
    return result


def scenario_window_coverage(segments: Sequence[Segment]) -> dict[str, dict[str, list[str]]]:
    scenarios = sorted({seg.scenario for seg in segments})
    result: dict[str, dict[str, list[str]]] = {}
    for fw, fs in WINDOW_CONFIGS:
        key = f"FW{fw}_FS{fs}"
        result[key] = {}
        for split in SPLITS:
            available = {
                seg.scenario for seg in segments
                if seg.split == split and window_count(seg.flow_count, fw, fs) > 0
            }
            result[key][split] = [scenario for scenario in scenarios if scenario not in available]
    return result


def _to_numeric_features(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in FEATURE_COLUMNS:
        try:
            numeric = pd.to_numeric(frame[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric value in expected behavior feature {name!r}") from exc
        columns.append(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
    return np.column_stack(columns)


def compute_window_stats_batch(windows: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the original per-column NumPy/SciPy formula.

    Input shape is (window_count, feature_count, FW).  SciPy defaults are
    deliberately retained: biased skew, Fisher kurtosis, and NaN for constant
    arrays.  The original nan_to_num policy is applied before every statistic.
    """
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_features, FW)")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.mean(values, axis=-1)
        stds = np.std(values, axis=-1, ddof=0)
        medians = np.median(values, axis=-1)
        skews = skew(values, axis=-1, bias=True, nan_policy="propagate")
        kurtoses = kurtosis(values, axis=-1, fisher=True, bias=True, nan_policy="propagate")
    # Preserve SciPy's one-dimensional constant-input behavior from the CTU-13
    # loop.  Axis-wise reductions can round a repeated decimal value's mean by
    # a few ulps and spuriously turn a constant feature into skew=1/kurtosis=-2.
    constant_inputs = np.all(values == values[..., :1], axis=-1)
    skews = np.where(constant_inputs, np.nan, skews)
    kurtoses = np.where(constant_inputs, np.nan, kurtoses)
    return np.stack((means, stds, medians, skews, kurtoses), axis=-1).reshape(
        values.shape[0], values.shape[1] * len(STATS)
    )


def existential_window_labels(attack_mask: np.ndarray, starts: np.ndarray, fw: int) -> np.ndarray:
    labels = np.asarray(attack_mask, dtype=np.bool_)
    if labels.ndim != 1:
        raise ValueError("attack_mask must be one-dimensional")
    if labels.size < fw:
        return np.zeros(len(starts), dtype=np.int8)
    windows = sliding_window_view(labels, fw)
    return windows[np.asarray(starts, dtype=np.int64)].any(axis=-1).astype(np.int8)


def original_reference_features(window: np.ndarray) -> np.ndarray:
    """Test oracle matching script 3 for a matrix shaped (FW, features)."""
    result: list[float] = []
    for column in range(window.shape[1]):
        values = np.nan_to_num(
            window[:, column].astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result.extend((
                np.mean(values), np.std(values), np.median(values),
                skew(values), kurtosis(values),
            ))
    return np.asarray(result, dtype=np.float64)


def _expected_csv_path(input_root: Path, segment: Segment) -> Path:
    root = input_root.resolve()
    candidate = (input_root / segment.output_relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest output path escapes the split root: {segment.output_relative_path}") from exc
    return candidate


def _parse_timestamps(values: Sequence[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(
        pd.Series(values, dtype="string"), format="mixed", dayfirst=True, errors="raise"
    ))


def _validate_chunk_metadata(frame: pd.DataFrame, segment: Segment, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=object), np.empty(0, dtype=np.bool_))
    if not (frame["Label"].astype("string") == "No Label").all():
        raise ValueError(f"Original Label was changed in {path}")
    if not (frame["Split"].astype("string").str.lower() == segment.split).all():
        raise ValueError(f"Split metadata mismatch in {path}")
    if not (frame["Proxy_Label"].astype("string") == segment.proxy_label).all():
        raise ValueError(f"Proxy_Label mismatch in {path}")
    if not (frame["Proxy_Label_Source"].astype("string") == "SCENARIO_PROXY").all():
        raise ValueError(f"Proxy label source mismatch in {path}")
    try:
        ids = pd.to_numeric(frame["Source_CSV_ID"], errors="raise").to_numpy(dtype=np.int64)
        row_ids = pd.to_numeric(frame["Original_Row_Index"], errors="raise").to_numpy(dtype=np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid source provenance in {path}") from exc
    if not np.all(ids == segment.source_csv_id):
        raise ValueError(f"Source_CSV_ID mismatch in {path}")
    source_values = frame["Source_CSV"].astype("string").str.replace("/", "\\", regex=False)
    expected_source = segment.source_csv.replace("/", "\\")
    if not (source_values.str.casefold() == expected_source.casefold()).all():
        raise ValueError(f"Source_CSV mismatch in {path}")
    timestamps = _parse_timestamps(frame["Timestamp"].astype("string"))
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"Timestamps are not sorted in {path}")
    attack = (frame["Proxy_Label"].astype("string") == "Attack").to_numpy(dtype=np.bool_)
    features = _to_numeric_features(frame)
    return features, row_ids, frame["Timestamp"].astype(str).to_numpy(dtype=object), attack


def _window_id(segment: Segment, fw: int, fs: int, window_index: int) -> str:
    return f"FW{fw}_FS{fs}_{segment.split.upper()}_SID{segment.source_csv_id:03d}_WIN{window_index:08d}"


def _reference_targets(segment: Segment) -> dict[tuple[int, int], set[int]]:
    """Choose two reproducible, pseudo-random windows per segment/config."""
    result: dict[tuple[int, int], set[int]] = {}
    for fw, fs in WINDOW_CONFIGS:
        count = window_count(segment.flow_count, fw, fs)
        seed = (segment.source_csv_id * 1_000_003 + fw * 10_007 + fs * 101) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        sample_count = min(2, count)
        result[(fw, fs)] = set(
            rng.choice(count, size=sample_count, replace=False).astype(np.int64).tolist()
        ) if sample_count else set()
    return result


def _build_output_frame(
    stat_matrix: np.ndarray,
    segment: Segment,
    fw: int,
    fs: int,
    window_indices: np.ndarray,
    starts: np.ndarray,
    row_ids: np.ndarray,
    timestamps: np.ndarray,
    attack_mask: np.ndarray,
) -> pd.DataFrame:
    labels = existential_window_labels(attack_mask, starts, fw)
    first_pos = starts
    last_pos = starts + fw - 1
    local_indices = window_indices.astype(np.int64)
    metadata = {
        "Window_ID": [_window_id(segment, fw, fs, int(i)) for i in local_indices],
        "Split": segment.split,
        "Scenario": segment.scenario,
        "Source_CSV": segment.source_csv,
        # Prefix provenance with text so generic numeric-column inference cannot
        # accidentally treat these identifiers as model features.
        "Source_CSV_ID": f"sid{segment.source_csv_id:03d}",
        "Original_Row_Index_Start": [f"row:{int(row_ids[i])}" for i in first_pos],
        "Original_Row_Index_End": [f"row:{int(row_ids[i])}" for i in last_pos],
        "Window_Start_Timestamp": timestamps[first_pos],
        "Window_End_Timestamp": timestamps[last_pos],
        "Window_Index": local_indices,
        "Num_Flows": np.full(len(starts), fw, dtype=np.int16),
        "Proxy_Label": segment.proxy_label,
        "Label_Origin": "SCENARIO_PROXY",
        "Label": labels,
    }
    result = pd.DataFrame(metadata)
    stat_frame = pd.DataFrame(stat_matrix, columns=OUTPUT_FEATURE_COLUMNS)
    return pd.concat((result, stat_frame), axis=1)


def _write_batches_for_buffer(
    *,
    buffer_features: np.ndarray,
    buffer_rows: np.ndarray,
    buffer_timestamps: np.ndarray,
    buffer_attack: np.ndarray,
    buffer_start: int,
    seen_rows: int,
    next_starts: dict[tuple[int, int], int],
    next_window_indices: dict[tuple[int, int], int],
    segment: Segment,
    partial_root: Path,
    batch_windows: int,
    reference_targets: dict[tuple[int, int], set[int]] | None = None,
    reference_results: list[dict] | None = None,
    nonfinite_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, int]:
    output_counts: dict[str, int] = {}
    buffer_end = buffer_start + len(buffer_features)
    for fw, fs in WINDOW_CONFIGS:
        key = (fw, fs)
        next_start = next_starts[key]
        last_start = seen_rows - fw
        if last_start < next_start:
            output_counts[f"FW{fw}_FS{fs}"] = 0
            continue
        starts = np.arange(next_start, last_start + 1, fs, dtype=np.int64)
        if starts.size == 0:
            output_counts[f"FW{fw}_FS{fs}"] = 0
            continue
        if starts[0] < buffer_start or starts[-1] + fw > buffer_end:
            raise AssertionError("Streaming carry buffer lost an eligible window")
        offsets = starts - buffer_start
        window_view = sliding_window_view(buffer_features, fw, axis=0)
        count_for_config = int(starts.size)
        out_dir = partial_root / f"FW_{fw}_FS_{fs}" / segment.split / segment.scenario
        out_path = out_dir / (
            f"sid{segment.source_csv_id:03d}__{Path(segment.output_relative_path).stem}"
            f"__FW{fw}_FS{fs}_windows.csv"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        first_batch = not out_path.exists()
        base_window_index = next_window_indices[key]
        for offset in range(0, len(starts), batch_windows):
            batch_slice = slice(offset, min(offset + batch_windows, len(starts)))
            batch_starts = starts[batch_slice]
            batch_offsets = offsets[batch_slice]
            batch_windows_view = window_view[batch_offsets]
            stats = compute_window_stats_batch(batch_windows_view)
            first_global = batch_starts - buffer_start
            window_indices = base_window_index + np.arange(offset, offset + len(batch_starts), dtype=np.int64)
            if nonfinite_counts is not None:
                stat_key = f"FW{fw}_FS{fs}"
                counts = nonfinite_counts.setdefault(stat_key, {"nan": 0, "inf": 0})
                counts["nan"] += int(np.isnan(stats).sum())
                counts["inf"] += int(np.isinf(stats).sum())
            targets = (reference_targets or {}).get(key, set())
            if targets and reference_results is not None:
                for local_index, global_index in enumerate(window_indices):
                    if int(global_index) not in targets:
                        continue
                    reference = original_reference_features(batch_windows_view[local_index].T)
                    actual_stats = stats[local_index]
                    passed = bool(np.allclose(
                        actual_stats, reference,
                        rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL, equal_nan=True,
                    ))
                    finite = np.isfinite(actual_stats) & np.isfinite(reference)
                    max_abs_diff = (
                        float(np.max(np.abs(actual_stats[finite] - reference[finite])))
                        if finite.any() else 0.0
                    )
                    reference_results.append({
                        "source_csv_id": segment.source_csv_id,
                        "source_csv": segment.source_csv,
                        "scenario": segment.scenario,
                        "split": segment.split,
                        "fw": fw,
                        "fs": fs,
                        "window_index": int(global_index),
                        "passed": passed,
                        "max_abs_diff_finite": max_abs_diff,
                        "nan_mask_equal": bool(np.array_equal(np.isnan(actual_stats), np.isnan(reference))),
                        "inf_mask_equal": bool(np.array_equal(np.isinf(actual_stats), np.isinf(reference))),
                    })
                    if not passed:
                        raise AssertionError(
                            f"Original NumPy/SciPy statistic reference mismatch in "
                            f"sid{segment.source_csv_id} FW{fw}/FS{fs} window {global_index}"
                        )
            output_frame = _build_output_frame(
                stats, segment, fw, fs, window_indices, first_global,
                buffer_rows, buffer_timestamps, buffer_attack,
            )
            output_frame.to_csv(
                out_path, mode="w" if first_batch else "a", header=first_batch,
                index=False, float_format="%.17g",
            )
            first_batch = False
        next_window_indices[key] += count_for_config
        next_starts[key] = int(starts[-1] + fs)
        output_counts[f"FW{fw}_FS{fs}"] = count_for_config
    return output_counts


def process_segment_streaming(
    input_root: Path,
    segment: Segment,
    partial_root: Path,
    *,
    chunksize: int = 50_000,
    batch_windows: int = 64,
    progress_callback=None,
    reference_results: list[dict] | None = None,
    nonfinite_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, int]:
    path = _expected_csv_path(input_root, segment)
    if not path.is_file():
        raise FileNotFoundError(path)
    validate_source_header(path)
    dtype = {
        "Timestamp": "string", "Label": "string", "Source_CSV": "string",
        "Split": "string", "Proxy_Label": "string", "Proxy_Label_Source": "string",
    }
    seen = 0
    first_timestamp = None
    last_timestamp = None
    min_row = math.inf
    max_row = -math.inf
    previous_timestamp = None
    carry_features = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
    carry_rows = np.empty(0, dtype=np.int64)
    carry_timestamps = np.empty(0, dtype=object)
    carry_attack = np.empty(0, dtype=np.bool_)
    next_starts = {(fw, fs): 0 for fw, fs in WINDOW_CONFIGS}
    next_window_indices = {(fw, fs): 0 for fw, fs in WINDOW_CONFIGS}
    reference_targets = _reference_targets(segment) if reference_results is not None else None

    for frame in pd.read_csv(
        path, usecols=list(INPUT_COLUMNS), dtype=dtype, chunksize=chunksize,
        encoding="utf-8-sig", low_memory=False,
    ):
        if frame.empty:
            continue
        chunk_features, chunk_rows, chunk_timestamps, chunk_attack = _validate_chunk_metadata(frame, segment, path)
        parsed = _parse_timestamps(chunk_timestamps)
        if previous_timestamp is not None and parsed[0] < previous_timestamp:
            raise ValueError(f"Timestamp order breaks between chunks in {path}")
        previous_timestamp = parsed[-1]
        if first_timestamp is None:
            first_timestamp = chunk_timestamps[0]
        last_timestamp = chunk_timestamps[-1]
        seen += len(frame)
        min_row = min(min_row, int(chunk_rows.min()))
        max_row = max(max_row, int(chunk_rows.max()))

        if len(carry_features):
            buffer_features = np.concatenate((carry_features, chunk_features), axis=0)
            buffer_rows = np.concatenate((carry_rows, chunk_rows))
            buffer_timestamps = np.concatenate((carry_timestamps, chunk_timestamps))
            buffer_attack = np.concatenate((carry_attack, chunk_attack))
        else:
            buffer_features, buffer_rows = chunk_features, chunk_rows
            buffer_timestamps, buffer_attack = chunk_timestamps, chunk_attack
        buffer_start = seen - len(buffer_features)
        _write_batches_for_buffer(
            buffer_features=buffer_features, buffer_rows=buffer_rows,
            buffer_timestamps=buffer_timestamps, buffer_attack=buffer_attack,
            buffer_start=buffer_start, seen_rows=seen, next_starts=next_starts,
            next_window_indices=next_window_indices, segment=segment,
            partial_root=partial_root, batch_windows=batch_windows,
            reference_targets=reference_targets, reference_results=reference_results,
            nonfinite_counts=nonfinite_counts,
        )
        keep = max(fw for fw, _ in WINDOW_CONFIGS) - 1
        if len(buffer_features) > keep:
            carry_features = buffer_features[-keep:].copy()
            carry_rows = buffer_rows[-keep:].copy()
            carry_timestamps = buffer_timestamps[-keep:].copy()
            carry_attack = buffer_attack[-keep:].copy()
        else:
            carry_features, carry_rows = buffer_features, buffer_rows
            carry_timestamps, carry_attack = buffer_timestamps, buffer_attack

        if progress_callback is not None:
            progress_callback(
                seen,
                {f"FW{fw}_FS{fs}": next_window_indices[(fw, fs)] for fw, fs in WINDOW_CONFIGS},
            )

    if seen != segment.flow_count:
        raise ValueError(f"Read {seen} rows, manifest expects {segment.flow_count}: {path}")
    if (int(min_row), int(max_row)) != (segment.min_original_row_index, segment.max_original_row_index):
        raise ValueError(f"Original_Row_Index bounds mismatch in {path}")
    if _parse_timestamps([first_timestamp])[0] != _parse_timestamps([segment.first_timestamp])[0]:
        raise ValueError(f"First timestamp mismatch against manifest in {path}")
    if _parse_timestamps([last_timestamp])[0] != _parse_timestamps([segment.last_timestamp])[0]:
        raise ValueError(f"Last timestamp mismatch against manifest in {path}")
    actual = {f"FW{fw}_FS{fs}": next_window_indices[(fw, fs)] for fw, fs in WINDOW_CONFIGS}
    expected = {f"FW{fw}_FS{fs}": window_count(segment.flow_count, fw, fs) for fw, fs in WINDOW_CONFIGS}
    if actual != expected:
        raise ValueError(f"Generated segment window counts mismatch: {actual} != {expected}")
    if reference_results is not None:
        expected_samples = sum(len(indices) for indices in (reference_targets or {}).values())
        actual_samples = sum(
            1 for item in reference_results
            if item["source_csv_id"] == segment.source_csv_id and item["split"] == segment.split
        )
        if actual_samples != expected_samples:
            raise AssertionError(
                f"Reference sample count mismatch for sid{segment.source_csv_id}: "
                f"{actual_samples} != {expected_samples}"
            )
    return actual


def _read_smoke_prefix(input_root: Path, segment: Segment, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _expected_csv_path(input_root, segment)
    validate_source_header(path)
    frame = pd.read_csv(
        path, usecols=list(INPUT_COLUMNS), nrows=rows,
        dtype={"Timestamp": "string", "Label": "string", "Source_CSV": "string",
               "Split": "string", "Proxy_Label": "string", "Proxy_Label_Source": "string"},
        encoding="utf-8-sig", low_memory=False,
    )
    arrays = _validate_chunk_metadata(frame, segment, path)
    if len(frame) < max(fw for fw, _ in WINDOW_CONFIGS):
        raise ValueError(f"Smoke prefix is too short to exercise FW=512: {path}")
    return arrays


def audit_raw_numeric_columns(input_root: Path, segment: Segment, rows: int) -> int:
    path = _expected_csv_path(input_root, segment)
    sample = pd.read_csv(
        path, usecols=list(RAW_NUMERIC_COLUMNS), nrows=rows,
        encoding="utf-8-sig", low_memory=False,
    )
    for name in RAW_NUMERIC_COLUMNS:
        try:
            pd.to_numeric(sample[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Expected numeric source column {name!r} is not numeric in {path}") from exc
    if len(RAW_NUMERIC_COLUMNS) != 79 or len(FEATURE_COLUMNS) != 76:
        raise AssertionError("Expected 79 raw numeric CICFlowMeter columns (76 behavior + 3 endpoint/protocol)")
    return len(RAW_NUMERIC_COLUMNS)


def _generate_smoke_outputs(
    segment: Segment,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_root: Path,
    prefix_rows: int,
) -> dict[str, dict[str, int | float | str]]:
    features, row_ids, timestamps, attack = arrays
    summary: dict[str, dict[str, int | float | str]] = {}
    for fw, fs in WINDOW_CONFIGS:
        starts = np.arange(0, max(0, len(features) - fw + 1), fs, dtype=np.int64)
        view = sliding_window_view(features, fw, axis=0)
        stats_rows: list[np.ndarray] = []
        index_rows = np.arange(len(starts), dtype=np.int64)
        for offset in range(0, len(starts), 64):
            stats_rows.append(compute_window_stats_batch(view[starts[offset:offset + 64]]))
        stats = np.concatenate(stats_rows, axis=0) if stats_rows else np.empty((0, len(OUTPUT_FEATURE_COLUMNS)))
        output = _build_output_frame(stats, segment, fw, fs, index_rows, starts, row_ids, timestamps, attack)
        out_dir = output_root / f"FW_{fw}_FS_{fs}" / segment.split / segment.scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sid{segment.source_csv_id:03d}__smoke.csv"
        output.to_csv(out_path, index=False, float_format="%.17g")
        summary[f"FW{fw}_FS{fs}"] = {
            "windows": len(output), "benign_windows": int((output["Label"] == 0).sum()),
            "attack_windows": int((output["Label"] == 1).sum()),
            "columns": len(output.columns), "feature_columns": len(OUTPUT_FEATURE_COLUMNS),
            "output_bytes": out_path.stat().st_size,
            "output_path": str(out_path), "input_prefix_rows": min(prefix_rows, len(features)),
        }
        if len(output) != window_count(len(features), fw, fs):
            raise AssertionError("Smoke output count differs from the CTU-13 range formula")
        if output["Window_ID"].duplicated().any():
            raise AssertionError("Window_ID is not unique within smoke segment/config")
        if not ((output["Num_Flows"] == fw) & (output["Label_Origin"] == "SCENARIO_PROXY")).all():
            raise AssertionError("Smoke metadata validation failed")
    return summary


def run_smoke(input_root: Path, output_root: Path, rows: int = 2048) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output_root}")
    segments = load_manifest(input_root)
    validation = verify_split_artifacts(input_root, segments)
    expected_window_table(segments)
    selected_segments: list[Segment] = []
    for split in SPLITS:
        for label in ("Benign", "Attack"):
            segment = next((
                s for s in segments
                if s.split == split and s.proxy_label == label and s.flow_count >= rows
            ), None)
            if segment is None:
                raise ValueError(f"No {split} segment with at least {rows} rows for {label} smoke")
            selected_segments.append(segment)
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    selected = {}
    selected_keys: dict[tuple[str, str], str] = {}
    for seg in selected_segments:
        audit_raw_numeric_columns(input_root, seg, rows)
        arrays = _read_smoke_prefix(input_root, seg, rows)
        entry_key = f"{seg.split}:{seg.proxy_label}:{seg.scenario}:sid{seg.source_csv_id}"
        selected[entry_key] = _generate_smoke_outputs(seg, arrays, output_root, rows)
        selected_keys[(seg.split, seg.proxy_label)] = entry_key
    smoke_readback_ids = {f"FW{fw}_FS{fs}": set() for fw, fs in WINDOW_CONFIGS}
    expected_numeric = set(OUTPUT_FEATURE_COLUMNS) | {"Window_Index", "Num_Flows", "Label"}
    for segment in selected_segments:
        entry_key = selected_keys[(segment.split, segment.proxy_label)]
        for fw, fs in WINDOW_CONFIGS:
            key = f"FW{fw}_FS{fs}"
            artifact = selected[entry_key][key]
            path = Path(str(artifact["output_path"]))
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                output_header = next(csv.reader(handle), None)
            if output_header != [*OUTPUT_METADATA_COLUMNS, *OUTPUT_FEATURE_COLUMNS]:
                raise AssertionError(f"Smoke output schema/readback mismatch: {path}")
            readback = pd.read_csv(path, low_memory=False)
            numeric_columns = set(readback.select_dtypes(include=[np.number]).columns)
            if numeric_columns != expected_numeric:
                raise AssertionError(f"Numeric metadata leaked into X candidates in {path}: {numeric_columns ^ expected_numeric}")
            expected_label = 0 if segment.proxy_label == "Benign" else 1
            if not (readback["Label"] == expected_label).all():
                raise AssertionError(f"Proxy label readback mismatch in {path}")
            if not (readback["Label_Origin"] == "SCENARIO_PROXY").all():
                raise AssertionError(f"Label_Origin readback mismatch in {path}")
            if not (readback["Split"] == segment.split).all() or (readback["Split"] == "embargo").any():
                raise AssertionError(f"Smoke rows crossed a split boundary in {path}")
            if not (readback["Proxy_Label"] == segment.proxy_label).all():
                raise AssertionError(f"Proxy label provenance mismatch in {path}")
            if not (readback["Scenario"] == segment.scenario).all():
                raise AssertionError(f"Scenario provenance mismatch in {path}")
            if not (readback["Source_CSV_ID"] == f"sid{segment.source_csv_id:03d}").all():
                raise AssertionError(f"Source id provenance mismatch in {path}")
            for provenance_col in ("Original_Row_Index_Start", "Original_Row_Index_End"):
                if not readback[provenance_col].astype(str).str.startswith("row:").all():
                    raise AssertionError(f"Source row provenance is malformed in {path}")
            ids = readback["Window_ID"].astype(str)
            if ids.duplicated().any() or smoke_readback_ids[key].intersection(ids):
                raise AssertionError(f"Window_ID collision in smoke outputs for {key}")
            smoke_readback_ids[key].update(ids)
    elapsed = time.perf_counter() - started
    expected_proxy_windows = expected_windows_by_proxy_label(segments)
    projected_bytes: dict[str, int] = {}
    for fw, fs in WINDOW_CONFIGS:
        key = f"FW{fw}_FS{fs}"
        estimated = 0.0
        for label in ("Benign", "Attack"):
            entry_key = selected_keys[("train", label)]
            artifact = selected[entry_key][key]
            per_window = int(artifact["output_bytes"]) / max(1, int(artifact["windows"]))
            windows = sum(expected_proxy_windows[key][split][label] for split in SPLITS)
            estimated += per_window * windows
        projected_bytes[key] = math.ceil(estimated)
    summary = {
        "status": "PASS",
        "mode": "bounded_smoke_only",
        "full_split_rows_read": False,
        "full_windows_created": False,
        "models_trained": False,
        "scaler_mi_rfe_fit": False,
        "input_source_rows": validation["source_rows"],
        "selected_segments": [
            {"split": s.split, "scenario": s.scenario, "source_csv_id": s.source_csv_id,
             "manifest_rows": s.flow_count, "smoke_rows": min(rows, s.flow_count),
             "proxy_label": s.proxy_label}
            for s in selected_segments
        ],
        "config_outputs": selected,
        "manifest_expected_windows_by_proxy_label": expected_proxy_windows,
        "empirical_projected_bytes_by_config": projected_bytes,
        "empirical_projected_bytes_total": sum(projected_bytes.values()),
        "empirical_projected_bytes_with_25pct_reserve": math.ceil(sum(projected_bytes.values()) * 1.25),
        "elapsed_seconds": elapsed,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_stat_count": len(OUTPUT_FEATURE_COLUMNS),
        "raw_numeric_source_fields_checked": len(RAW_NUMERIC_COLUMNS),
        "metadata_columns": list(OUTPUT_METADATA_COLUMNS),
        "validation_checks": {
            "stage34_artifact_safety_checks": "PASS",
            "all_four_fw_fs_configs_both_proxy_classes_all_three_splits": "PASS",
            "original_label_preserved_as_no_label": "PASS",
            "79_raw_numeric_fields_76_behavior_fields": "PASS_ON_BOUNDED_BENIGN_ATTACK_SAMPLES",
            "window_label_uses_any_attack_flow": "PASS_BY_UNIT_TEST_AND_SMOKE",
            "source_segment_boundaries_respected": "PASS_BY_PER_SEGMENT_OUTPUT",
            "smoke_output_readback_schema_labels_provenance_numeric_metadata": "PASS",
            "full_manifest_row_readback": "NOT_RUN_BY_SMOKE",
        },
    }
    (output_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _tree_sha256(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): _sha256_file(path) for path in files}


def _ctu13_script_hashes(input_root: Path) -> dict[str, str]:
    code_root = input_root.resolve().parents[1] / "CODE_ctu13"
    result = {}
    for name in ORIGINAL_CT13_SCRIPT_HASHES:
        path = code_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Original CTU-13 script missing: {path}")
        result[name] = _sha256_file(path)
    return result


def _expected_window_output_path(root: Path, segment: Segment, fw: int, fs: int) -> Path:
    out_dir = root / f"FW_{fw}_FS_{fs}" / segment.split / segment.scenario
    return out_dir / (
        f"sid{segment.source_csv_id:03d}__{Path(segment.output_relative_path).stem}"
        f"__FW{fw}_FS{fs}_windows.csv"
    )


def validate_full_outputs(
    output_root: Path,
    segments: Sequence[Segment],
    expected: dict[str, dict[str, int]],
    *,
    chunksize: int = 8192,
) -> dict:
    """Read every output CSV back in bounded chunks and validate its contract."""
    expected_paths: set[Path] = set()
    segment_by_path: dict[Path, tuple[Segment, int, int, int]] = {}
    expected_proxy = expected_windows_by_proxy_label(segments)
    for segment in segments:
        for fw, fs in WINDOW_CONFIGS:
            count = window_count(segment.flow_count, fw, fs)
            if not count:
                continue
            path = _expected_window_output_path(output_root, segment, fw, fs)
            expected_paths.add(path.resolve())
            segment_by_path[path.resolve()] = (segment, fw, fs, count)
    actual_paths = {
        path.resolve() for path in output_root.rglob("*.csv")
        if path.name != "reference_stat_samples.csv"
    }
    if actual_paths != expected_paths:
        missing = sorted(map(str, expected_paths - actual_paths))[:5]
        extra = sorted(map(str, actual_paths - expected_paths))[:5]
        raise ValueError(f"Output CSV inventory mismatch; missing={missing}, extra={extra}")

    config_split = {
        f"FW{fw}_FS{fs}": {
            split: {"windows": 0, "benign": 0, "attack": 0, "nan_features": 0, "inf_features": 0}
            for split in SPLITS
        }
        for fw, fs in WINDOW_CONFIGS
    }
    verified_files = 0
    verified_rows = 0
    expected_header = [*OUTPUT_METADATA_COLUMNS, *OUTPUT_FEATURE_COLUMNS]
    feature_dtype = {name: "float64" for name in OUTPUT_FEATURE_COLUMNS}
    dtype = {
        **feature_dtype,
        "Window_ID": "string", "Split": "string", "Scenario": "string",
        "Source_CSV": "string", "Source_CSV_ID": "string",
        "Original_Row_Index_Start": "string", "Original_Row_Index_End": "string",
        "Window_Start_Timestamp": "string", "Window_End_Timestamp": "string",
        "Proxy_Label": "string", "Label_Origin": "string",
    }
    for path in sorted(expected_paths):
        segment, fw, fs, expected_count = segment_by_path[path]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header != expected_header or len(header) != 394:
            raise ValueError(f"Full output schema mismatch (expected 394 columns): {path}")
        key = f"FW{fw}_FS{fs}"
        next_index = 0
        rows_in_file = 0
        label_count = {0: 0, 1: 0}
        for frame in pd.read_csv(path, dtype=dtype, chunksize=chunksize, low_memory=False):
            if list(frame.columns) != expected_header:
                raise ValueError(f"Schema changed while reading output: {path}")
            size = len(frame)
            indices = pd.to_numeric(frame["Window_Index"], errors="raise").to_numpy(dtype=np.int64)
            expected_indices = np.arange(next_index, next_index + size, dtype=np.int64)
            if not np.array_equal(indices, expected_indices):
                raise ValueError(f"Window_Index gap, duplicate, or reordering in {path}")
            ids = frame["Window_ID"].astype("string")
            expected_ids = [
                _window_id(segment, fw, fs, int(index)) for index in expected_indices
            ]
            if not np.array_equal(ids.to_numpy(dtype=str), np.asarray(expected_ids, dtype=str)):
                raise ValueError(f"Window_ID mismatch or duplicate in {path}")
            if not (frame["Split"] == segment.split).all():
                raise ValueError(f"Split boundary violation in {path}")
            if not (frame["Scenario"] == segment.scenario).all():
                raise ValueError(f"Scenario/source segment boundary violation in {path}")
            if not (frame["Source_CSV"].str.replace("/", "\\", regex=False).str.casefold()
                    == segment.source_csv.replace("/", "\\").casefold()).all():
                raise ValueError(f"Source CSV provenance mismatch in {path}")
            if not (frame["Source_CSV_ID"] == f"sid{segment.source_csv_id:03d}").all():
                raise ValueError(f"Source CSV id mismatch in {path}")
            if not (frame["Proxy_Label"] == segment.proxy_label).all() or not (
                frame["Label_Origin"] == "SCENARIO_PROXY"
            ).all():
                raise ValueError(f"Proxy label provenance mismatch in {path}")
            start_rows = frame["Original_Row_Index_Start"].str.extract(r"^row:(\d+)$", expand=False)
            end_rows = frame["Original_Row_Index_End"].str.extract(r"^row:(\d+)$", expand=False)
            if start_rows.isna().any() or end_rows.isna().any():
                raise ValueError(f"Malformed source-row provenance in {path}")
            if not (frame["Num_Flows"] == fw).all():
                raise ValueError(f"Num_Flows mismatch in {path}")
            expected_label = 0 if segment.proxy_label == "Benign" else 1
            if not (frame["Label"] == expected_label).all():
                raise ValueError(f"Existential proxy label mismatch in {path}")
            if frame["Window_Start_Timestamp"].isna().any() or frame["Window_End_Timestamp"].isna().any():
                raise ValueError(f"Missing window timestamp provenance in {path}")
            numeric_columns = set(frame.select_dtypes(include=[np.number]).columns)
            expected_numeric = set(OUTPUT_FEATURE_COLUMNS) | {"Window_Index", "Num_Flows", "Label"}
            if numeric_columns != expected_numeric:
                raise ValueError(
                    f"Numeric feature/metadata schema mismatch in {path}: "
                    f"{numeric_columns ^ expected_numeric}"
                )
            start_row_values = start_rows.to_numpy(dtype=np.int64)
            end_row_values = end_rows.to_numpy(dtype=np.int64)
            if (
                (start_row_values < segment.min_original_row_index).any()
                or (start_row_values > segment.max_original_row_index).any()
                or (end_row_values < segment.min_original_row_index).any()
                or (end_row_values > segment.max_original_row_index).any()
            ):
                raise ValueError(f"Source row provenance is outside the manifest segment in {path}")
            feature_values = frame.loc[:, list(OUTPUT_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            values = config_split[key][segment.split]
            values["nan_features"] += int(np.isnan(feature_values).sum())
            values["inf_features"] += int(np.isinf(feature_values).sum())
            values["windows"] += size
            label_count[expected_label] += size
            next_index += size
            rows_in_file += size
        if rows_in_file != expected_count:
            raise ValueError(f"Output rows for {path} are {rows_in_file}, expected {expected_count}")
        if label_count[0] + label_count[1] != expected_count:
            raise AssertionError(f"Label count does not cover all windows for {path}")
        config_split[key][segment.split]["benign"] += label_count[0]
        config_split[key][segment.split]["attack"] += label_count[1]
        verified_files += 1
        verified_rows += rows_in_file

    expected_label_counts = {
        key: {
            split: {
                "windows": expected[key][split],
                "benign": expected_proxy[key][split]["Benign"],
                "attack": expected_proxy[key][split]["Attack"],
            }
            for split in SPLITS
        }
        for key in expected
    }
    for key in expected:
        for split in SPLITS:
            actual_row = config_split[key][split]
            target = expected_label_counts[key][split]
            if any(actual_row[name] != target[name] for name in ("windows", "benign", "attack")):
                raise ValueError(f"Window or label total mismatch for {key}/{split}: {actual_row} != {target}")
    return {
        "status": "PASS",
        "verified_files": verified_files,
        "verified_rows": verified_rows,
        "counts_by_config_split": config_split,
        "expected_counts_by_config_split": expected_label_counts,
        "schema_columns": len(expected_header),
        "feature_columns": len(OUTPUT_FEATURE_COLUMNS),
        "window_id_uniqueness": (
            "PASS: manifest has unique (Source_CSV_ID, Split); each segment's Window_Index is "
            "strictly contiguous and every ID equals the deterministic config/split/source/index key"
        ),
        "segment_boundary_check": "PASS: each CSV is checked against one manifest segment and has constant split/scenario/source ID",
    }


def run_full(
    input_root: Path,
    output_root: Path,
    chunksize: int,
    batch_windows: int,
    *,
    resume_incomplete: bool = False,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if chunksize < 512 or batch_windows < 1:
        raise ValueError("chunksize must be >=512 and batch_windows must be positive")
    segments = load_manifest(input_root)
    validation = verify_split_artifacts(input_root, segments)
    expected = expected_window_table(segments)
    total_windows = sum(sum(counts.values()) for counts in expected.values())
    minimum_free_bytes = math.ceil(total_windows * 11_000 * 1.20)
    disk_anchor = output_root.parent
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    free_bytes = shutil.disk_usage(disk_anchor).free
    if free_bytes < minimum_free_bytes:
        raise OSError(
            f"Insufficient free space for conservative preflight: {free_bytes:,} free, "
            f"at least {minimum_free_bytes:,} required (11,000 bytes/window + 20% reserve)"
        )
    staged = output_root.with_name(output_root.name + ".incomplete")
    if staged.exists() and not resume_incomplete:
        raise FileExistsError(f"Refusing to reuse an existing incomplete directory: {staged}")
    if resume_incomplete and not staged.is_dir():
        raise FileNotFoundError(f"No incomplete run exists to resume: {staged}")
    scripts_before = _ctu13_script_hashes(input_root)
    script_mismatches = {
        name: {"expected": ORIGINAL_CT13_SCRIPT_HASHES[name], "actual": scripts_before[name]}
        for name in ORIGINAL_CT13_SCRIPT_HASHES
        if scripts_before[name] != ORIGINAL_CT13_SCRIPT_HASHES[name]
    }
    if script_mismatches:
        raise ValueError(f"Original CTU-13 script hash preflight failed: {script_mismatches}")
    resume_completed = 0
    previous_attempt_elapsed = 0.0
    resume_archive = None
    prefix_validation = None
    actual = {f"FW{fw}_FS{fs}": {split: 0 for split in SPLITS} for fw, fs in WINDOW_CONFIGS}
    nonfinite_generated: dict[str, dict[str, int]] = {}
    if resume_incomplete:
        progress_path = staged / "progress.json"
        failed_path = staged / "FAILED.json"
        if not progress_path.is_file():
            raise ValueError("Resume requires progress.json in the incomplete directory")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        failure_was_recorded = failed_path.is_file()
        if failure_was_recorded:
            previous_failure = json.loads(failed_path.read_text(encoding="utf-8"))
        elif progress.get("status") == "RUNNING":
            # A user may stop the process between progress checkpoints. In that case
            # the last atomic checkpoint is valid, but the exception handler never
            # got a chance to write FAILED.json.
            previous_failure = {
                "status": "INTERRUPTED_WITHOUT_FAILURE_MARKER",
                "reason": "Resuming a RUNNING checkpoint after confirming no generator process is active",
                "elapsed_seconds": 0.0,
            }
        else:
            raise ValueError("Resume without FAILED.json is allowed only for a RUNNING checkpoint")
        resume_completed = int(progress.get("completed_segments", -1))
        if not 0 <= resume_completed < len(segments):
            raise ValueError(f"Invalid completed-segment checkpoint for resume: {resume_completed}")
        current_segment = segments[resume_completed]
        current_progress = progress.get("current_segment")
        if current_progress:
            if (
                int(current_progress.get("source_csv_id", -1)) != current_segment.source_csv_id
                or current_progress.get("split") != current_segment.split
            ):
                raise ValueError("Incomplete checkpoint does not point to the next manifest segment")
        elif resume_completed:
            last_completed = progress.get("last_completed_segment") or {}
            previous_segment = segments[resume_completed - 1]
            if (
                int(last_completed.get("source_csv_id", -1)) != previous_segment.source_csv_id
                or last_completed.get("split") != previous_segment.split
            ):
                raise ValueError("Stale checkpoint does not identify the completed manifest prefix")
        input_hashes_before = json.loads((staged / "input_sha256_before.json").read_text(encoding="utf-8"))
        stored_script_hashes = json.loads((staged / "ctu13_sha256_before.json").read_text(encoding="utf-8"))
        if stored_script_hashes != scripts_before:
            raise ValueError("CTU-13 script hashes differ from the original incomplete run snapshot")
        if _tree_sha256(input_root) != input_hashes_before:
            raise ValueError("C+ input tree changed since the interrupted run; refusing resume")
        resume_archive = output_root.with_name(
            f"{output_root.name}.partial_segment_{current_segment.source_csv_id:03d}_{int(time.time())}"
        )
        if resume_archive.exists():
            raise FileExistsError(f"Partial-segment archive already exists: {resume_archive}")
        prior_indices = []
        for path in (*staged.glob("prior_failure_attempt_*.json"), *staged.glob("resume_record_attempt_*.json")):
            match = re.search(r"attempt_(\d+)\.json$", path.name)
            if match:
                prior_indices.append(int(match.group(1)))
        if (staged / "resume_record.json").exists() or (staged / "prior_failure_attempt_01.json").exists():
            prior_indices.append(1)
        prior_failure_index = max(prior_indices, default=0) + 1
        prior_failure_path = staged / f"prior_failure_attempt_{prior_failure_index:02d}.json"
        moved_partial_files = []
        for fw, fs in WINDOW_CONFIGS:
            partial_path = _expected_window_output_path(staged, current_segment, fw, fs)
            if partial_path.exists():
                archived_path = resume_archive / partial_path.relative_to(staged)
                archived_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.rename(archived_path)
                moved_partial_files.append(str(archived_path))
        previous_attempt_elapsed = float(previous_failure.get("elapsed_seconds", 0.0)) + sum(
            float(json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
            for path in staged.glob("prior_failure*.json")
        )
        prefix_segments = segments[:resume_completed]
        prefix_expected = expected_window_table(prefix_segments, verify_approved_totals=False)
        prefix_validation = validate_full_outputs(staged, prefix_segments, prefix_expected)
        actual = prefix_expected
        for key in prefix_validation["counts_by_config_split"]:
            nonfinite_generated[key] = {
                "nan": sum(
                    prefix_validation["counts_by_config_split"][key][split]["nan_features"]
                    for split in SPLITS
                ),
                "inf": sum(
                    prefix_validation["counts_by_config_split"][key][split]["inf_features"]
                    for split in SPLITS
                ),
            }
        if failure_was_recorded:
            failed_path.rename(prior_failure_path)
        resume_record_path = (
            staged / "resume_record.json" if prior_failure_index == 1
            else staged / f"resume_record_attempt_{prior_failure_index:02d}.json"
        )
        _write_json_atomic(resume_record_path, {
            "resumed_at_unix_time": time.time(),
            "completed_segments_verified": resume_completed,
            "current_segment_restarted_from_row_zero": {
                "source_csv_id": current_segment.source_csv_id,
                "source_csv": current_segment.source_csv,
                "scenario": current_segment.scenario,
                "split": current_segment.split,
            },
            "prior_run_state": previous_failure,
            "failure_marker_was_present": failure_was_recorded,
            "archived_partial_segment_files": moved_partial_files,
            "partial_segment_archive": str(resume_archive),
            "completed_prefix_readback": "PASS",
            "cplus_input_hashes_match_original_run_snapshot": True,
        })
        print(
            f"RESUME verified {resume_completed} completed segments; "
            f"archived partial sid{current_segment.source_csv_id:03d} outputs at {resume_archive}",
            flush=True,
        )
    else:
        input_hashes_before = _tree_sha256(input_root)
        staged.mkdir(parents=True)
        _write_json_atomic(staged / "input_sha256_before.json", input_hashes_before)
        _write_json_atomic(staged / "ctu13_sha256_before.json", scripts_before)
    started = time.perf_counter()
    reference_results: list[dict] = []
    peak_rss_bytes = 0
    try:
        total_segments = len(segments)
        for index, segment in enumerate(segments[resume_completed:], start=resume_completed + 1):
            print(f"[{index}/{total_segments}] {segment.split} {segment.scenario} sid={segment.source_csv_id} rows={segment.flow_count}", flush=True)

            def checkpoint(rows_read: int, local_counts: dict[str, int]) -> None:
                nonlocal peak_rss_bytes
                try:
                    import psutil  # type: ignore
                    peak_rss_bytes = max(peak_rss_bytes, int(psutil.Process().memory_info().rss))
                except Exception:
                    pass
                generated_so_far = {
                    key: {
                        split: actual[key][split] + (local_counts[key] if split == segment.split else 0)
                        for split in SPLITS
                    }
                    for key in actual
                }
                free_now = shutil.disk_usage(disk_anchor).free
                windows_written = sum(sum(counts.values()) for counts in generated_so_far.values())
                windows_remaining = total_windows - windows_written
                required_remaining = math.ceil(windows_remaining * 11_000 * 1.20)
                if free_now < required_remaining:
                    raise OSError(
                        f"Free space fell below conservative remaining-output reserve: "
                        f"{free_now:,} free, {required_remaining:,} required"
                    )
                state = {
                    "status": "RUNNING",
                    "completed_segments": index - 1,
                    "total_segments": total_segments,
                    "current_segment": {
                        "source_csv_id": segment.source_csv_id,
                        "source_csv": segment.source_csv,
                        "scenario": segment.scenario,
                        "split": segment.split,
                        "rows_expected": segment.flow_count,
                        "rows_read": rows_read,
                    },
                    "windows_generated_by_config_split": generated_so_far,
                    "windows_generated_total": windows_written,
                    "windows_expected_total": total_windows,
                    "free_bytes": free_now,
                    "conservative_free_bytes_required_for_remaining": required_remaining,
                    "peak_rss_bytes_sampled": peak_rss_bytes,
                    "updated_unix_time": time.time(),
                }
                _write_json_atomic(staged / "progress.json", state)
                print(
                    f"  progress rows={rows_read:,}/{segment.flow_count:,}; "
                    f"windows={windows_written:,}/{total_windows:,}; free={free_now:,} bytes",
                    flush=True,
                )

            counts = process_segment_streaming(
                input_root, segment, staged, chunksize=chunksize, batch_windows=batch_windows,
                progress_callback=checkpoint,
                reference_results=reference_results,
                nonfinite_counts=nonfinite_generated,
            )
            for key, count in counts.items():
                actual[key][segment.split] += count
            _write_json_atomic(staged / "progress.json", {
                "status": "RUNNING",
                "completed_segments": index,
                "total_segments": total_segments,
                "last_completed_segment": {
                    "source_csv_id": segment.source_csv_id,
                    "source_csv": segment.source_csv,
                    "scenario": segment.scenario,
                    "split": segment.split,
                    "rows_read": segment.flow_count,
                },
                "windows_generated_by_config_split": actual,
                "windows_generated_total": sum(sum(values.values()) for values in actual.values()),
                "windows_expected_total": total_windows,
                "peak_rss_bytes_sampled": peak_rss_bytes,
                "updated_unix_time": time.time(),
            })
        if actual != expected:
            raise ValueError(f"Full generated window totals differ from manifest: {actual} != {expected}")
        expected_reference_samples = sum(
            sum(len(targets) for targets in _reference_targets(segment).values())
            for segment in segments[resume_completed:]
        )
        if len(reference_results) != expected_reference_samples or not all(
            sample["passed"] and sample["nan_mask_equal"] and sample["inf_mask_equal"]
            for sample in reference_results
        ):
            raise AssertionError(
                f"Random statistic reference audit failed: {len(reference_results)} samples, "
                f"expected {expected_reference_samples}"
            )
        reference_frame = pd.DataFrame(reference_results).sort_values(
            ["split", "source_csv_id", "fw", "fs", "window_index"]
        )
        reference_frame.to_csv(staged / "reference_stat_samples.csv", index=False)
        readback_validation = validate_full_outputs(staged, segments, expected)
        for key in nonfinite_generated:
            readback_totals = {
                metric: sum(readback_validation["counts_by_config_split"][key][split][metric] for split in SPLITS)
                for metric in ("nan_features", "inf_features")
            }
            generated_totals = {
                "nan_features": nonfinite_generated[key]["nan"],
                "inf_features": nonfinite_generated[key]["inf"],
            }
            if generated_totals != readback_totals:
                raise ValueError(
                    f"Generation/readback NaN/Inf counts differ for {key}: "
                    f"{generated_totals} != {readback_totals}"
                )
        input_hashes_after = _tree_sha256(input_root)
        input_integrity = input_hashes_before == input_hashes_after
        _write_json_atomic(staged / "input_sha256_after.json", input_hashes_after)
        scripts_after = _ctu13_script_hashes(input_root)
        scripts_integrity = scripts_before == scripts_after == ORIGINAL_CT13_SCRIPT_HASHES
        _write_json_atomic(staged / "ctu13_sha256_after.json", scripts_after)
        if not input_integrity:
            changed = sorted(set(input_hashes_before) ^ set(input_hashes_after))
            changed.extend(
                name for name in set(input_hashes_before) & set(input_hashes_after)
                if input_hashes_before[name] != input_hashes_after[name]
            )
            raise ValueError(f"C+ input tree changed during window generation: {changed[:20]}")
        if not scripts_integrity:
            raise ValueError("One or more original CTU-13 script hashes changed during generation")
        csv_bytes = sum(path.stat().st_size for path in staged.rglob("*windows.csv"))
        total_output_bytes = sum(path.stat().st_size for path in staged.rglob("*") if path.is_file())
        summary = {
            "status": "PASS",
            "design": "C+ validated split; windows generated independently per split CSV",
            "input_source_rows": validation["source_rows"],
            "segment_count": len(segments),
            "window_counts": actual,
            "proxy_window_counts": readback_validation["expected_counts_by_config_split"],
            "readback_validation": readback_validation,
            "generation_nonfinite_counts": nonfinite_generated,
            "random_reference_stat_samples": {
                "count": len(reference_results),
                "passed": sum(bool(sample["passed"]) for sample in reference_results),
                "rtol": REFERENCE_RTOL,
                "atol": REFERENCE_ATOL,
                "segments_sampled_this_attempt": len(segments) - resume_completed,
                "previous_completed_segments_reference_checks": {
                    "count_passed": sum(
                        sum(len(targets) for targets in _reference_targets(segment).values())
                        for segment in segments[:resume_completed]
                    ),
                    "status": (
                        "PASS_IN_PRIOR_ATTEMPT; per-segment assertion passed before checkpoint"
                        if resume_completed else "NOT_APPLICABLE"
                    ),
                    "sample_rows_persisted": False,
                },
                "max_abs_diff_finite": max(
                    (sample["max_abs_diff_finite"] for sample in reference_results), default=0.0
                ),
                "sample_file": "reference_stat_samples.csv",
            },
            "data_integrity": {
                "cplus_input_tree_sha256_before_after_match": input_integrity,
                "ctu13_scripts_sha256_match_approved_snapshot": scripts_integrity,
                "ctu13_script_hashes": scripts_after,
            },
            "resume_archive": str(resume_archive) if resume_archive else None,
            "output_size_bytes": {"window_csv": csv_bytes, "all_files": total_output_bytes},
            "disk_preflight": {
                "free_bytes_at_start": free_bytes,
                "minimum_required_bytes": minimum_free_bytes,
                "basis": "11,000 bytes/window with 20% reserve",
            },
            "elapsed_seconds_current_attempt_including_full_readback_and_hashes": time.perf_counter() - started,
            "elapsed_seconds_prior_failed_attempt": previous_attempt_elapsed,
            "elapsed_seconds_all_attempts": previous_attempt_elapsed + time.perf_counter() - started,
            "peak_rss_bytes_sampled": peak_rss_bytes or None,
            "feature_columns": len(OUTPUT_FEATURE_COLUMNS),
            "feature_names": list(OUTPUT_FEATURE_COLUMNS),
            "output_columns": len(OUTPUT_METADATA_COLUMNS) + len(OUTPUT_FEATURE_COLUMNS),
            "models_trained": False,
            "scaler_mi_rfe_fit": False,
            "provenance_policy": "source path + textual source id + start/end original row index",
            "validation_checks": {
                "manifest_and_validation_json": "PASS",
                "expected_counts_all_12_config_split_cells": "PASS",
                "output_csv_full_readback_and_schema_394": "PASS",
                "feature_count_380": "PASS",
                "window_id_unique_deterministic_and_contiguous": "PASS",
                "source_segment_split_boundaries": "PASS",
                "source_row_provenance_present": "PASS",
                "proxy_label_counts": "PASS",
                "nan_inf_generation_matches_full_readback": "PASS",
                "random_numpy_scipy_reference_samples": "PASS",
                "cplus_input_unchanged_sha256": "PASS",
                "five_ctu13_script_hashes_unchanged": "PASS",
                "no_window_crosses_source_segment_or_split": "PASS_BY_PER_SEGMENT_STREAMING_AND_READBACK",
                "capture_session_independence": "NOT_ESTABLISHED",
            },
            "capture_session_independence": "NOT_ESTABLISHED_BY_SPLIT_OR_WINDOWING",
        }
        _write_json_atomic(staged / "window_run_summary.json", summary)
        _write_json_atomic(staged / "progress.json", {
            "status": "VALIDATED_COMPLETE",
            "completed_segments": len(segments),
            "total_segments": len(segments),
            "windows_generated_by_config_split": actual,
            "windows_generated_total": total_windows,
            "peak_rss_bytes_sampled": peak_rss_bytes,
            "updated_unix_time": time.time(),
        })
        os.replace(staged, output_root)
        return summary
    except BaseException as exc:
        failure = {
            "status": "INCOMPLETE_DO_NOT_USE",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }
        try:
            _write_json_atomic(staged / "FAILED.json", failure)
        except Exception:
            pass
        raise


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-manifest", help="Verify approved C+ metadata and expected window counts")
    audit.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    smoke = sub.add_parser("smoke", help="Run a small Benign/Attack smoke test, not the full dataset")
    smoke.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--rows", type=int, default=2048)
    full = sub.add_parser("run", help="Generate all windows in bounded-memory streaming mode")
    full.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    full.add_argument("--output-root", type=Path, required=True)
    full.add_argument("--chunksize", type=int, default=50_000)
    full.add_argument("--batch-windows", type=int, default=64)
    full.add_argument("--confirm-full", action="store_true", help="Required acknowledgement before full output is written")
    full.add_argument("--resume-incomplete", action="store_true", help="Resume a failed run after validating and preserving its partial state")
    args = parser.parse_args()
    if args.command == "audit-manifest":
        segments = load_manifest(args.input_root)
        validation = verify_split_artifacts(args.input_root, segments)
        print(json.dumps({
            "status": "PASS", "segments": len(segments),
            "window_counts": expected_window_table(segments),
            "window_counts_by_proxy_label": expected_windows_by_proxy_label(segments),
            "scenarios_without_validation_test_windows": scenario_window_coverage(segments),
            "input_source_rows": validation["source_rows"],
        }, indent=2))
    elif args.command == "smoke":
        print(json.dumps(run_smoke(args.input_root, args.output_root, args.rows), indent=2))
    elif args.command == "run":
        if not args.confirm_full:
            parser.error("Full generation requires --confirm-full after explicit user approval")
        print(json.dumps(run_full(
            args.input_root, args.output_root, args.chunksize, args.batch_windows,
            resume_incomplete=args.resume_incomplete,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
