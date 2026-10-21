#!/usr/bin/env python3
"""
STAGE 1 - CICFlowMeter dataset audit with safe checkpoint/resume.

This program is intentionally audit-only:
  * reads CSV files with pandas chunks;
  * never edits, moves, or deletes source CSV files;
  * does not create train/validation/test splits;
  * does not create sliding windows;
  * does not train a model.

The checkpoint is written after every fully completed CSV.  If the process is
stopped with Ctrl+C, rerun the exact same command.  Completed files are
skipped and any partial duplicate-index rows belonging to an unfinished file
are removed before that file is rescanned.

The duplicate index is an intermediate SQLite file in --output-root. It is
hash-based (64-bit pandas row hashes), so its results are reported as
candidate duplicates and should be confirmed before making modeling decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_CLASSES = [
    "BENIGN",
    "BRUTE_FORCE",
    "DDOS",
    "DOS",
    "MIRAI",
    "RECON",
    "SPOOFING",
    "WEB-BASED",
]

LABEL_CANDIDATES = {
    "label",
    "class",
    "target",
    "attack",
    "attack_label",
    "traffic_label",
}

MISSING_TEXT = {"", "na", "n/a", "nan", "null", "none", "missing"}
INF_TEXT = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_dump(value) + "\n")


def clean_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_header(value).lower()).strip("_")


def unique_headers(headers: Sequence[str]) -> List[str]:
    counts: Counter[str] = Counter()
    result: List[str] = []
    for index, header in enumerate(headers):
        base = header if header else f"__unnamed_{index + 1}__"
        counts[base] += 1
        suffix = "" if counts[base] == 1 else f"__DUP{counts[base] - 1}"
        result.append(base + suffix)
    return result


def detect_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for line in handle:
                if line.strip():
                    candidates = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t")}
                    return max(candidates, key=candidates.get) if max(candidates.values()) else ","
    except OSError:
        pass
    return ","


def read_header(path: Path, delimiter: str) -> List[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            return [normalize_header(value) for value in next(reader)]
        except StopIteration:
            return []


def schema_hash(headers: Sequence[str]) -> str:
    payload = json.dumps(list(headers), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def file_id_for(relative_path: str) -> str:
    return hashlib.sha1(relative_path.lower().encode("utf-8")).hexdigest()[:20]


def dataset_signature(root: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().lower()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def label_indexes(headers: Sequence[str]) -> List[int]:
    result = []
    for index, header in enumerate(headers):
        key = header_key(header)
        if key in LABEL_CANDIDATES or key.endswith("_label"):
            result.append(index)
    return result


def metadata_risk(header: str) -> Tuple[str, str]:
    key = header_key(header)
    compact = key.replace("_", "")
    if key in {"flow_id", "flowid", "id"} or compact == "flowid":
        return "HIGH", "Flow identifier can link rows from the same flow or capture."
    if any(token in key for token in ("timestamp", "time", "date")):
        return "HIGH", "Temporal ordering or session overlap can leak across a split."
    if any(token in key for token in ("src_ip", "dst_ip", "source_ip", "destination_ip", "srcip", "dstip")):
        return "HIGH", "IP identity can encode host, attack scenario, or capture membership."
    if any(token in key for token in ("mac", "host", "hostname", "device", "user", "filename", "file_name")):
        return "HIGH", "Host/device/file identity can act as a scenario identifier."
    if key in LABEL_CANDIDATES or key.endswith("_label"):
        return "CRITICAL", "This column is the target label and must not be used as a feature."
    if any(token in key for token in ("session", "pcap", "capture", "scenario", "attack_type")):
        return "HIGH", "Session/capture/scenario metadata can leak the target or split membership."
    return "NONE", ""


def timestamp_indexes(headers: Sequence[str]) -> List[int]:
    indexes = []
    for index, header in enumerate(headers):
        key = header_key(header)
        if "timestamp" in key or key in {"time", "datetime", "date"} or key.endswith("_time"):
            indexes.append(index)
    return indexes


def candidate_family(relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.stem.lower()
    stem = re.sub(r"[\s._-]+", "_", stem).strip("_")
    # Conservative: only remove explicit part/chunk/split suffixes.
    stem = re.sub(r"(?:_?(?:part|chunk|split|segment|shard|file))?_?\d+$", "", stem)
    parent = path.parent.as_posix().lower()
    return f"{parent}/{stem}" if parent not in {"", "."} else stem


def candidate_group_id(family: str) -> str:
    return "CANDIDATE_" + hashlib.sha1(family.encode("utf-8")).hexdigest()[:14]


def find_sidecars(path: Path) -> List[str]:
    sidecars = []
    for extension in (".pcap", ".pcapng", ".json", ".yaml", ".yml", ".manifest", ".txt"):
        candidate = path.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            sidecars.append(str(candidate))
    return sidecars


def count_inf_values(frame: pd.DataFrame) -> int:
    total = 0
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            total += int(np.isinf(numeric).sum())
        else:
            text = series.astype("string").str.strip().str.lower()
            total += int(text.isin(INF_TEXT).sum())
    return total


def parse_timestamps(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


def update_quality(quality: Dict[str, Dict[str, Any]], frame: pd.DataFrame) -> None:
    for column in frame.columns:
        item = quality.setdefault(
            column,
            {
                "dtype_counts": {},
                "chunks_seen": 0,
                "non_null_cells_sampled": 0,
                "missing_cells_sampled": 0,
                "inf_cells_sampled": 0,
                "sample_values": [],
            },
        )
        dtype_name = str(frame[column].dtype)
        item["dtype_counts"][dtype_name] = item["dtype_counts"].get(dtype_name, 0) + 1
        item["chunks_seen"] += 1
        series = frame[column]
        item["non_null_cells_sampled"] += int(series.notna().sum())
        item["missing_cells_sampled"] += int(series.isna().sum())
        item["inf_cells_sampled"] += int(count_inf_values(frame[[column]]))
        if len(item["sample_values"]) < 50:
            for value in series.dropna().astype(str).head(50).tolist():
                if value not in item["sample_values"]:
                    item["sample_values"].append(value)
                if len(item["sample_values"]) >= 50:
                    break


def restore_quality(value: Any) -> Dict[str, Dict[str, Any]]:
    return value if isinstance(value, dict) else {}


class DuplicateIndex:
    def __init__(self, path: Path, enabled: bool, resume: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.connection: Optional[sqlite3.Connection] = None
        if not enabled:
            return
        if path.exists() and not resume:
            path.unlink()
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS occurrence ("
            "digest BLOB NOT NULL, file_id TEXT NOT NULL, row_number INTEGER NOT NULL, "
            "PRIMARY KEY (digest, file_id, row_number))"
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS occurrence_file ON occurrence(file_id)")
        self.connection.commit()

    def remove_files(self, file_ids: Iterable[str]) -> None:
        if not self.enabled or self.connection is None:
            return
        self.connection.executemany("DELETE FROM occurrence WHERE file_id = ?", [(value,) for value in file_ids])
        self.connection.commit()

    def add_hashes(self, file_id: str, first_row_number: int, hashes: np.ndarray) -> None:
        if not self.enabled or self.connection is None:
            return
        rows = []
        for offset, value in enumerate(hashes.tolist()):
            rows.append((int(value).to_bytes(8, "little", signed=False), file_id, first_row_number + offset))
        self.connection.executemany(
            "INSERT OR IGNORE INTO occurrence(digest, file_id, row_number) VALUES (?, ?, ?)", rows
        )
        self.connection.commit()

    def summary(self) -> Dict[str, Any]:
        if not self.enabled or self.connection is None:
            return {"enabled": False}
        total = int(self.connection.execute("SELECT COUNT(*) FROM occurrence").fetchone()[0])
        candidate_groups = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM (SELECT digest FROM occurrence GROUP BY digest "
                "HAVING COUNT(DISTINCT file_id) > 1)"
            ).fetchone()[0]
        )
        cross_file_rows = int(
            self.connection.execute(
                "SELECT COALESCE(SUM(n), 0) FROM (SELECT COUNT(*) AS n FROM occurrence "
                "GROUP BY digest HAVING COUNT(DISTINCT file_id) > 1)"
            ).fetchone()[0]
        )
        samples: List[Dict[str, Any]] = []
        rows = self.connection.execute(
            "SELECT digest, file_id, row_number FROM occurrence WHERE digest IN ("
            "SELECT digest FROM occurrence GROUP BY digest HAVING COUNT(DISTINCT file_id) > 1) "
            "ORDER BY digest, file_id, row_number LIMIT 100"
        ).fetchall()
        for digest, file_id, row_number in rows:
            samples.append(
                {"digest": bytes(digest).hex(), "file_id": file_id, "row_number": int(row_number)}
            )
        return {
            "enabled": True,
            "hash_algorithm": "pandas.util.hash_pandas_object uint64",
            "total_indexed_rows": total,
            "candidate_cross_file_duplicate_hash_groups": candidate_groups,
            "candidate_cross_file_duplicate_occurrences": cross_file_rows,
            "sample_occurrences": samples,
            "caveat": "Hash candidates require confirmation; near-duplicates are not detected.",
        }

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None


def read_chunks(path: Path, delimiter: str, read_headers: Sequence[str], chunksize: int):
    return pd.read_csv(
        path,
        sep=delimiter,
        header=0,
        names=list(read_headers),
        chunksize=chunksize,
        low_memory=True,
        keep_default_na=True,
        na_values=sorted(MISSING_TEXT),
        encoding="utf-8-sig",
        encoding_errors="replace",
        on_bad_lines="skip",
        engine="c",
    )


def scan_file(
    record: Dict[str, Any],
    chunksize: int,
    duplicate_index: DuplicateIndex,
    quality: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    path = Path(record["absolute_path"])
    headers = list(record["headers"])
    read_headers = list(record["read_headers"])
    labels = label_indexes(headers)
    times = timestamp_indexes(headers)
    label_column = read_headers[labels[0]] if labels else None
    timestamp_column = read_headers[times[0]] if times else None
    duplicate_columns = [column for index, column in enumerate(read_headers) if index not in labels]

    stats: Dict[str, Any] = {
        "file_id": record["file_id"],
        "absolute_path": str(path),
        "relative_path": record["relative_path"],
        "class_folder": record["class_folder"],
        "file_name": path.name,
        "size_bytes": int(path.stat().st_size),
        "modified_time_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "delimiter": record["delimiter"],
        "headers": headers,
        "read_headers": read_headers,
        "num_columns": len(headers),
        "schema_hash": record["schema_hash"],
        "duplicate_column_names": record["duplicate_column_names"],
        "label_columns": [headers[index] for index in labels],
        "timestamp_columns": [headers[index] for index in times],
        "rows": 0,
        "missing_cells": 0,
        "inf_cells": 0,
        "label_counts": {},
        "folder_label_mismatch_rows": 0,
        "timestamp_parseable": 0,
        "timestamp_unparseable_nonempty": 0,
        "timestamp_min_utc": None,
        "timestamp_max_utc": None,
        "parser_note": "Rows with malformed width may be skipped by pandas on_bad_lines=skip; exact skipped count is unavailable.",
        "dtype_counts": {},
    }
    local_labels: Counter[str] = Counter()
    local_dtypes: Counter[str] = Counter()
    timestamp_min: Optional[pd.Timestamp] = None
    timestamp_max: Optional[pd.Timestamp] = None
    row_number = 2

    reader = read_chunks(path, record["delimiter"], read_headers, chunksize)
    for chunk in reader:
        rows = int(len(chunk))
        if rows == 0:
            continue
        stats["rows"] += rows
        stats["missing_cells"] += int(chunk.isna().sum().sum())
        stats["inf_cells"] += count_inf_values(chunk)
        local_dtypes.update(str(dtype) for dtype in chunk.dtypes.tolist())
        update_quality(quality, chunk)

        if label_column is not None:
            series = chunk[label_column]
            for value, count in series.map(clean_text).value_counts(dropna=False).items():
                label = value if value else "<MISSING_LABEL>"
                local_labels[label] += int(count)
            folder = record["class_folder"]
            for value in series.map(clean_text):
                label = value if value else "<MISSING_LABEL>"
                if label != folder:
                    stats["folder_label_mismatch_rows"] += 1

        if timestamp_column is not None:
            raw = chunk[timestamp_column]
            parsed = parse_timestamps(raw)
            nonempty = raw.notna() & raw.astype("string").str.strip().ne("")
            valid = parsed.notna()
            stats["timestamp_parseable"] += int(valid.sum())
            stats["timestamp_unparseable_nonempty"] += int((nonempty & ~valid).sum())
            if valid.any():
                current_min = parsed[valid].min()
                current_max = parsed[valid].max()
                timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
                timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

        if duplicate_index.enabled and duplicate_columns:
            hashes = pd.util.hash_pandas_object(chunk[duplicate_columns], index=False).to_numpy(dtype="uint64")
            duplicate_index.add_hashes(record["file_id"], row_number, hashes)
        row_number += rows

    if timestamp_min is not None:
        stats["timestamp_min_utc"] = timestamp_min.isoformat()
    if timestamp_max is not None:
        stats["timestamp_max_utc"] = timestamp_max.isoformat()
    stats["label_counts"] = dict(local_labels)
    stats["dtype_counts"] = dict(local_dtypes)
    return stats


def build_records(root: Path) -> Tuple[List[Path], List[Dict[str, Any]]]:
    files = sorted((path for path in root.rglob("*.csv") if path.is_file()), key=lambda value: str(value).lower())
    records: List[Dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        folder = parts[0] if len(parts) > 1 else "<ROOT>"
        delimiter = detect_delimiter(path)
        headers = read_header(path, delimiter)
        duplicate_names = sorted(name for name, count in Counter(headers).items() if name and count > 1)
        read_headers = unique_headers(headers)
        records.append(
            {
                "file_id": file_id_for(relative),
                "absolute_path": str(path.resolve()),
                "relative_path": relative,
                "class_folder": folder,
                "delimiter": delimiter,
                "headers": headers,
                "read_headers": read_headers,
                "num_columns": len(headers),
                "schema_hash": schema_hash(headers),
                "duplicate_column_names": duplicate_names,
            }
        )
    return files, records


def aggregate_labels(file_stats: Iterable[Dict[str, Any]]) -> Tuple[Counter[str], Dict[str, Counter[str]], Counter[str]]:
    actual: Counter[str] = Counter()
    by_folder: Dict[str, Counter[str]] = defaultdict(Counter)
    folder_flows: Counter[str] = Counter()
    for stats in file_stats:
        folder = stats["class_folder"]
        folder_flows[folder] += int(stats.get("rows", 0))
        for label, count in stats.get("label_counts", {}).items():
            actual[label] += int(count)
            by_folder[folder][label] += int(count)
    return actual, by_folder, folder_flows


def capture_rows(file_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_counts = Counter(candidate_family(stats["relative_path"]) for stats in file_stats)
    output: List[Dict[str, Any]] = []
    for stats in file_stats:
        path = Path(stats["absolute_path"])
        family = candidate_family(stats["relative_path"])
        proposed = candidate_group_id(family)
        sidecars = find_sidecars(path)
        if sidecars:
            status = "VERIFIED_BY_SIDECAR_NAME_ONLY"
            confidence = "MEDIUM"
            group_id = proposed
            evidence = "Same-stem sidecar exists: " + "; ".join(sidecars)
        elif family_counts[family] > 1:
            status = "INFERRED_NAME_ONLY"
            confidence = "LOW"
            group_id = "UNKNOWN"
            evidence = f"Conservative filename family repeated for {family_counts[family]} CSV files; no capture manifest verified."
        else:
            status = "UNKNOWN_NOT_VERIFIED"
            confidence = "UNKNOWN"
            group_id = "UNKNOWN"
            evidence = "No authoritative PCAP/session manifest or same-stem sidecar was found."
        output.append(
            {
                "file_id": stats["file_id"],
                "relative_path": stats["relative_path"],
                "absolute_path": stats["absolute_path"],
                "class_folder": stats["class_folder"],
                "file_name": stats["file_name"],
                "proposed_capture_group_id": proposed,
                "capture_group_id": group_id,
                "capture_group_status": status,
                "confidence": confidence,
                "filename_family": family,
                "family_file_count": family_counts[family],
                "evidence": evidence,
            }
        )
    return output


def temporal_overlap_pairs(file_stats: List[Dict[str, Any]], limit: int = 200) -> Tuple[int, List[Dict[str, Any]]]:
    intervals = []
    for stats in file_stats:
        start = stats.get("timestamp_min_utc")
        end = stats.get("timestamp_max_utc")
        if start and end:
            intervals.append((stats["file_id"], stats["relative_path"], start, end))
    pairs: List[Dict[str, Any]] = []
    total = 0
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if left[2] <= right[3] and right[2] <= left[3]:
                total += 1
                if len(pairs) < limit:
                    pairs.append(
                        {
                            "file_id_a": left[0],
                            "relative_path_a": left[1],
                            "start_a": left[2],
                            "end_a": left[3],
                            "file_id_b": right[0],
                            "relative_path_b": right[1],
                            "start_b": right[2],
                            "end_b": right[3],
                        }
                    )
    return total, pairs


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_inventory(file_stats: List[Dict[str, Any]], captures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    capture_by_id = {row["file_id"]: row for row in captures}
    rows = []
    for stats in file_stats:
        capture = capture_by_id[stats["file_id"]]
        rows.append(
            {
                "file_id": stats["file_id"],
                "absolute_path": stats["absolute_path"],
                "relative_path": stats["relative_path"],
                "class_folder": stats["class_folder"],
                "file_name": stats["file_name"],
                "size_bytes": stats["size_bytes"],
                "rows": stats["rows"],
                "num_columns": stats["num_columns"],
                "schema_hash": stats["schema_hash"],
                "headers_json": json.dumps(stats["headers"], ensure_ascii=False),
                "delimiter": stats["delimiter"],
                "duplicate_column_names_json": json.dumps(stats["duplicate_column_names"], ensure_ascii=False),
                "label_columns_json": json.dumps(stats["label_columns"], ensure_ascii=False),
                "label_counts_json": json.dumps(stats["label_counts"], ensure_ascii=False, sort_keys=True),
                "missing_cells": stats["missing_cells"],
                "inf_cells": stats["inf_cells"],
                "folder_label_mismatch_rows": stats["folder_label_mismatch_rows"],
                "timestamp_columns_json": json.dumps(stats["timestamp_columns"], ensure_ascii=False),
                "timestamp_min_utc": stats["timestamp_min_utc"],
                "timestamp_max_utc": stats["timestamp_max_utc"],
                "timestamp_parseable": stats["timestamp_parseable"],
                "timestamp_unparseable_nonempty": stats["timestamp_unparseable_nonempty"],
                "capture_group_id": capture["capture_group_id"],
                "capture_group_status": capture["capture_group_status"],
                "capture_confidence": capture["confidence"],
                "parser_note": stats["parser_note"],
            }
        )
    return rows


def write_class_distribution(
    output_path: Path,
    actual: Counter[str],
    by_folder: Dict[str, Counter[str]],
    folder_flows: Counter[str],
) -> None:
    rows = []
    folders = list(EXPECTED_CLASSES) + sorted(set(folder_flows) - set(EXPECTED_CLASSES))
    for folder in folders:
        rows.append(
            {
                "scope": "folder",
                "folder_or_label": folder,
                "flow_count": int(folder_flows.get(folder, 0)),
                "csv_count": "",
                "source_folder": folder,
                "label_value": "",
            }
        )
        for label, count in sorted(by_folder.get(folder, {}).items()):
            rows.append(
                {
                    "scope": "folder_label",
                    "folder_or_label": folder,
                    "flow_count": int(count),
                    "csv_count": "",
                    "source_folder": folder,
                    "label_value": label,
                }
            )
    for label, count in sorted(actual.items()):
        rows.append(
            {
                "scope": "dataset_label",
                "folder_or_label": label,
                "flow_count": int(count),
                "csv_count": "",
                "source_folder": "",
                "label_value": label,
            }
        )
    write_csv(output_path, rows, ["scope", "folder_or_label", "flow_count", "csv_count", "source_folder", "label_value"])


def build_schema_audit(
    file_stats: List[Dict[str, Any]],
    quality: Dict[str, Dict[str, Any]],
    actual_labels: Counter[str],
    duplicate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    schemas: Dict[str, Dict[str, Any]] = {}
    all_headers = set()
    duplicate_columns = {}
    label_file_count = 0
    mismatch_rows = 0
    total_missing = 0
    total_inf = 0
    for stats in file_stats:
        schema = schemas.setdefault(
            stats["schema_hash"],
            {"headers": stats["headers"], "file_count": 0, "flow_count": 0, "files": []},
        )
        schema["file_count"] += 1
        schema["flow_count"] += int(stats["rows"])
        schema["files"].append(stats["relative_path"])
        all_headers.update(stats["headers"])
        if stats["duplicate_column_names"]:
            duplicate_columns[stats["relative_path"]] = stats["duplicate_column_names"]
        if stats["label_columns"]:
            label_file_count += 1
        mismatch_rows += int(stats["folder_label_mismatch_rows"])
        total_missing += int(stats["missing_cells"])
        total_inf += int(stats["inf_cells"])

    schema_rows = []
    for schema_hash_value, schema in sorted(schemas.items()):
        label_indexes_for_schema = label_indexes(schema["headers"])
        metadata_indexes = [index for index, header in enumerate(schema["headers"]) if metadata_risk(header)[0] != "NONE"]
        schema_rows.append(
            {
                "schema_hash": schema_hash_value,
                "headers": schema["headers"],
                "file_count": schema["file_count"],
                "flow_count": schema["flow_count"],
                "feature_columns_excluding_label": max(0, len(schema["headers"]) - len(label_indexes_for_schema)),
                "model_candidate_features_excluding_label_metadata": max(
                    0, len(schema["headers"]) - len(set(label_indexes_for_schema).union(metadata_indexes))
                ),
                "files": schema["files"][:100],
                "files_truncated": len(schema["files"]) > 100,
            }
        )

    metadata = []
    for header in sorted(all_headers, key=lambda value: value.lower()):
        risk, reason = metadata_risk(header)
        metadata.append({"column": header, "risk": risk, "reason": reason})

    label_confidence = "LOW"
    expected_set = set(EXPECTED_CLASSES)
    actual_non_missing = {label for label in actual_labels if label != "<MISSING_LABEL>"}
    if label_file_count == len(file_stats) and actual_non_missing and actual_non_missing.issubset(expected_set) and mismatch_rows == 0:
        label_confidence = "HIGH_STRUCTURAL_ONLY"
    elif label_file_count > 0:
        label_confidence = "MEDIUM_OR_LOW_REQUIRES_SEMANTIC_CHECK"

    return {
        "generated_at_utc": now_utc(),
        "audit_scope": "read-only streaming audit; no split/window/training",
        "total_csv": len(file_stats),
        "total_flow": sum(int(stats["rows"]) for stats in file_stats),
        "expected_class_folders": EXPECTED_CLASSES,
        "actual_label_values": dict(actual_labels),
        "label_audit": {
            "files_with_label_column": label_file_count,
            "files_without_label_column": len(file_stats) - label_file_count,
            "folder_label_mismatch_rows": mismatch_rows,
            "label_confidence": label_confidence,
            "note": "Confidence is structural only; it is not proof that the labels are semantically correct.",
        },
        "quality_totals": {"missing_cells": total_missing, "inf_cells": total_inf},
        "schema_consistency": {
            "unique_schema_count": len(schemas),
            "all_csv_same_schema": len(schemas) <= 1,
            "schemas": schema_rows,
        },
        "duplicate_column_names_by_file": duplicate_columns,
        "all_observed_columns": sorted(all_headers, key=lambda value: value.lower()),
        "metadata_and_leakage_risk_columns": metadata,
        "column_quality_sample": quality,
        "duplicate_flow_audit": duplicate_summary,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(value).replace("|", "\\|") for value in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_leakage_report(
    path: Path,
    file_stats: List[Dict[str, Any]],
    captures: List[Dict[str, Any]],
    duplicate_summary: Dict[str, Any],
    temporal_count: int,
    temporal_samples: List[Dict[str, Any]],
    schema_audit: Dict[str, Any],
) -> None:
    unknown_capture = sum(row["capture_group_id"] == "UNKNOWN" for row in captures)
    inferred_capture = sum(row["capture_group_status"] == "INFERRED_NAME_ONLY" for row in captures)
    sidecar_capture = sum(row["capture_group_status"] == "VERIFIED_BY_SIDECAR_NAME_ONLY" for row in captures)
    metadata_high = [
        row["column"]
        for row in schema_audit["metadata_and_leakage_risk_columns"]
        if row["risk"] in {"HIGH", "CRITICAL"}
    ]
    lines = [
        "# Leakage Risk Report - STAGE 1",
        "",
        f"Generated UTC: `{now_utc()}`",
        "",
        "> This is an audit only. No source CSV was edited, no split was created, no sliding window was created, and no model was trained.",
        "",
        "## Overall gate",
        "",
        "**BLOCKED / NOT READY FOR STAGE 2 SPLITTING.** Capture/session independence is not verified for the dataset, so a 60:20:20 split cannot be called leakage-safe yet.",
        "",
        "## Findings",
        "",
        markdown_table(
            ["Risk", "Evidence", "Assessment", "Required control"],
            [
                ["1. Capture/session leakage", f"{unknown_capture} CSVs have UNKNOWN capture_group_id; {inferred_capture} are filename-only inferences; {sidecar_capture} have same-stem sidecars.", "CRITICAL / unresolved", "Obtain authoritative CSV-to-PCAP/session mapping and split by verified capture group."],
                ["2. Temporal overlap", f"{temporal_count} overlapping timestamp interval pairs detected; {len(temporal_samples)} sample pairs exported in report context.", "HIGH if timestamps are comparable", "Normalize timezone and keep time intervals disjoint across splits."],
                ["3. Duplicate flows", f"{duplicate_summary.get('candidate_cross_file_duplicate_hash_groups', 0)} candidate cross-file duplicate hash groups; {duplicate_summary.get('candidate_cross_file_duplicate_occurrences', 0)} occurrences.", "HIGH when nonzero; hash candidates need confirmation", "Deduplicate or group duplicates before splitting; do not split identical flows across sets."],
                ["4. Label leakage", f"Label audit: {schema_audit['label_audit']['files_with_label_column']} files with label column; {schema_audit['label_audit']['folder_label_mismatch_rows']} folder/label mismatch rows.", "Target must never enter features", "Drop label only inside the future training pipeline; preserve raw labels in audit."],
                ["5. Feature leakage", "High-risk columns: " + (", ".join(metadata_high) if metadata_high else "none detected by name"), "Potentially severe", "Review Flow ID, IP, time, host, capture, and scenario metadata before modeling."],
                ["6. Sliding-window boundary", "Any future windowing across files or splits can mix sessions and leak context.", "CRITICAL if done", "Window only inside a single verified capture group after splitting."],
            ],
        ),
        "",
        "## Capture evidence interpretation",
        "",
        "Filename similarity is not proof that files came from one PCAP, and a unique filename is not proof of capture independence. Rows marked `UNKNOWN` must remain unknown until an external manifest, PCAP metadata, or reproducible capture log is supplied.",
        "",
        "## Temporal overlap examples",
        "",
    ]
    if temporal_samples:
        lines.append(markdown_table(["File A", "A interval", "File B", "B interval"], [[row["relative_path_a"], f"{row['start_a']} → {row['end_a']}", row["relative_path_b"], f"{row['start_b']} → {row['end_b']}"] for row in temporal_samples[:20]]))
    else:
        lines.append("No comparable timestamp intervals were available, so temporal independence is **not proven**.")
    lines.extend(
        [
            "",
            "## Safe Stage 2 proposal (not executed)",
            "",
            "1. Build and verify a manifest mapping every CSV to a real PCAP/capture/session identifier. Keep unresolved files as `UNKNOWN` and do not silently assign them.",
            "2. Use capture-group-disjoint assignment to train/validation/test. Choose groups, not individual rows or files, as the split unit.",
            "3. Use a deterministic group-level optimizer to target 60:20:20 while preserving the eight-class distribution as far as the group constraints permit.",
            "4. Check temporal overlap, duplicate hashes, and label proportions after assignment. Freeze the manifest and split seed.",
            "5. Only then create windows inside each already assigned capture group; never cross file or split boundaries.",
            "",
            "## Required information before Stage 2",
            "",
            "- Authoritative mapping `CSV path -> PCAP/capture/session ID` (or a statement that a mapping does not exist).",
            "- Whether similarly named CSVs are parts of one capture, and the ordering/timezone used by each capture.",
            "- Meaning and provenance of the label column, including whether labels were assigned per flow or per file.",
            "- Decision on duplicate-flow policy and whether Flow ID/IP/timestamp fields may be retained for analysis only or must be excluded from model features.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_stage_report(
    path: Path,
    root: Path,
    output_root: Path,
    file_stats: List[Dict[str, Any]],
    captures: List[Dict[str, Any]],
    actual: Counter[str],
    by_folder: Dict[str, Counter[str]],
    folder_flows: Counter[str],
    schema_audit: Dict[str, Any],
    duplicate_summary: Dict[str, Any],
    temporal_count: int,
    report_paths: Dict[str, Path],
) -> None:
    total_flow = sum(int(stats["rows"]) for stats in file_stats)
    csv_by_folder = Counter(stats["class_folder"] for stats in file_stats)
    unknown_capture = sum(row["capture_group_id"] == "UNKNOWN" for row in captures)
    flow_rows = []
    for folder in list(EXPECTED_CLASSES) + sorted(set(folder_flows) - set(EXPECTED_CLASSES)):
        flow_rows.append([folder, csv_by_folder.get(folder, 0), folder_flows.get(folder, 0), ", ".join(f"{key}={value}" for key, value in sorted(by_folder.get(folder, {}).items())) or "<no label column>"])
    label_rows = [[label, count] for label, count in sorted(actual.items())]
    lines = [
        "# STAGE 1 - CICFlowMeter Dataset Audit",
        "",
        f"Generated UTC: `{now_utc()}`",
        f"Dataset root: `{root}`",
        f"Output root: `{output_root}`",
        "",
        "## Final gate",
        "",
        "**BLOCKED / NOT READY FOR STAGE 2.** The audit completed, but capture/session independence is not verified. Do not perform a 60:20:20 split until the requested capture manifest is available and reviewed.",
        "",
        "## Dataset summary",
        "",
        markdown_table(["Folder", "CSV files", "Flow rows", "Observed labels"], flow_rows),
        "",
        f"- Total CSV files: **{len(file_stats):,}**",
        f"- Total flow rows: **{total_flow:,}**",
        f"- Unique schema hashes: **{schema_audit['schema_consistency']['unique_schema_count']}**",
        f"- Same schema for all CSVs: **{schema_audit['schema_consistency']['all_csv_same_schema']}**",
        f"- Actual feature columns: see `schema_audit.json`; per-schema counts exclude detected label columns and separately report metadata-risk columns.",
        f"- Total missing cells: **{schema_audit['quality_totals']['missing_cells']:,}**",
        f"- Total Inf cells: **{schema_audit['quality_totals']['inf_cells']:,}**",
        "",
        "## Actual label distribution",
        "",
        markdown_table(["Label value (raw text)", "Flow rows"], label_rows or [["<no label values detected>", 0]]),
        "",
        "## Label reliability",
        "",
        f"- Label confidence: **{schema_audit['label_audit']['label_confidence']}**.",
        f"- Files with label column: **{schema_audit['label_audit']['files_with_label_column']} / {len(file_stats)}**.",
        f"- Folder/label mismatch rows: **{schema_audit['label_audit']['folder_label_mismatch_rows']:,}**.",
        "- This is structural confidence only. It does not validate the semantic correctness of labels or their original annotation process.",
        "",
        "## Capture/session audit",
        "",
        f"- CSVs with `capture_group_id=UNKNOWN`: **{unknown_capture:,} / {len(captures):,}**.",
        "- Filename-derived groups are explicitly marked as inference, not as verified capture identity.",
        "- A unique CSV filename does not prove an independent PCAP/capture/session.",
        "",
        "## Leakage findings",
        "",
        f"- Candidate cross-file duplicate hash groups: **{duplicate_summary.get('candidate_cross_file_duplicate_hash_groups', 0):,}**.",
        f"- Timestamp interval-overlap pairs: **{temporal_count:,}**.",
        "- High-risk metadata columns are listed in `schema_audit.json`; Flow ID, IP, timestamp, host, capture, and scenario fields require explicit review.",
        "- Future sliding windows must be created only after split assignment and only within one capture group.",
        "",
        "## Safe Stage 2 recommendation (not implemented)",
        "",
        "Use a verified capture/session group as the atomic split unit. Optimize group assignments toward 60:20:20 and eight-class proportions, then validate temporal overlap and duplicate-flow separation. Freeze the mapping and random seed before any windowing or training.",
        "",
        "## Required user input",
        "",
        "1. Provide a trustworthy `CSV -> PCAP/capture/session` manifest, or confirm that it does not exist.",
        "2. Confirm timestamp timezone/order and whether similarly named files are capture parts.",
        "3. Confirm label provenance and the policy for Flow ID/IP/timestamp metadata and duplicate flows.",
        "",
        "## Generated reports",
        "",
    ]
    for name, report_path in report_paths.items():
        lines.append(f"- `{name}`: `{report_path}`")
    lines.extend(
        [
            "",
            "## Execution boundary",
            "",
            "This Stage 1 script did not modify source CSV files, create a dataset split, create sliding windows, or train a model.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def checkpoint_payload(
    root: Path,
    output_root: Path,
    signature: str,
    duplicate_enabled: bool,
    records: List[Dict[str, Any]],
    completed: set[str],
    stats_by_id: Dict[str, Dict[str, Any]],
    quality: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "updated_at_utc": now_utc(),
        "dataset_root": str(root),
        "output_root": str(output_root),
        "dataset_signature": signature,
        "duplicate_enabled": duplicate_enabled,
        "total_files": len(records),
        "completed_file_ids": sorted(completed),
        "file_stats": [stats_by_id[file_id] for file_id in sorted(stats_by_id)],
        "quality": quality,
        "complete": len(completed) == len(records),
    }


def load_or_initialize(
    checkpoint_path: Path,
    root: Path,
    output_root: Path,
    signature: str,
    duplicate_enabled: bool,
    records: List[Dict[str, Any]],
) -> Tuple[set[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], bool]:
    if not checkpoint_path.exists():
        return set(), {}, {}, False
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Checkpoint cannot be read: {checkpoint_path} ({exc})") from exc
    if state.get("dataset_signature") != signature:
        raise RuntimeError(
            "Dataset signature changed since the checkpoint was written. "
            f"Review or remove only this checkpoint before restarting: {checkpoint_path}"
        )
    if bool(state.get("duplicate_enabled", True)) != duplicate_enabled:
        raise RuntimeError("The resume command must use the same duplicate-index setting as the checkpoint.")
    valid_ids = {record["file_id"] for record in records}
    completed = set(state.get("completed_file_ids", [])) & valid_ids
    stats_by_id = {item["file_id"]: item for item in state.get("file_stats", []) if item.get("file_id") in completed}
    completed &= set(stats_by_id)
    quality = restore_quality(state.get("quality", {}))
    return completed, stats_by_id, quality, True


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root == root or root in output_root.parents:
        print("WARNING: output-root is inside dataset-root; use a separate output directory when possible.", file=sys.stderr)

    print(f"Scanning CSV inventory under: {root}")
    files, records = build_records(root)
    if not records:
        raise RuntimeError(f"No CSV files found under {root}")
    signature = dataset_signature(root, files)
    checkpoint_path = output_root / ".stage1_checkpoint.json"
    duplicate_path = output_root / ".stage1_duplicate_index.sqlite"
    completed, stats_by_id, quality, resuming = load_or_initialize(
        checkpoint_path, root, output_root, signature, not args.no_duplicates, records
    )
    duplicate_index = DuplicateIndex(duplicate_path, not args.no_duplicates, resuming)
    try:
        if resuming:
            unfinished = {record["file_id"] for record in records} - completed
            duplicate_index.remove_files(unfinished)
            print(f"Checkpoint detected: resuming with {len(completed):,} completed CSVs; {len(unfinished):,} remain.")
        else:
            print(f"New audit: {len(records):,} CSV files found.")

        atomic_write_json(
            checkpoint_path,
            checkpoint_payload(root, output_root, signature, not args.no_duplicates, records, completed, stats_by_id, quality),
        )
        record_by_id = {record["file_id"]: record for record in records}
        for position, record in enumerate(records, start=1):
            file_id = record["file_id"]
            if file_id in completed:
                continue
            print(f"[{position}/{len(records)}] {record['relative_path']}", flush=True)
            stats_by_id[file_id] = scan_file(record, args.chunksize, duplicate_index, quality)
            completed.add(file_id)
            atomic_write_json(
                checkpoint_path,
                checkpoint_payload(root, output_root, signature, not args.no_duplicates, records, completed, stats_by_id, quality),
            )
        if completed != set(record_by_id):
            raise RuntimeError("Internal checkpoint error: not all CSVs were completed.")

        file_stats = [stats_by_id[record["file_id"]] for record in records]
        captures = capture_rows(file_stats)
        duplicate_summary = duplicate_index.summary()
        temporal_count, temporal_samples = temporal_overlap_pairs(file_stats)
        actual, by_folder, folder_flows = aggregate_labels(file_stats)
        schema_audit = build_schema_audit(file_stats, quality, actual, duplicate_summary)

        inventory_path = output_root / "inventory.csv"
        class_distribution_path = output_root / "class_distribution.csv"
        schema_path = output_root / "schema_audit.json"
        captures_path = output_root / "capture_groups.csv"
        leakage_path = output_root / "leakage_risk_report.md"
        stage_report_path = output_root / "STAGE1_REPORT.md"
        report_paths = {
            "inventory.csv": inventory_path,
            "class_distribution.csv": class_distribution_path,
            "schema_audit.json": schema_path,
            "capture_groups.csv": captures_path,
            "leakage_risk_report.md": leakage_path,
            "STAGE1_REPORT.md": stage_report_path,
        }

        inventory_rows = make_inventory(file_stats, captures)
        write_csv(
            inventory_path,
            inventory_rows,
            [
                "file_id", "absolute_path", "relative_path", "class_folder", "file_name", "size_bytes", "rows",
                "num_columns", "schema_hash", "headers_json", "delimiter", "duplicate_column_names_json",
                "label_columns_json", "label_counts_json", "missing_cells", "inf_cells", "folder_label_mismatch_rows",
                "timestamp_columns_json", "timestamp_min_utc", "timestamp_max_utc", "timestamp_parseable",
                "timestamp_unparseable_nonempty", "capture_group_id", "capture_group_status", "capture_confidence",
                "parser_note",
            ],
        )
        write_class_distribution(class_distribution_path, actual, by_folder, folder_flows)
        atomic_write_json(schema_path, schema_audit)
        write_csv(
            captures_path,
            captures,
            [
                "file_id", "relative_path", "absolute_path", "class_folder", "file_name", "proposed_capture_group_id",
                "capture_group_id", "capture_group_status", "confidence", "filename_family", "family_file_count", "evidence",
            ],
        )
        write_leakage_report(leakage_path, file_stats, captures, duplicate_summary, temporal_count, temporal_samples, schema_audit)
        write_stage_report(
            stage_report_path,
            root,
            output_root,
            file_stats,
            captures,
            actual,
            by_folder,
            folder_flows,
            schema_audit,
            duplicate_summary,
            temporal_count,
            report_paths,
        )
        atomic_write_json(
            checkpoint_path,
            checkpoint_payload(root, output_root, signature, not args.no_duplicates, records, completed, stats_by_id, quality),
        )
        print("Audit complete. Checkpoint retained with complete=true for traceability.")
        print(f"Reports: {output_root}")
        return 0
    finally:
        duplicate_index.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only CICFlowMeter Stage 1 audit with checkpoint/resume.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing the CICFlowMeter CSV folders.")
    parser.add_argument("--output-root", required=True, help="Separate directory for audit outputs and checkpoint files.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="Rows per pandas chunk; default: 50000.")
    parser.add_argument("--no-duplicates", action="store_true", help="Skip the SQLite duplicate-flow hash index.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    started = time.time()
    try:
        result = run(args)
        print(f"Elapsed seconds: {time.time() - started:.1f}")
        return result
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C. The last fully completed CSV is saved in .stage1_checkpoint.json.", file=sys.stderr)
        print("Rerun the exact same command to resume; source CSV files were not modified.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
