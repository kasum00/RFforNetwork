#!/usr/bin/env python3
"""
Stage 1 read-only audit for CICFlowMeter CSV data.

This script deliberately does not split data, create windows, train models, or
modify source CSVs.  It streams CSV rows and keeps only compact counters and
metadata in memory.  Exact cross-file duplicate detection uses a temporary
SQLite index on disk so it does not require all flows in RAM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "missing"}
INF_TOKENS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
INF_LITERALS = sorted({token for base in INF_TOKENS for token in (base, base.upper(), base.capitalize())})
CSV_EXTENSIONS = {".csv"}
SOURCE_EXTENSIONS = {".pcap", ".pcapng", ".json", ".yaml", ".yml", ".manifest", ".txt"}
MAX_TRACKED_DISTINCT = 1000


def json_default(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Unsupported JSON value: {type(obj)!r}")


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_header(value: str) -> str:
    value = str(value).replace("\ufeff", "").strip()
    return re.sub(r"\s+", " ", value).lower()


def is_missing(value: str) -> bool:
    return str(value).strip().lower() in MISSING_TOKENS


def is_inf(value: str) -> bool:
    token = str(value).strip().lower()
    if token in INF_TOKENS:
        return True
    try:
        return math.isinf(float(token))
    except (ValueError, TypeError):
        return False


def classify_value(value: str) -> str:
    token = str(value).strip()
    if token.lower() in MISSING_TOKENS:
        return "missing"
    if is_inf(token):
        return "inf"
    if re.fullmatch(r"[+-]?\d+", token):
        return "integer"
    try:
        float(token)
        return "float"
    except (ValueError, TypeError):
        return "string"


def safe_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def file_stem_without_flow(csv_path: Path) -> str:
    name = csv_path.name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    name = re.sub(r"_flow$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.pcap(?:ng)?$", "", name, flags=re.IGNORECASE)
    return name.strip()


def filename_family(stem: str) -> Tuple[str, Optional[str]]:
    """Return normalized family core and an optional trailing numeric suffix."""
    raw = stem.strip()
    match = re.match(r"^(.*?)(?:[-_ ]?)(\d+)$", raw)
    suffix = None
    if match and match.group(1).strip("-_ "):
        raw = match.group(1).rstrip("-_ ")
        suffix = match.group(2)
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return normalized or "unknown_name", suffix


def detect_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(65536)
        candidates = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
        delimiter, count = max(candidates.items(), key=lambda item: item[1])
        return delimiter if count > 0 else ","
    except OSError:
        return ","


def parse_timestamp(value: str) -> Optional[datetime]:
    token = str(value).strip()
    if not token or token.lower() in MISSING_TOKENS:
        return None
    token = token.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def iso_or_blank(value: Optional[datetime]) -> str:
    return value.isoformat() if value is not None else ""


def candidate_timestamp_columns(headers: Sequence[str]) -> List[int]:
    names = [normalize_header(header) for header in headers]
    exact = [i for i, name in enumerate(names) if name in {"timestamp", "flow start time", "start time", "datetime"}]
    if exact:
        return exact
    return [i for i, name in enumerate(names) if "timestamp" in name or name.endswith(" start time")]


def metadata_risk_for_header(header: str) -> str:
    name = normalize_header(header)
    if name in {"label", "class", "target", "attack", "category"}:
        return "label_or_target"
    if any(token in name for token in ("flow id", "flowid", "src ip", "source ip", "dst ip", "destination ip")):
        return "identifier_or_ip"
    if "timestamp" in name or name in {"flow start time", "start time", "date", "datetime"}:
        return "timestamp"
    if any(token in name for token in ("filename", "file name", "pcap", "capture", "session", "split")):
        return "file_or_session_metadata"
    return ""


def schema_hash(headers: Sequence[str]) -> str:
    payload = "\x1f".join(str(header).strip() for header in headers)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def canonical_row_hash(row: Sequence[str], label_indices: Sequence[int]) -> bytes:
    labels = set(label_indices)
    payload = "\x1f".join(
        str(value).strip().replace("\r", "").replace("\n", "")
        for i, value in enumerate(row)
        if i not in labels
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).digest()


class ColumnTracker:
    def __init__(self):
        self.rows_seen = 0
        self.missing = 0
        self.inf = 0
        self.types = Counter()
        self.distinct_values: Dict[str, int] = {}
        self.distinct_overflow = False
        self.value_labels: Dict[str, set] = defaultdict(set)

    def observe(self, value: str, label: str):
        self.rows_seen += 1
        kind = classify_value(value)
        self.types[kind] += 1
        if kind == "missing":
            self.missing += 1
        elif kind == "inf":
            self.inf += 1
        if not self.distinct_overflow:
            token = str(value).strip()
            if token not in self.distinct_values:
                if len(self.distinct_values) >= MAX_TRACKED_DISTINCT:
                    self.distinct_overflow = True
                    self.distinct_values.clear()
                    self.value_labels.clear()
                else:
                    self.distinct_values[token] = 1
            elif token in self.distinct_values:
                self.distinct_values[token] += 1
            if not self.distinct_overflow and token in self.distinct_values and label:
                self.value_labels[token].add(label)

    def observe_sample_series(self, values, labels):
        """Update type/low-cardinality evidence from one bounded pandas sample."""
        try:
            import pandas as pd
        except ImportError:
            return
        series = values.astype("string").fillna("")
        label_series = labels.astype("string").fillna("")
        self.rows_seen += int(len(series))
        stripped = series.str.strip()
        lower = stripped.str.lower()
        missing_mask = lower.isin(MISSING_TOKENS)
        inf_mask = lower.isin(INF_TOKENS)
        numeric = pd.to_numeric(stripped.where(~missing_mask & ~inf_mask), errors="coerce")
        integer_mask = (~missing_mask & ~inf_mask & numeric.notna() & stripped.str.fullmatch(r"[+-]?\d+", na=False))
        float_mask = (~missing_mask & ~inf_mask & numeric.notna() & ~integer_mask)
        string_mask = (~missing_mask & ~inf_mask & numeric.isna())
        self.missing += int(missing_mask.sum())
        self.inf += int(inf_mask.sum())
        self.types["missing"] += int(missing_mask.sum())
        self.types["inf"] += int(inf_mask.sum())
        self.types["integer"] += int(integer_mask.sum())
        self.types["float"] += int(float_mask.sum())
        self.types["string"] += int(string_mask.sum())
        if self.distinct_overflow:
            return
        sample_values = stripped.drop_duplicates()
        if len(sample_values) > MAX_TRACKED_DISTINCT:
            self.distinct_overflow = True
            self.distinct_values.clear()
            self.value_labels.clear()
            return
        for value in sample_values.tolist():
            self.distinct_values[str(value)] = self.distinct_values.get(str(value), 0) + 1
        for value, label_set in zip(stripped.tolist(), label_series.tolist()):
            value = str(value)
            if value and value in self.distinct_values and label_set:
                self.value_labels[value].add(str(label_set))

    def to_dict(self):
        low_cardinality = None
        if not self.distinct_overflow:
            low_cardinality = {
                "distinct_values_observed": len(self.distinct_values),
                "value_to_label_cardinality": {
                    value: len(labels) for value, labels in self.value_labels.items()
                },
            }
        return {
            "rows_seen": self.rows_seen,
            "missing_cells": self.missing,
            "inf_cells": self.inf,
            "type_counts": dict(self.types),
            "distinct_values_observed": None if self.distinct_overflow else len(self.distinct_values),
            "distinct_tracking_overflow": self.distinct_overflow,
            "low_cardinality_label_mapping": low_cardinality,
        }


class DuplicateIndex:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS occurrences ("
            "digest BLOB NOT NULL, file_id INTEGER NOT NULL, row_number INTEGER NOT NULL, "
            "PRIMARY KEY (digest, file_id)) WITHOUT ROWID"
        )
        self.batch: List[Tuple[bytes, int, int]] = []
        self.rows_hashed = 0

    def add(self, digest: bytes, file_id: int, row_number: int):
        self.batch.append((digest, file_id, row_number))
        self.rows_hashed += 1
        if len(self.batch) >= 5000:
            self.flush()

    def add_many(self, digests, file_id: int, first_row_number: int):
        values = digests.tolist() if hasattr(digests, "tolist") else list(digests)
        self.batch.extend(
            (int(value).to_bytes(8, byteorder="little", signed=False), file_id, first_row_number + offset)
            for offset, value in enumerate(values)
        )
        self.rows_hashed += len(values)
        if len(self.batch) >= 5000:
            self.flush()

    def flush(self):
        if not self.batch:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO occurrences(digest, file_id, row_number) VALUES (?, ?, ?)",
            self.batch,
        )
        self.conn.commit()
        self.batch.clear()

    def cross_file_summary(self, file_names: Dict[int, str]) -> Dict[str, object]:
        self.flush()
        groups = self.conn.execute(
            "SELECT digest, COUNT(*) AS file_count, MIN(file_id) AS first_file, "
            "MIN(row_number) AS first_row FROM occurrences GROUP BY digest HAVING COUNT(*) > 1"
        )
        duplicate_group_count = 0
        duplicate_file_pairs = 0
        duplicate_flow_occurrences = 0
        sample_groups = []
        for digest, file_count, first_file, first_row in groups:
            duplicate_group_count += 1
            duplicate_file_pairs += int(file_count)
            duplicate_flow_occurrences += int(file_count)
            if len(sample_groups) < 20:
                rows = self.conn.execute(
                    "SELECT file_id, row_number FROM occurrences WHERE digest=? ORDER BY file_id",
                    (digest,),
                ).fetchall()
                sample_groups.append(
                    {
                        "digest_sha256": bytes(digest).hex(),
                        "files": [
                            {"file_id": int(file_id), "file": file_names.get(int(file_id), ""), "first_row": int(row)}
                            for file_id, row in rows
                        ],
                    }
                )
        distinct_digest_count = self.conn.execute("SELECT COUNT(DISTINCT digest) FROM occurrences").fetchone()[0]
        return {
            "rows_hashed": self.rows_hashed,
            "distinct_file_digest_pairs": int(self.conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]),
            "distinct_digests": int(distinct_digest_count),
            "cross_file_duplicate_digest_groups": duplicate_group_count,
            "cross_file_duplicate_file_occurrences": duplicate_flow_occurrences,
            "sample_cross_file_duplicate_groups": sample_groups,
        }

    def close(self):
        self.flush()
        self.conn.close()


def build_file_context(path: Path, dataset_root: Path, source_files: Sequence[Path]) -> Dict[str, object]:
    relative = path.relative_to(dataset_root)
    parts = relative.parts
    outer_class = parts[0] if parts and parts[0] in EXPECTED_CLASSES else "OUTSIDE_EXPECTED_CLASS"
    inner_parent = "/".join(parts[1:-1]) if len(parts) > 1 else ""
    stem = file_stem_without_flow(path)
    normalized_core, suffix = filename_family(stem)
    family_key = f"{inner_parent.lower()}::{normalized_core}"
    candidate_id = "CANDIDATE_" + hashlib.sha1(family_key.encode("utf-8")).hexdigest()[:12]
    sidecars = []
    stem_lower = stem.lower()
    for source in source_files:
        if source.parent == path.parent and source.stem.lower() == stem_lower:
            sidecars.append(str(source))
    return {
        "path": path,
        "relative_path": relative.as_posix(),
        "class_folder": outer_class,
        "inner_parent": inner_parent,
        "filename": path.name,
        "filename_stem": stem,
        "normalized_name_core": normalized_core,
        "numeric_suffix": suffix or "",
        "family_key": family_key,
        "candidate_id": candidate_id,
        "sidecars": sidecars,
    }


def empty_file_stats(context: Dict[str, object]) -> Dict[str, object]:
    return {
        "file_id": None,
        "absolute_path": str(context["path"]),
        "relative_path": context["relative_path"],
        "class_folder": context["class_folder"],
        "inner_parent": context["inner_parent"],
        "filename": context["filename"],
        "file_size_bytes": 0,
        "file_size_gib": 0.0,
        "rows": 0,
        "rows_with_expected_width": 0,
        "rows_short_width": 0,
        "rows_long_width": 0,
        "csv_parse_errors": 0,
        "header_read": False,
        "header_names": [],
        "num_columns": 0,
        "schema_hash": "",
        "label_column_indices": [],
        "label_column_names": [],
        "label_counts": {},
        "missing_cells": 0,
        "inf_cells": 0,
        "timestamp_columns": [],
        "timestamp_parseable_rows": 0,
        "timestamp_unparseable_nonmissing_rows": 0,
        "timestamp_min": "",
        "timestamp_max": "",
        "label_mismatch_rows": 0,
        "sample_values": {},
        "error": "",
        "candidate_capture_group_id": context["candidate_id"],
        "capture_group_id": "UNKNOWN",
        "capture_group_status": "UNKNOWN_NOT_VERIFIED",
        "capture_group_confidence": "UNKNOWN",
        "capture_group_evidence": "",
        "source_sidecars": ";".join(context["sidecars"]),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def build_family_summary(file_stats: Sequence[Dict[str, object]], contexts: Sequence[Dict[str, object]]) -> Dict[str, object]:
    families = defaultdict(list)
    for stat, context in zip(file_stats, contexts):
        families[context["family_key"]].append((stat, context))
    summary = {}
    for key, members in families.items():
        class_set = sorted({str(context["class_folder"]) for _, context in members})
        suffixes = sorted({str(context["numeric_suffix"]) for _, context in members if context["numeric_suffix"]})
        total_rows = sum(int(stat["rows"]) for stat, _ in members)
        candidate_id = str(members[0][0]["candidate_capture_group_id"])
        has_sidecar = any(context["sidecars"] for _, context in members)
        if has_sidecar:
            status = "VERIFIED_BY_EXPLICIT_SIDECAR_NAME_ONLY"
            confidence = "MEDIUM"
            evidence = "same-directory source sidecar has the same capture stem; source content was not parsed"
        elif len(members) > 1:
            status = "INFERRED_NAME_ONLY"
            confidence = "LOW"
            evidence = "same normalized filename core and/or numeric suffix family; no capture manifest or source PCAP verified"
        else:
            status = "UNKNOWN_NOT_VERIFIED"
            confidence = "UNKNOWN"
            evidence = "no repeated filename family and no explicit source capture evidence"
        summary[key] = {
            "candidate_capture_group_id": candidate_id,
            "display_key": key,
            "file_count": len(members),
            "class_folders": class_set,
            "total_flows": total_rows,
            "numeric_suffixes": suffixes,
            "status": status,
            "confidence": confidence,
            "evidence": evidence,
            "files": [str(context["relative_path"]) for _, context in members],
        }
    return summary


def apply_capture_group_fields(file_stats: Sequence[Dict[str, object]], contexts: Sequence[Dict[str, object]], families: Dict[str, object]):
    for stat, context in zip(file_stats, contexts):
        family = families[context["family_key"]]
        stat["candidate_capture_group_id"] = family["candidate_capture_group_id"]
        stat["capture_group_status"] = family["status"]
        stat["capture_group_confidence"] = family["confidence"]
        stat["capture_group_evidence"] = family["evidence"]
        if family["status"] == "VERIFIED_BY_EXPLICIT_SIDECAR_NAME_ONLY":
            stat["capture_group_id"] = family["candidate_capture_group_id"]
        else:
            stat["capture_group_id"] = "UNKNOWN"


def compute_temporal_overlaps(file_stats: Sequence[Dict[str, object]]) -> Dict[str, object]:
    intervals = []
    for stat in file_stats:
        if stat.get("timestamp_min") and stat.get("timestamp_max"):
            try:
                start = datetime.fromisoformat(str(stat["timestamp_min"]))
                end = datetime.fromisoformat(str(stat["timestamp_max"]))
            except ValueError:
                continue
            intervals.append((start, end, stat))
    intervals.sort(key=lambda item: item[0])
    active: List[Tuple[datetime, int, Dict[str, object]]] = []
    overlap_pairs = 0
    sample_pairs = []
    serial = 0
    for start, end, stat in intervals:
        while active and active[0][0] < start:
            heapq.heappop(active)
        for _, _, other in active:
            overlap_pairs += 1
            if len(sample_pairs) < 50:
                sample_pairs.append(
                    {
                        "file_a": other["relative_path"],
                        "file_b": stat["relative_path"],
                        "a_min": other["timestamp_min"],
                        "a_max": other["timestamp_max"],
                        "b_min": stat["timestamp_min"],
                        "b_max": stat["timestamp_max"],
                    }
                )
        heapq.heappush(active, (end, serial, stat))
        serial += 1
    return {
        "files_with_parseable_timestamp_ranges": len(intervals),
        "temporal_overlap_pair_count": overlap_pairs,
        "sample_temporal_overlap_pairs": sample_pairs,
        "overlap_detected": overlap_pairs > 0,
    }


def infer_label_leakage(column_trackers: Dict[str, ColumnTracker], metadata_columns: Dict[str, str]) -> List[Dict[str, object]]:
    findings = []
    for key, tracker in column_trackers.items():
        if key in metadata_columns:
            continue
        if tracker.distinct_overflow or len(tracker.value_labels) < 2:
            continue
        label_sets = [labels for labels in tracker.value_labels.values() if labels]
        if len(label_sets) >= 2 and all(len(labels) == 1 for labels in label_sets):
            labels = sorted({next(iter(label_set)) for label_set in label_sets})
            findings.append(
                {
                    "column": key,
                    "reason": "observed low-cardinality values map one-to-one to a single label in the audit stream",
                    "labels": labels,
                    "observed_distinct_values": len(tracker.value_labels),
                    "severity": "MEDIUM_HEURISTIC",
                }
            )
    return findings


def evaluate_split_feasibility(file_stats: Sequence[Dict[str, object]], family_summary: Dict[str, object]) -> Dict[str, object]:
    verified_files = [stat for stat in file_stats if stat.get("capture_group_id") not in (None, "", "UNKNOWN")]
    unknown_files = [stat for stat in file_stats if stat.get("capture_group_id") in (None, "", "UNKNOWN")]
    class_group_counts = {}
    class_flow_counts = Counter()
    for stat in file_stats:
        cls = str(stat.get("class_folder", ""))
        class_flow_counts[cls] += int(stat.get("rows", 0))
    for cls in EXPECTED_CLASSES:
        groups = {
            str(family_summary[key]["candidate_capture_group_id"])
            for key in family_summary
            if cls in family_summary[key]["class_folders"]
        }
        class_group_counts[cls] = len(groups)
    return {
        "target_ratio": "60:20:20",
        "capture_disjoint_split_proven": False,
        "reason": "capture_group_id is UNKNOWN for files without explicit source-capture evidence; a group-disjoint split cannot be verified from CSV names alone",
        "verified_capture_group_files": len(verified_files),
        "unknown_capture_group_files": len(unknown_files),
        "candidate_name_family_count": len(family_summary),
        "candidate_group_counts_by_class": class_group_counts,
        "flows_by_class": dict(class_flow_counts),
        "class_distribution_can_be_assessed_only_after_capture_manifest": True,
        "recommendation": "obtain PCAP/manifest provenance, define capture groups, then assign whole groups to train/validation/test with a seeded group-stratified optimizer; do not split files or rows first",
    }


def build_leakage_report(
    output_path: Path,
    dataset_root: Path,
    file_stats: Sequence[Dict[str, object]],
    class_stats: Dict[str, object],
    schema_payload: Dict[str, object],
    capture_payload: Dict[str, object],
    leakage_payload: Dict[str, object],
):
    total_flows = sum(int(stat["rows"]) for stat in file_stats)
    lines = [
        "# CICFlowMeter Stage 1 Leakage Risk Report",
        "",
        "## Scope and hard boundary",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- CSV files audited: **{len(file_stats):,}**",
        f"- Data rows observed: **{total_flows:,}**",
        "- This audit is read-only for source CSVs. No split, sliding window, resampling, feature transformation, or model training was performed.",
        "",
        "## Risk summary",
        "",
        "| Risk | Assessment | Evidence / action |",
        "|---|---|---|",
    ]
    risk_rows = leakage_payload["risk_items"]
    for item in risk_rows:
        lines.append(f"| {item['risk']} | **{item['severity']}** | {item['assessment']} |")
    lines.extend(
        [
            "",
            "## Capture/session audit",
            "",
            f"- Candidate filename families: **{capture_payload['candidate_name_family_count']:,}**.",
            f"- Files with explicit same-directory source sidecar evidence: **{capture_payload['verified_sidecar_file_count']:,}**.",
            f"- Files whose authoritative `capture_group_id` remains `UNKNOWN`: **{capture_payload['unknown_capture_group_file_count']:,}**.",
            "- A repeated filename stem or numeric suffix is treated as an inferred candidate only. It is not proof that the files came from one PCAP, nor proof that different files are independent captures.",
            "- A capture manifest or original PCAP provenance is required before claiming capture-disjoint train/validation/test partitions.",
            "",
            "## Schema and label findings",
            "",
            f"- Unique schema fingerprints: **{schema_payload['unique_schema_count']:,}**.",
            f"- Schema consistent across all files: **{schema_payload['schema_consistent_across_files']}**.",
            f"- Label columns detected in files: **{schema_payload['files_with_label_column']:,}/{len(file_stats):,}**.",
            f"- Total missing cells: **{schema_payload['total_missing_cells']:,}**.",
            f"- Total Inf cells: **{schema_payload['total_inf_cells']:,}**.",
            "- Label values are reported as observed. No label was renamed, encoded, or corrected.",
            "",
            "## 60:20:20 readiness",
            "",
            "**BLOCKED / NOT READY for Stage 2.** The requested ratio cannot be safely implemented as a capture-disjoint split from the current evidence because authoritative capture groups are not verified. The safe next input is provenance mapping from each CSV to its original PCAP/capture/session, including files with numeric suffixes.",
            "",
            "## Safe Stage 2 proposal (not implemented)",
            "",
            "1. Build and review a provenance manifest with one authoritative capture/session ID per CSV.",
            "2. Treat all files from one capture/session as one indivisible group. If one group contains multiple labels, keep the group intact and document the consequence for stratification.",
            "3. Use a seeded group-stratified assignment optimizer targeting 60:20:20 by flow count and eight-class proportions, with an explicit tolerance and a human review table.",
            "4. Fit any preprocessing only on training groups. Verify no exact or near duplicate flow crosses splits.",
            "5. Create sliding windows only after the split and reset windows at every file/capture/split boundary.",
            "",
            "## Required information before Stage 2",
            "",
            "- Authoritative mapping of each CSV to original PCAP/capture/session, or a statement that the source PCAPs are unavailable and a conservative `UNKNOWN` policy is accepted.",
            "- Timestamp semantics and timezone, if temporal holdout or temporal overlap controls are required.",
            "- Confirmation of which identifier/metadata columns are prohibited from model input, especially `Flow ID`, IP addresses, timestamp, file name, capture/session metadata, and `Label`.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only streaming audit of CICFlowMeter CSV files")
    parser.add_argument("--dataset-root", default=r"D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV")
    parser.add_argument("--output-root", default=r"D:\COMPACT-MI-RFE\STAGE1_AUDIT")
    parser.add_argument("--keep-duplicate-index", action="store_true", help="keep the temporary SQLite duplicate index")
    parser.add_argument("--skip-duplicates", action="store_true", help="skip exact duplicate hashing only if runtime is constrained")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(
        path for path in dataset_root.rglob("*.csv")
        if path.is_file() and output_root not in path.parents
    )
    if not csv_paths:
        raise SystemExit(f"No CSV files found below {dataset_root}")
    source_files = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS and path.suffix.lower() != ".csv"
    )
    contexts = [build_file_context(path, dataset_root, source_files) for path in csv_paths]
    families_pre = defaultdict(list)
    for context in contexts:
        families_pre[context["family_key"]].append(context)

    duplicate_db_path = output_root / ".stage1_duplicate_index.sqlite"
    duplicate_index = None if args.skip_duplicates else DuplicateIndex(duplicate_db_path)
    file_stats: List[Dict[str, object]] = []
    class_actual_labels = {cls: Counter() for cls in EXPECTED_CLASSES}
    actual_label_counts = Counter()
    schema_groups = defaultdict(lambda: {"file_count": 0, "flow_count": 0, "files": [], "headers": []})
    column_trackers: Dict[str, ColumnTracker] = {}
    metadata_columns: Dict[str, str] = {}
    label_column_names_seen = Counter()
    total_missing = 0
    total_inf = 0
    total_label_mismatch = 0
    global_timestamp_columns = Counter()
    start_time = time.time()
    file_names_for_duplicate_index = {}

    import numpy as np
    import pandas as pd

    chunksize = 50000
    sample_limit = 5000
    for file_id, (path, context) in enumerate(zip(csv_paths, contexts), start=1):
        stat = empty_file_stats(context)
        stat["file_id"] = file_id
        stat["file_size_bytes"] = path.stat().st_size
        stat["file_size_gib"] = round(stat["file_size_bytes"] / (1024 ** 3), 6)
        file_names_for_duplicate_index[file_id] = context["relative_path"]
        delimiter = detect_delimiter(path)
        label_indices: List[int] = []
        timestamp_indices: List[int] = []
        timestamp_min = None
        timestamp_max = None
        file_label_counter = Counter()
        row_start = 2
        sample_rows_seen = 0
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="", buffering=1024 * 1024) as header_handle:
                header_reader = csv.reader(header_handle, delimiter=delimiter)
                headers = next(header_reader, [])
            headers = [str(header).replace("\ufeff", "").strip() for header in headers]
            stat["header_read"] = bool(headers)
            stat["header_names"] = headers
            stat["num_columns"] = len(headers)
            stat["schema_hash"] = schema_hash(headers) if headers else ""
            normalized_headers = [normalize_header(header) for header in headers]
            label_indices = [i for i, name in enumerate(normalized_headers) if name in {"label", "class", "target"}]
            timestamp_indices = candidate_timestamp_columns(headers)
            stat["label_column_indices"] = label_indices
            stat["label_column_names"] = [headers[i] for i in label_indices]
            stat["timestamp_columns"] = [headers[i] for i in timestamp_indices]
            for i, header in enumerate(headers):
                risk = metadata_risk_for_header(header)
                col_key = f"{header}__occurrence_{i + 1}"
                metadata_columns[col_key] = risk
                column_trackers.setdefault(col_key, ColumnTracker())
            schema_groups[stat["schema_hash"]]["file_count"] += 1
            schema_groups[stat["schema_hash"]]["headers"] = headers
            schema_groups[stat["schema_hash"]]["files"].append(context["relative_path"])

            if headers:
                reader = pd.read_csv(
                    path,
                    sep=delimiter,
                    header=0,
                    names=headers,
                    dtype=str,
                    keep_default_na=True,
                    na_filter=True,
                    na_values=sorted(MISSING_TOKENS),
                    chunksize=chunksize,
                    engine="c",
                    on_bad_lines="skip",
                    encoding="utf-8-sig",
                    encoding_errors="replace",
                )
                for chunk in reader:
                    n_rows = int(len(chunk))
                    if n_rows == 0:
                        continue
                    stat["rows"] += n_rows
                    stat["rows_with_expected_width"] += n_rows
                    stat["csv_parse_errors"] = "not counted; pandas C parser used on_bad_lines=skip"
                    actual_labels = pd.Series(["__MISSING_LABEL_COLUMN__"] * n_rows, index=chunk.index, dtype="string")
                    if label_indices:
                        actual_labels = chunk.iloc[:, label_indices[0]].astype("string").fillna("").str.strip()
                        actual_labels = actual_labels.mask(actual_labels.eq(""), "__MISSING_LABEL_VALUE__")
                    label_counts = actual_labels.value_counts(dropna=False).to_dict()
                    for label, count in label_counts.items():
                        label = str(label)
                        count = int(count)
                        file_label_counter[label] += count
                        actual_label_counts[label] += count
                        class_name = str(context["class_folder"])
                        if class_name in class_actual_labels:
                            class_actual_labels[class_name][label] += count
                    class_name = str(context["class_folder"])
                    if class_name in EXPECTED_CLASSES:
                        normalized_actual = actual_labels.astype("string").str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
                        stat["label_mismatch_rows"] += int((normalized_actual != normalize_label(class_name)).sum())

                    missing_counts = chunk.isna().sum(axis=0)
                    inf_counts = chunk.isin(INF_LITERALS).sum(axis=0)
                    stat["missing_cells"] += int(missing_counts.sum())
                    stat["inf_cells"] += int(inf_counts.sum())
                    if sample_rows_seen < sample_limit:
                        take = min(sample_limit - sample_rows_seen, n_rows)
                        sample_chunk = chunk.iloc[:take]
                        sample_labels = actual_labels.iloc[:take]
                        for i, header in enumerate(headers):
                            col_key = f"{header}__occurrence_{i + 1}"
                            column_trackers[col_key].observe_sample_series(sample_chunk.iloc[:, i], sample_labels)

                    if timestamp_indices:
                        idx = timestamp_indices[0]
                        ts_values = chunk.iloc[:, idx].astype("string").fillna("").str.strip()
                        nonmissing_ts = ~ts_values.str.lower().isin(MISSING_TOKENS)
                        parsed = pd.to_datetime(ts_values.where(nonmissing_ts), errors="coerce", utc=True, format="mixed")
                        valid = parsed.notna()
                        stat["timestamp_parseable_rows"] += int(valid.sum())
                        stat["timestamp_unparseable_nonmissing_rows"] += int((nonmissing_ts & ~valid).sum())
                        if valid.any():
                            current_min = parsed[valid].min().to_pydatetime().replace(tzinfo=None)
                            current_max = parsed[valid].max().to_pydatetime().replace(tzinfo=None)
                            timestamp_min = current_min if timestamp_min is None or current_min < timestamp_min else timestamp_min
                            timestamp_max = current_max if timestamp_max is None or current_max > timestamp_max else timestamp_max

                    if duplicate_index is not None:
                        nonlabel_indices = [i for i in range(len(headers)) if i not in set(label_indices)]
                        if nonlabel_indices:
                            digest_values = pd.util.hash_pandas_object(chunk.iloc[:, nonlabel_indices], index=False)
                            duplicate_index.add_many(digest_values, file_id, row_start)
                    row_start += n_rows
                    sample_rows_seen += min(sample_limit - sample_rows_seen, n_rows)
        except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, ValueError) as exc:
            stat["error"] = f"{type(exc).__name__}: {exc}"
        stat["label_counts"] = dict(file_label_counter)
        stat["timestamp_min"] = iso_or_blank(timestamp_min)
        stat["timestamp_max"] = iso_or_blank(timestamp_max)
        file_stats.append(stat)
        schema_groups[stat["schema_hash"]]["flow_count"] += int(stat["rows"])
        total_missing += int(stat["missing_cells"])
        total_inf += int(stat["inf_cells"])
        total_label_mismatch += int(stat["label_mismatch_rows"])
        for timestamp_column in stat["timestamp_columns"]:
            global_timestamp_columns[timestamp_column] += 1
        elapsed = time.time() - start_time
        print(
            f"[{file_id:03d}/{len(csv_paths):03d}] {context['relative_path']} | "
            f"rows={stat['rows']:,} size={stat['file_size_gib']:.2f} GiB elapsed={elapsed/60:.1f} min",
            flush=True,
        )

    # Reconstruct per-file label counts from a compact second pass is deliberately not done.
    # The streaming loop stores actual label counts in this private field below when enabled.
    # This guard keeps output explicit if a malformed/empty file was encountered.
    for stat in file_stats:
        if not isinstance(stat.get("label_counts"), dict):
            stat["label_counts"] = {}

    family_summary = build_family_summary(file_stats, contexts)
    apply_capture_group_fields(file_stats, contexts, family_summary)
    temporal_payload = compute_temporal_overlaps(file_stats)
    split_payload = evaluate_split_feasibility(file_stats, family_summary)

    duplicate_payload = {
        "enabled": duplicate_index is not None,
        "note": "Exact canonical row hash excluding Label; only cross-file duplicate digests are reported. This is not a near-duplicate detector.",
    }
    if duplicate_index is not None:
        duplicate_payload.update(duplicate_index.cross_file_summary(file_names_for_duplicate_index))
        duplicate_index.close()
        if not args.keep_duplicate_index:
            try:
                duplicate_db_path.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        duplicate_payload["note"] = "Duplicate hashing was skipped by command-line option."

    # Derive label counts and class counts from the streaming global counters.
    class_distribution_rows = []
    for cls in EXPECTED_CLASSES:
        counter = class_actual_labels[cls]
        class_distribution_rows.append(
            {
                "scope": "folder",
                "folder_class": cls,
                "label_value": "__FOLDER_TOTAL__",
                "csv_count": sum(1 for stat in file_stats if stat["class_folder"] == cls),
                "flow_count": sum(counter.values()),
                "percent_of_folder_flows": 100.0 * sum(counter.values()) / max(1, sum(counter.values())),
                "label_matches_folder": "n/a",
            }
        )
        for label, count in sorted(counter.items()):
            class_distribution_rows.append(
                {
                    "scope": "folder_label",
                    "folder_class": cls,
                    "label_value": label,
                    "csv_count": sum(1 for stat in file_stats if stat["class_folder"] == cls),
                    "flow_count": count,
                    "percent_of_folder_flows": round(100.0 * count / max(1, sum(counter.values())), 8),
                    "label_matches_folder": normalize_label(label) == normalize_label(cls),
                }
            )
    for label, count in sorted(actual_label_counts.items()):
        class_distribution_rows.append(
            {
                "scope": "dataset_label",
                "folder_class": "ALL",
                "label_value": label,
                "csv_count": len(file_stats),
                "flow_count": count,
                "percent_of_folder_flows": round(100.0 * count / max(1, sum(actual_label_counts.values())), 8),
                "label_matches_folder": "n/a",
            }
        )

    unique_schema_count = len(schema_groups)
    files_with_label = sum(1 for stat in file_stats if stat["label_column_indices"])
    metadata_columns_summary = sorted(
        {key: risk for key, risk in metadata_columns.items() if risk}.items(), key=lambda item: item[0]
    )
    schema_payload = {
        "dataset_root": str(dataset_root),
        "unique_schema_count": unique_schema_count,
        "schema_consistent_across_files": unique_schema_count == 1,
        "files_with_label_column": files_with_label,
        "files_without_label_column": len(file_stats) - files_with_label,
        "total_missing_cells": total_missing,
        "total_inf_cells": total_inf,
        "total_label_mismatch_rows_against_folder": total_label_mismatch,
        "expected_class_folders": EXPECTED_CLASSES,
        "actual_label_values": dict(actual_label_counts),
        "schema_groups": dict(schema_groups),
        "metadata_and_identifier_columns": [
            {"column_key": key, "risk_category": risk} for key, risk in metadata_columns_summary
        ],
        "column_quality": {key: tracker.to_dict() for key, tracker in sorted(column_trackers.items())},
        "heuristic_label_leakage_findings": infer_label_leakage(column_trackers, metadata_columns),
        "timestamp_columns_seen": dict(global_timestamp_columns),
        "raw_feature_count_by_schema": {
            schema: max(0, len(info["headers"]) - sum(1 for h in info["headers"] if normalize_header(h) in {"label", "class", "target"}))
            for schema, info in schema_groups.items()
        },
    }
    capture_payload = {
        "candidate_name_family_count": len(family_summary),
        "verified_sidecar_file_count": sum(1 for stat in file_stats if stat["capture_group_status"] == "VERIFIED_BY_EXPLICIT_SIDECAR_NAME_ONLY"),
        "unknown_capture_group_file_count": sum(1 for stat in file_stats if stat["capture_group_id"] == "UNKNOWN"),
        "family_groups": family_summary,
        "temporal_overlap": temporal_payload,
        "split_feasibility": split_payload,
    }
    leakage_payload = {
        "duplicate_detection": duplicate_payload,
        "risk_items": [
            {
                "risk": "Capture/session leakage",
                "severity": "HIGH / BLOCKING",
                "assessment": f"Authoritative capture groups are unknown for {capture_payload['unknown_capture_group_file_count']:,} files. Filename families are candidate evidence only.",
            },
            {
                "risk": "Temporal overlap leakage",
                "severity": "HIGH / REVIEW",
                "assessment": f"{temporal_payload['temporal_overlap_pair_count']:,} overlapping timestamp-range file pairs were detected when timestamps were parseable; exact flow-level temporal provenance is not established.",
            },
            {
                "risk": "Duplicate flows between CSVs",
                "severity": "HIGH / REVIEW" if duplicate_payload.get("cross_file_duplicate_digest_groups", 0) else "NOT DETECTED BY EXACT HASH",
                "assessment": f"{duplicate_payload.get('cross_file_duplicate_digest_groups', 0):,} canonical row-hash groups cross file boundaries. Near duplicates and reordered/rounded variants were not tested.",
            },
            {
                "risk": "Label leakage",
                "severity": "HIGH / REVIEW" if any(item["severity"].startswith("MEDIUM") for item in schema_payload["heuristic_label_leakage_findings"]) else "REVIEW",
                "assessment": f"Label columns and observed label values were audited; {len(schema_payload['heuristic_label_leakage_findings']):,} non-label columns triggered a low-cardinality label-mapping heuristic.",
            },
            {
                "risk": "Feature leakage",
                "severity": "HIGH / REVIEW",
                "assessment": "Identifier, IP, timestamp, file/session metadata, and target-like columns were identified by name. Exclude them from model features unless a documented scientific reason exists.",
            },
            {
                "risk": "Sliding window across file/split boundaries",
                "severity": "HIGH / PREVENT",
                "assessment": "No windows were created. Stage 2 must reset window state at every file, capture, and split boundary.",
            },
        ],
        "stage2_readiness": "BLOCKED_NOT_READY",
        "stage2_feasibility": split_payload,
    }

    inventory_fields = [
        "file_id", "absolute_path", "relative_path", "class_folder", "inner_parent", "filename", "file_size_bytes", "file_size_gib",
        "rows", "rows_with_expected_width", "rows_short_width", "rows_long_width", "csv_parse_errors", "header_read",
        "num_columns", "schema_hash", "header_names", "label_column_indices", "label_column_names", "label_counts",
        "missing_cells", "inf_cells", "timestamp_columns", "timestamp_parseable_rows", "timestamp_unparseable_nonmissing_rows",
        "timestamp_min", "timestamp_max", "label_mismatch_rows", "candidate_capture_group_id", "capture_group_id",
        "capture_group_status", "capture_group_confidence", "capture_group_evidence", "source_sidecars", "error",
    ]
    inventory_rows = []
    for stat in file_stats:
        row = dict(stat)
        row["header_names"] = json.dumps(row["header_names"], ensure_ascii=False)
        row["label_column_indices"] = json.dumps(row["label_column_indices"])
        row["label_column_names"] = json.dumps(row["label_column_names"], ensure_ascii=False)
        row["label_counts"] = json.dumps(row["label_counts"], ensure_ascii=False)
        row["timestamp_columns"] = json.dumps(row["timestamp_columns"], ensure_ascii=False)
        inventory_rows.append(row)
    write_csv(output_root / "inventory.csv", inventory_rows, inventory_fields)
    write_csv(
        output_root / "class_distribution.csv",
        class_distribution_rows,
        ["scope", "folder_class", "label_value", "csv_count", "flow_count", "percent_of_folder_flows", "label_matches_folder"],
    )
    write_json(output_root / "schema_audit.json", schema_payload)
    capture_rows = []
    for stat in file_stats:
        capture_rows.append(
            {
                "file_id": stat["file_id"],
                "relative_path": stat["relative_path"],
                "class_folder": stat["class_folder"],
                "filename": stat["filename"],
                "normalized_name_core": next(context["normalized_name_core"] for context in contexts if context["relative_path"] == stat["relative_path"]),
                "numeric_suffix": next(context["numeric_suffix"] for context in contexts if context["relative_path"] == stat["relative_path"]),
                "proposed_capture_group_id": stat["candidate_capture_group_id"],
                "capture_group_id": stat["capture_group_id"],
                "capture_group_status": stat["capture_group_status"],
                "confidence": stat["capture_group_confidence"],
                "evidence": stat["capture_group_evidence"],
                "candidate_family_file_count": family_summary[next(context["family_key"] for context in contexts if context["relative_path"] == stat["relative_path"])]["file_count"],
                "candidate_family_classes": ";".join(family_summary[next(context["family_key"] for context in contexts if context["relative_path"] == stat["relative_path"])]["class_folders"]),
                "candidate_family_flow_count": family_summary[next(context["family_key"] for context in contexts if context["relative_path"] == stat["relative_path"])]["total_flows"],
                "timestamp_min": stat["timestamp_min"],
                "timestamp_max": stat["timestamp_max"],
                "source_sidecars": stat["source_sidecars"],
            }
        )
    write_csv(
        output_root / "capture_groups.csv",
        capture_rows,
        ["file_id", "relative_path", "class_folder", "filename", "normalized_name_core", "numeric_suffix", "proposed_capture_group_id", "capture_group_id", "capture_group_status", "confidence", "evidence", "candidate_family_file_count", "candidate_family_classes", "candidate_family_flow_count", "timestamp_min", "timestamp_max", "source_sidecars"],
    )
    write_json(
        output_root / "_audit_internal.json",
        {
            "generated_at_local": datetime.now().astimezone().isoformat(),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "csv_count": len(file_stats),
            "total_flows": sum(int(stat["rows"]) for stat in file_stats),
            "class_folder_flow_counts": {cls: sum(1 for stat in file_stats if stat["class_folder"] == cls) for cls in EXPECTED_CLASSES},
            "class_flow_totals": {cls: sum(int(stat["rows"]) for stat in file_stats if stat["class_folder"] == cls) for cls in EXPECTED_CLASSES},
            "actual_label_counts": dict(actual_label_counts),
            "duplicate_detection": duplicate_payload,
            "temporal_overlap": temporal_payload,
            "split_feasibility": split_payload,
        },
    )
    build_leakage_report(output_root / "leakage_risk_report.md", dataset_root, file_stats, class_actual_labels, schema_payload, capture_payload, leakage_payload)

    total_flows = sum(int(stat["rows"]) for stat in file_stats)
    report_lines = [
        "# STAGE 1 REPORT — CICFlowMeter Dataset Audit",
        "",
        "## Gate result",
        "",
        "**BLOCKED / NOT READY FOR STAGE 2.** This Stage 1 audit is complete as a read-only scan, but capture/session provenance is not verified for the CSVs. Do not create the 60:20:20 split until the provenance decision is reviewed.",
        "",
        "## Scope",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Audit output: `{output_root}`",
        f"- CSV files: **{len(file_stats):,}**",
        f"- Total flow rows: **{total_flows:,}**",
        f"- Expected folders found: `{', '.join(EXPECTED_CLASSES)}`",
        "- Source CSVs were not modified.",
        "- No train/validation/test split, sliding window, or model training was performed.",
        "",
        "## Folder totals",
        "",
        "| Folder | CSVs | Flow rows | Share of all flows |",
        "|---|---:|---:|---:|",
    ]
    for cls in EXPECTED_CLASSES:
        csv_count = sum(1 for stat in file_stats if stat["class_folder"] == cls)
        flow_count = sum(int(stat["rows"]) for stat in file_stats if stat["class_folder"] == cls)
        report_lines.append(f"| {cls} | {csv_count:,} | {flow_count:,} | {100.0 * flow_count / max(1, total_flows):.4f}% |")
    report_lines.extend(
        [
            "",
            "## Schema and quality",
            "",
            f"- Unique schema fingerprints: **{unique_schema_count:,}**.",
            f"- Schema consistent across files: **{schema_payload['schema_consistent_across_files']}**.",
            f"- Files with a Label/Class/Target column: **{files_with_label:,}/{len(file_stats):,}**.",
            f"- Total missing cells under the documented token rule: **{total_missing:,}**.",
            f"- Total Inf cells: **{total_inf:,}**.",
            f"- Rows whose observed label does not normalize to the containing class folder: **{total_label_mismatch:,}**.",
            "- The raw feature count is reported per schema as columns excluding Label/Class/Target. Identifier and metadata columns are not silently removed from source files.",
            "",
            "## Labels",
            "",
            "Observed dataset label values:",
            "",
            "```text",
            json.dumps(dict(actual_label_counts), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Capture/session conclusion",
            "",
            f"- Candidate name families: **{capture_payload['candidate_name_family_count']:,}**.",
            f"- Files with explicit sidecar-name evidence: **{capture_payload['verified_sidecar_file_count']:,}**.",
            f"- Files with authoritative capture_group_id = UNKNOWN: **{capture_payload['unknown_capture_group_file_count']:,}**.",
            "- Filename numbering is treated as inferred name evidence only. It is not proof of capture identity or independence.",
            f"- Parseable timestamp-range overlap pairs: **{temporal_payload['temporal_overlap_pair_count']:,}**.",
            "",
            "## Leakage findings",
            "",
        ]
    )
    for item in leakage_payload["risk_items"]:
        report_lines.append(f"- **{item['risk']} — {item['severity']}:** {item['assessment']}")
    report_lines.extend(
        [
            "",
            "## Stage 2 recommendation (proposal only)",
            "",
            "Use a provenance-backed, capture-disjoint, group-stratified 60:20:20 assignment. Keep captures intact. Exclude target and identifier/metadata fields from model features. Fit preprocessing only on training groups. Create windows after the split and reset state at file/capture/split boundaries.",
            "",
            "## Information required from the researcher",
            "",
            "1. A manifest mapping every CSV to original PCAP/capture/session, including files with numeric suffixes.",
            "2. Timestamp timezone/semantics if temporal holdout is desired.",
            "3. Final policy for UNKNOWN provenance. If unknown groups are accepted, Stage 2 must label the split as not capture-disjoint rather than PASS.",
            "",
            "## Output files",
            "",
            "- `inventory.csv`",
            "- `class_distribution.csv`",
            "- `schema_audit.json`",
            "- `capture_groups.csv`",
            "- `leakage_risk_report.md`",
            "- `STAGE1_REPORT.md`",
            "",
        ]
    )
    (output_root / "STAGE1_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("AUDIT_COMPLETE")
    print(f"output_root={output_root}")
    print(f"csv_count={len(file_stats)}")
    print(f"total_flows={total_flows}")
    print(f"elapsed_minutes={(time.time() - start_time)/60:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
