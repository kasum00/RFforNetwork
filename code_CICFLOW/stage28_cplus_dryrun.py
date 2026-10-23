"""CICFLOW Stage 2.8: metadata-only C+ adapted split dry-run.

Research assumptions are explicit. No raw feature CSV is opened or exported.
The ordered Stage 2.5 binary indexes are replayed and verified; whole global
(scenario gid, timestamp) buckets are constructed. A CICFLOW-specific 60/20/20
quantile anchor is used because CICIoMT24's frozen B2/C cutoffs are not
transferable. C+ local +/-300 second optimization and guards are then applied.
This is NOT a byte-for-byte reproduction of the original C+ policy.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import heapq
import json
import os
import shutil
import struct
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


VERSION = "stage28_cicflow_adapted_v1"
# Exact fingerprint of the first Stage 2.8 release. This one-time migration
# preserves already-verified per-gid checkpoints after the zero-count fix below.
ZERO_COUNT_FIX_PREVIOUS_SHA256 = "06f597874941b749d0d39118724fa789292286171d6cc4fca08aa6f882982915"
ROOT = Path(r"D:\COMPACT-MI-RFE")
EXPECTED_SOURCES = 133
EXPECTED_FLOWS = 8_877_213
EXPECTED_GROUPS = 33
EXPECTED_INVALID_DURATION = 5
EXPECTED_SHARED_TIMESTAMPS = 49
RECORD = struct.Struct("<qqqI")
BAD_TIMESTAMP, BAD_DURATION, BAD_KEY, NON_NO_LABEL = 1, 2, 4, 8
BAD_METADATA_MASK = BAD_TIMESTAMP | BAD_DURATION | BAD_KEY
MIN_GUARD = 511
MIN_ROLE = 512
RADIUS_US = 300 * 1_000_000
EPOCH = datetime(1970, 1, 1)
GROUP_FIELDS = (
    "gid", "category", "timestamp_us", "timestamp_iso", "flow_count",
    "raw_metadata_invalid_rows", "eligible_rows", "max_end_upper_us",
    "source_count", "source_members",
)
CLASS_BUDGET = {
    "BENIGN": 437_856, "BRUTE_FORCE": 9_417, "DDOS": 1_500_000,
    "DOS": 750_000, "MIRAI": 561_485, "RECON": 687_934,
    "SPOOFING": 198_126, "WEB-BASED": 506_810,
}
ROLES = ("train", "val", "test")


class DryRunError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DryRunError(message)


def as_int(value: str, context: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise DryRunError(f"Invalid integer {context}: {value!r}") from error
    require(number >= 0, f"Negative integer {context}: {number}")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames is not None, f"No CSV header: {path}")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"Duplicate CSV header: {path}")
        require(required.issubset(reader.fieldnames),
                f"Missing CSV columns in {path}: {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)
    require(bool(rows), f"Empty metadata: {path}")
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f"Malformed CSV row width: {path}")
    return rows


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing metadata: {path}")
    with path.open("r", encoding="utf-8-sig") as source:
        value = json.load(source)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".stage28_tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_csv_atomic(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    temp = path.with_name(path.name + ".stage28_tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def iso_us(value: int) -> str:
    return (EPOCH + timedelta(microseconds=value)).isoformat(timespec="seconds")


def load_inputs(repo: Path) -> dict:
    mpath = repo / "STAGE25_ORDERED_INDEX" / "ordered_index_manifest.csv"
    cpath = repo / "STAGE26_GROUP_MAPPING" / "candidate_gid_mapping.csv"
    spath = repo / "STAGE26_GROUP_MAPPING" / "scenario_group_summary.csv"
    dpath = repo / "STAGE25_ORDERED_INDEX" / "duration_evidence.json"
    ppath = repo / "STAGE21_SUBSET_PLAN" / "representative_subset_manifest.csv"
    rpath = repo / "huyen_code" / "ciciomt24_split.py"
    manifest = read_csv(mpath, {
        "relative_path", "source_id", "category", "source_rows", "ordered_rows",
        "index_path", "index_bytes", "index_sha256", "source_local_order_pass",
        "source_local_bucket_pass", "row_identity_pass",
        "stage24_bucket_ledger_match", "status",
        "invalid_timestamp_rows", "invalid_duration_rows", "invalid_key_rows",
        "label_not_no_label_rows", "distinct_source_buckets",
    })
    candidates = read_csv(cpath, {
        "relative_path", "category", "scenario_candidate", "proposed_gid",
        "source_id_stage25", "capture_provenance_status",
        "cross_source_clock_status",
    })
    summaries = read_csv(spath, {
        "proposed_gid", "category", "scenario_candidate", "source_count",
        "exact_shared_timestamp_bucket_count",
    })
    proposed = read_csv(ppath, {
        "relative_path", "category", "selection_mode",
        "original_flow_count", "proposed_flow_count",
    })
    evidence = read_json(dpath)
    require(len(manifest) == len(candidates) == EXPECTED_SOURCES,
            "Expected exactly 133 Stage 2.5/2.6 selected sources")
    require(len(summaries) == EXPECTED_GROUPS, "Expected 33 scenario candidates")
    require(sum(as_int(s["exact_shared_timestamp_bucket_count"], "shared timestamp")
                for s in summaries) == EXPECTED_SHARED_TIMESTAMPS,
            "Stage 2.6 shared-timestamp evidence changed")
    require(evidence.get("status") == "CORROBORATED_BY_DATA_ONLY",
            "Duration assumption evidence changed")
    require(sha256_file(rpath) == evidence.get("reference_cplus_sha256"),
            "C+ reference source changed since Stage 2.5")
    c_by_path = {r["relative_path"]: r for r in candidates}
    m_by_path = {r["relative_path"]: r for r in manifest}
    require(len(c_by_path) == len(candidates) and len(m_by_path) == len(manifest),
            "Duplicate relative path in metadata")
    require(set(c_by_path) == set(m_by_path), "Stage 2.5/2.6 source sets differ")
    selected_plan = {r["relative_path"]: r for r in proposed
                     if r["selection_mode"] in {"FULL", "PARTIAL"}}
    require(len(selected_plan) == EXPECTED_SOURCES
            and set(selected_plan) == set(m_by_path),
            "Stage 2.1 selected source budgets differ from Stage 2.5/2.6")
    summary_by_gid = {r["proposed_gid"]: r for r in summaries}
    require(len(summary_by_gid) == EXPECTED_GROUPS, "Duplicate proposed gid")
    groups = defaultdict(list)
    source_budgets = {}
    source_categories = {}
    seen_ids = set()
    total = Counter()
    for path, row in m_by_path.items():
        proposal = c_by_path[path]
        sid = as_int(row["source_id"], f"source ID {path}")
        require(sid not in seen_ids, f"Duplicate source ID: {sid}")
        seen_ids.add(sid)
        require(row["source_id"] == proposal["source_id_stage25"]
                and row["category"] == proposal["category"],
                f"Stage 2.5/2.6 source mismatch: {path}")
        budget_row = selected_plan[path]
        require(budget_row["category"] == row["category"],
                f"Stage 2.1 category mismatch: {path}")
        gid = proposal["proposed_gid"]
        require(gid in summary_by_gid and row["category"] == summary_by_gid[gid]["category"],
                f"Unknown/mixed candidate gid: {path}")
        require(proposal["capture_provenance_status"] == "UNKNOWN"
                and proposal["cross_source_clock_status"] == "UNKNOWN",
                f"Stage 2.6 assumption status changed: {path}")
        require(row["status"] == "SOURCE_LOCAL_PASS_ONLY"
                and all(row[x] == "True" for x in (
                    "source_local_order_pass", "source_local_bucket_pass",
                    "row_identity_pass", "stage24_bucket_ledger_match")),
                f"Stage 2.5 index not locally verified: {path}")
        require(as_int(row["source_rows"], path) == as_int(row["ordered_rows"], path),
                f"Stage 2.5 source row mismatch: {path}")
        require(as_int(budget_row["original_flow_count"], path)
                == as_int(row["source_rows"], path),
                f"Stage 2.1 original row count differs: {path}")
        budget = as_int(budget_row["proposed_flow_count"], path)
        require(0 < budget <= as_int(row["source_rows"], path),
                f"Invalid Stage 2.1 source budget: {path}")
        source_budgets[sid] = budget
        source_categories[sid] = row["category"]
        require(all(row[x] == "0" for x in (
                    "invalid_timestamp_rows", "invalid_key_rows", "label_not_no_label_rows")),
                f"Unexpected Stage 2.5 timestamp/key/label metadata issue: {path}")
        index_path = Path(row["index_path"])
        require(index_path.is_file(), f"Missing ordered index: {index_path}")
        require(index_path.stat().st_size == as_int(row["index_bytes"], path)
                == as_int(row["ordered_rows"], path) * RECORD.size,
                f"Index byte count mismatch: {path}")
        groups[gid].append(row)
        total["flows"] += as_int(row["source_rows"], path)
        total["bad_duration"] += as_int(row["invalid_duration_rows"], path)
        total["source_buckets"] += as_int(row["distinct_source_buckets"], path)
    require(total["flows"] == EXPECTED_FLOWS
            and total["bad_duration"] == EXPECTED_INVALID_DURATION,
            "Stage 2.5 conservation/duration evidence changed")
    require(sum(source_budgets.values()) == sum(CLASS_BUDGET.values()),
            "Stage 2.1 per-file budgets do not sum to 4,651,628")
    require(all(sum(source_budgets[sid] for sid in source_budgets
                    if source_categories[sid] == category) == budget
                for category, budget in CLASS_BUDGET.items()),
            "Stage 2.1 per-file budgets differ from specified class budgets")
    require(set(groups) == set(summary_by_gid), "Group coverage mismatch")
    for gid, members in groups.items():
        require(len(members) == as_int(summary_by_gid[gid]["source_count"], gid),
                f"Group source count changed: {gid}")
        members.sort(key=lambda x: int(x["source_id"]))
    return {
        "groups": dict(groups), "summary": summary_by_gid, "total": dict(total),
        "source_budgets": source_budgets,
        "source_categories": source_categories,
        "input_sha256": {
            "stage25_manifest": sha256_file(mpath),
            "stage26_mapping": sha256_file(cpath),
            "stage26_summary": sha256_file(spath),
            "duration_evidence": sha256_file(dpath),
            "reference_cplus": sha256_file(rpath),
            "stage21_source_budgets": sha256_file(ppath),
        },
    }


def preflight(inputs: dict, output: Path) -> dict:
    total_rows = inputs["total"]["flows"]
    buckets = inputs["total"]["source_buckets"]
    source_index_bytes = total_rows * RECORD.size
    # Source member lists vary; this is an allowance, not a guaranteed cap.
    output_allowance = buckets * 220 * 2 + 1024 ** 3
    free = shutil.disk_usage(output.parent).free
    return {
        "stage": "2.8", "sources": EXPECTED_SOURCES, "rows": total_rows,
        "source_index_bytes_read_approx": source_index_bytes,
        "source_local_buckets_upper_bound": buckets,
        "recommended_output_free_bytes": output_allowance,
        "actual_free_bytes": free, "space_estimate_pass": free >= output_allowance,
        "note": "Allows per-gid checkpoints plus final global summary and 1 GiB reserve; not a hard upper bound.",
        "raw_flow_csvs_to_read": 0, "feature_csvs_to_write": 0,
        "duration_policy": "ASSUMED_MICROSECONDS_FOR_EXPERIMENT",
        "capture_provenance": "UNKNOWN", "cross_source_clock": "ASSUMED_COMMON_NAIVE",
        "baseline_policy": "CICFLOW_ADAPTED_QUANTILE_60_20_20_NOT_ORIGINAL_C_PLUS",
    }


def verify_indexes(inputs: dict) -> None:
    for members in inputs["groups"].values():
        for item in members:
            path = Path(item["index_path"])
            require(sha256_file(path) == item["index_sha256"],
                    f"Stage 2.5 ordered index SHA256 changed: {path}")


class SourceBuckets:
    """One verified source index; bounded state plus a source-local row-ID bitset."""

    def __init__(self, manifest: dict[str, str]):
        self.manifest = manifest
        self.path = Path(manifest["index_path"])
        self.sid = int(manifest["source_id"])
        self.expected = int(manifest["ordered_rows"])
        self.stream = self.path.open("rb")
        self.digest = hashlib.sha256()
        self.seen = bytearray((self.expected + 7) // 8)
        self.rows = 0
        self.bad_duration = 0
        self.last_key = None
        self.lookahead = self._read()
        self.closed = False

    def _read(self):
        raw = self.stream.read(RECORD.size)
        if not raw:
            return None
        require(len(raw) == RECORD.size, f"Truncated index: {self.path}")
        self.digest.update(raw)
        ts, rid, duration, flags = RECORD.unpack(raw)
        require(0 <= rid < self.expected, f"Original row out of range: {self.path}: {rid}")
        key = (ts, rid)
        require(self.last_key is None or key > self.last_key,
                f"Index chronological order changed: {self.path}")
        self.last_key = key
        byte, bit = divmod(rid, 8)
        require(not (self.seen[byte] & (1 << bit)),
                f"Duplicate original row identity: {self.path}: {rid}")
        self.seen[byte] |= 1 << bit
        self.rows += 1
        self.bad_duration += bool(flags & BAD_DURATION)
        require(flags & (BAD_TIMESTAMP | BAD_KEY | NON_NO_LABEL) == 0,
                f"Unexpected C+ flag in Stage 2.5 index: {self.path}: {rid}")
        return ts, rid, duration, flags

    def __iter__(self):
        return self

    def __next__(self):
        if self.lookahead is None:
            if not self.closed:
                self.finish()
            raise StopIteration
        first_rank = self.rows - 1
        timestamp = self.lookahead[0]
        n = invalid = 0
        max_end = None
        while self.lookahead is not None and self.lookahead[0] == timestamp:
            ts, _, duration, flags = self.lookahead
            n += 1
            if flags & BAD_METADATA_MASK:
                invalid += 1
            else:
                require(0 <= duration <= 315_576_000_000_000,
                        f"Duration violates C+ bound: {self.path}")
                end = ts + duration + 1_000_000
                max_end = end if max_end is None else max(max_end, end)
            self.lookahead = self._read()
        return {
            "sid": self.sid, "ts": timestamp, "n": n, "invalid": invalid,
            "max_end": max_end, "first_rank": first_rank,
            "last_rank": first_rank + n - 1,
        }

    def finish(self):
        require(self.rows == self.expected, f"Index rows changed: {self.path}")
        require(self.bad_duration == int(self.manifest["invalid_duration_rows"]),
                f"Duration flags changed: {self.path}")
        require(self.digest.hexdigest() == self.manifest["index_sha256"],
                f"Index SHA256 changed during read: {self.path}")
        self.stream.close()
        self.closed = True

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True


def group_checkpoint_binding(inputs: dict, gid: str) -> dict:
    return {
        "version": VERSION, "gid": gid,
        "mapping_sha256": inputs["input_sha256"]["stage26_mapping"],
        "source_indexes": [
            [int(item["source_id"]), item["index_sha256"]]
            for item in inputs["groups"][gid]
        ],
    }


def build_one_group(inputs: dict, gid: str, checkpoint_dir: Path) -> dict:
    group_file = checkpoint_dir / f"{gid}.buckets.csv"
    checkpoint_file = checkpoint_dir / f"{gid}.checkpoint.json"
    binding = group_checkpoint_binding(inputs, gid)
    if checkpoint_file.is_file() and group_file.is_file():
        saved = read_json(checkpoint_file)
        require(saved.get("input_binding") == binding,
                f"Checkpoint input changed; review before resume: {gid}")
        require(sha256_file(group_file) == saved.get("group_sha256"),
                f"Checkpoint group file changed: {gid}")
        return saved
    # An interruption can occur between publishing the group CSV and its JSON.
    # In that case rebuild this one group from immutable Stage 2.5 indexes.
    if checkpoint_file.exists() != group_file.exists():
        print(f"[resume {gid}] incomplete checkpoint pair; rebuilding group",
              flush=True)
    temp = checkpoint_dir / f"{gid}.buckets.incomplete.csv"
    members = inputs["groups"][gid]
    readers = []
    heap = []
    stats = Counter()
    sample_shared = []
    try:
        for item in members:
            reader = SourceBuckets(item)
            readers.append(reader)
            try:
                first = next(reader)
            except StopIteration:
                continue
            heapq.heappush(heap, (first["ts"], first["sid"], first, reader))
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=GROUP_FIELDS)
            writer.writeheader()
            while heap:
                ts = heap[0][0]
                at_time = []
                while heap and heap[0][0] == ts:
                    _, _, bucket, reader = heapq.heappop(heap)
                    at_time.append(bucket)
                    try:
                        nxt = next(reader)
                    except StopIteration:
                        nxt = None
                    if nxt is not None:
                        heapq.heappush(heap, (nxt["ts"], nxt["sid"], nxt, reader))
                at_time.sort(key=lambda x: x["sid"])
                n = sum(x["n"] for x in at_time)
                invalid = sum(x["invalid"] for x in at_time)
                # Whole-bucket preservation: a mixed valid/invalid timestamp is
                # entirely embargoed; this is a disclosed CICFLOW adaptation.
                eligible = n if invalid == 0 else 0
                valid_ends = [x["max_end"] for x in at_time if x["max_end"] is not None]
                end = max(valid_ends) if eligible and valid_ends else None
                writer.writerow({
                    "gid": gid, "category": inputs["summary"][gid]["category"],
                    "timestamp_us": ts, "timestamp_iso": iso_us(ts),
                    "flow_count": n, "raw_metadata_invalid_rows": invalid,
                    "eligible_rows": eligible, "max_end_upper_us": "" if end is None else end,
                    "source_count": len(at_time),
                    "source_members": ";".join(
                        f"{x['sid']}:{x['first_rank']}:{x['last_rank']}" for x in at_time),
                })
                stats["buckets"] += 1
                stats["flows"] += n
                stats["raw_invalid"] += invalid
                stats["eligible"] += eligible
                stats["whole_bucket_invalid_embargo"] += n if invalid else 0
                if len(at_time) > 1:
                    stats["shared_timestamps"] += 1
                    if len(sample_shared) < 5:
                        sample_shared.append(iso_us(ts))
            stream.flush()
            os.fsync(stream.fileno())
        require(stats["flows"] == sum(int(r["ordered_rows"]) for r in members),
                f"Global group does not conserve Stage 2.5 source rows: {gid}")
        require(stats["raw_invalid"] == sum(int(r["invalid_duration_rows"]) for r in members),
                f"Global group invalid-duration count changed: {gid}")
        require(stats["shared_timestamps"] == int(
                    inputs["summary"][gid]["exact_shared_timestamp_bucket_count"]),
                f"Cross-source timestamp overlap differs from Stage 2.6: {gid}")
        os.replace(temp, group_file)
        saved = {
            "input_binding": binding, "group_sha256": sha256_file(group_file),
            "stats": {**dict(stats), "shared_timestamps": stats["shared_timestamps"]},
            "shared_timestamp_samples": sample_shared,
        }
        write_json_atomic(checkpoint_file, saved)
        return saved
    finally:
        for reader in readers:
            reader.close()


def assemble_global(checkpoint_dir: Path, output: Path, gids: list[str],
                    plans: dict[str, dict]) -> Counter:
    final = output / "global_bucket_summary.csv"
    temp = output / "global_bucket_summary.incomplete.csv"
    totals = Counter()
    with temp.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=(*GROUP_FIELDS, "dryrun_role"))
        writer.writeheader()
        for gid in gids:
            source = checkpoint_dir / f"{gid}.buckets.csv"
            with source.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                for index, row in enumerate(reader):
                    role = bucket_role(plans[gid], index,
                                       as_int(row["raw_metadata_invalid_rows"], gid))
                    row["dryrun_role"] = role
                    writer.writerow(row)
                    totals[role] += as_int(row["flow_count"], gid)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, final)
    return totals


class GroupData:
    """Compact per-scenario arrays; only one group needs to be resident at once."""

    def __init__(self, path: Path, gid: str, category: str):
        self.gid, self.category = gid, category
        self.ts = array("q")
        self.n = array("q")
        self.eligible = array("q")
        self.invalid = array("q")
        self.end = array("q")
        self.cum_all = array("q")
        self.cum_eligible = array("q")
        self.prefix_end = array("q")
        neg_inf = -(1 << 63)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None
                    and set(GROUP_FIELDS).issubset(reader.fieldnames),
                    f"Invalid global group bucket header: {path}")
            previous = None
            for row in reader:
                require(row["gid"] == gid and row["category"] == category,
                        f"Mixed gid/category in group bucket file: {path}")
                ts = as_int(row["timestamp_us"], gid)
                require(previous is None or ts > previous, f"Unsorted global buckets: {gid}")
                previous = ts
                n = as_int(row["flow_count"], gid)
                invalid = as_int(row["raw_metadata_invalid_rows"], gid)
                eligible = as_int(row["eligible_rows"], gid)
                require(n > 0 and invalid <= n
                        and eligible == (0 if invalid else n),
                        f"Whole-bucket eligibility broken: {gid} {ts}")
                end = neg_inf if not row["max_end_upper_us"] else as_int(
                    row["max_end_upper_us"], gid)
                require((eligible == 0 and end == neg_inf)
                        or (eligible > 0 and end > ts),
                        f"Invalid activity end for global bucket: {gid} {ts}")
                self.ts.append(ts)
                self.n.append(n)
                self.invalid.append(invalid)
                self.eligible.append(eligible)
                self.end.append(end)
                self.cum_all.append(n + (self.cum_all[-1] if self.cum_all else 0))
                self.cum_eligible.append(
                    eligible + (self.cum_eligible[-1] if self.cum_eligible else 0))
                self.prefix_end.append(
                    max(end, self.prefix_end[-1]) if self.prefix_end else end)
        require(bool(self.ts), f"Empty scenario: {gid}")
        self.total = int(self.cum_all[-1])
        self.valid = int(self.cum_eligible[-1])
        self.neg_inf = neg_inf
        # Iterative segment tree for exact max activity end over any retained block.
        self.tree_size = 1 << (len(self.ts) - 1).bit_length()
        self.tree = array("q", [neg_inf]) * (2 * self.tree_size)
        for i, value in enumerate(self.end):
            self.tree[self.tree_size + i] = value
        for i in range(self.tree_size - 1, 0, -1):
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def prefix(self, values: array, index: int) -> int:
        return int(values[index]) if index >= 0 else 0

    def range_max(self, left: int, right: int) -> int:
        require(0 <= left <= right < len(self.ts), f"Invalid range_max: {self.gid}")
        left += self.tree_size
        right += self.tree_size + 1
        result = self.neg_inf
        while left < right:
            if left & 1:
                result = max(result, self.tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = max(result, self.tree[right])
            left //= 2
            right //= 2
        return result

    def gap_start(self, end_index: int, latest_end_upper: int) -> int:
        # Need >=511 WHOLE embargoed flows strictly between retained blocks.
        target = int(self.cum_all[end_index]) + MIN_GUARD
        guard_end = bisect.bisect_left(self.cum_all, target, lo=end_index + 1)
        first_after_guard = guard_end + 1
        first_after_activity = bisect.bisect_right(self.ts, latest_end_upper,
                                                  lo=end_index + 1)
        return max(first_after_guard, first_after_activity)

    def role_counts(self, plan: dict) -> tuple[int, int, int]:
        if plan["mode"] == "TRAIN_ONLY":
            return self.valid, 0, 0
        ti, vj, ve, xj = (plan[k] for k in ("train_end_i", "val_start_i",
                                             "val_end_i", "test_start_i"))
        return (
            int(self.cum_eligible[ti]),
            self.prefix(self.cum_eligible, ve) - self.prefix(self.cum_eligible, vj - 1),
            self.valid - self.prefix(self.cum_eligible, xj - 1),
        )


def ratio_objective(counts: tuple[int, int, int]) -> tuple:
    retained = sum(counts)
    if retained <= 0:
        return (1, float("inf"), float("inf"), 0)
    deviations = [abs(100 * counts[i] / retained - target)
                  for i, target in enumerate((60.0, 20.0, 20.0))]
    maximum = max(deviations)
    return (int(maximum > 0.5000001), maximum, sum(deviations), -retained)


def quantile_anchor(group: GroupData, fraction: float) -> int:
    target = max(1, int(group.valid * fraction + 0.999999999))
    return min(len(group.ts) - 1, bisect.bisect_left(group.cum_eligible, target))


def baseline_plan(group: GroupData) -> dict:
    """CICFLOW-adapted quantile anchor followed by C-style safety shortening.

    The original CICIoMT24 B2/C baseline used frozen scenario timestamps. This
    deterministic baseline is an explicitly approved adaptation, not original C+.
    """
    if group.valid < MIN_ROLE * 3 + MIN_GUARD * 2:
        return {"mode": "TRAIN_ONLY", "reason": "Insufficient valid flow budget for three >=512 roles plus two 511-flow guards"}
    train_anchor = quantile_anchor(group, 0.6)
    val_anchor = quantile_anchor(group, 0.8)
    # Move the earlier cutoff left as C does when activity/embargo safety needs
    # room. For each train prefix, choose the latest safe val suffix near 80%.
    for ti in range(train_anchor, -1, -1):
        if int(group.cum_eligible[ti]) < MIN_ROLE:
            break
        vj = group.gap_start(ti, int(group.prefix_end[ti]))
        if vj >= len(group.ts):
            continue
        before_val = group.prefix(group.cum_eligible, vj - 1)
        min_ve = bisect.bisect_left(group.cum_eligible, before_val + MIN_ROLE,
                                   lo=vj)
        if min_ve >= len(group.ts):
            continue
        hi = max(min_ve, val_anchor)
        if hi >= len(group.ts):
            continue

        def feasible_right(ve: int) -> tuple[bool, int]:
            val_end_upper = group.range_max(vj, ve)
            xj = group.gap_start(ve, val_end_upper)
            ok = (xj < len(group.ts)
                  and group.valid - group.prefix(group.cum_eligible, xj - 1)
                  >= MIN_ROLE)
            return ok, xj

        # Feasibility decreases as ve advances: smaller test supply, no shorter
        # embargo or activity guard. Find the latest feasible ve <= anchor.
        low, high = min_ve, hi
        best = None
        while low <= high:
            mid = (low + high) // 2
            ok, xj = feasible_right(mid)
            if ok:
                best = (mid, xj)
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            continue
        ve, xj = best
        plan = {
            "mode": "THREE_WAY", "train_end_i": ti, "val_start_i": vj,
            "val_end_i": ve, "test_start_i": xj,
            "quantile_train_anchor_i": train_anchor,
            "quantile_val_anchor_i": val_anchor,
            "reason": "CICFLOW 60/20/20 quantile anchor with C-style earlier-cutoff shortening",
        }
        if validate_group_plan(group, plan):
            return plan
    return {"mode": "TRAIN_ONLY", "reason": "No group-local safe three-way quantile baseline with 511-flow guards and activity-end separation"}


def validate_group_plan(group: GroupData, plan: dict) -> bool:
    if plan["mode"] == "TRAIN_ONLY":
        return True
    ti, vj, ve, xj = (plan[k] for k in ("train_end_i", "val_start_i",
                                         "val_end_i", "test_start_i"))
    if not 0 <= ti < vj <= ve < xj < len(group.ts):
        return False
    if min(group.role_counts(plan)) < MIN_ROLE:
        return False
    left_guard = group.prefix(group.cum_all, vj - 1) - group.prefix(group.cum_all, ti)
    right_guard = group.prefix(group.cum_all, xj - 1) - group.prefix(group.cum_all, ve)
    if left_guard < MIN_GUARD or right_guard < MIN_GUARD:
        return False
    if group.prefix_end[ti] >= group.ts[vj]:
        return False
    if group.range_max(vj, ve) >= group.ts[xj]:
        return False
    # Original C+ also checks Train -> Test; implied by Train -> Val when
    # timestamps strictly increase, but verify the invariant explicitly.
    return group.prefix_end[ti] < group.ts[xj]


def near_indices(group: GroupData, center_index: int) -> range:
    center = group.ts[center_index]
    left = bisect.bisect_left(group.ts, center - RADIUS_US)
    right = bisect.bisect_right(group.ts, center + RADIUS_US)
    return range(left, right)


def optimize_one_group(group: GroupData, current: dict,
                       global_counts: tuple[int, int, int],
                       minimum_retained: int) -> tuple[dict, tuple[int, int, int], list[dict], int]:
    """Reference C+ coordinate-search mechanics around adapted safe baseline."""
    if current["mode"] == "TRAIN_ONLY" or group.category == "BENIGN":
        return current, global_counts, [], 0
    plan = dict(current)
    old = group.role_counts(plan)
    examined = 0
    moves = []

    def propose(candidate: dict, current_global: tuple[int, int, int],
                old_local: tuple[int, int, int]):
        if not validate_group_plan(group, candidate):
            return None
        local = group.role_counts(candidate)
        new_global = tuple(current_global[i] - old_local[i] + local[i] for i in range(3))
        if sum(new_global) < minimum_retained:
            return None
        return new_global

    baseline = dict(plan)
    best_obj = ratio_objective(global_counts)
    best = None
    for vj in near_indices(group, baseline["val_start_i"]):
        if vj > plan["val_end_i"]:
            continue
        if group.range_max(vj, plan["val_end_i"]) >= group.ts[plan["test_start_i"]]:
            continue
        for ti in near_indices(group, baseline["train_end_i"]):
            examined += 1
            if ti >= vj:
                continue
            candidate = dict(plan, train_end_i=ti, val_start_i=vj)
            new_global = propose(candidate, global_counts, old)
            if new_global is None:
                continue
            obj = ratio_objective(new_global)
            if obj < best_obj:
                best_obj, best = obj, (candidate, new_global)
    if best:
        plan, global_counts = best
        old = group.role_counts(plan)
        moves.append({"gid": group.gid, "boundary": "train_to_val",
                      "train_end": iso_us(group.ts[plan["train_end_i"]]),
                      "val_start": iso_us(group.ts[plan["val_start_i"]])})

    baseline = dict(plan)
    best_obj = ratio_objective(global_counts)
    best = None
    for ve in near_indices(group, current["val_end_i"]):
        if ve < plan["val_start_i"]:
            continue
        latest_val = group.range_max(plan["val_start_i"], ve)
        val_count = group.prefix(group.cum_eligible, ve) - group.prefix(
            group.cum_eligible, plan["val_start_i"] - 1)
        if val_count < MIN_ROLE:
            continue
        for xj in near_indices(group, current["test_start_i"]):
            examined += 1
            if xj <= ve or latest_val >= group.ts[xj]:
                continue
            candidate = dict(plan, val_end_i=ve, test_start_i=xj)
            new_global = propose(candidate, global_counts, old)
            if new_global is None:
                continue
            obj = ratio_objective(new_global)
            if obj < best_obj:
                best_obj, best = obj, (candidate, new_global)
    if best:
        plan, global_counts = best
        moves.append({"gid": group.gid, "boundary": "val_to_test",
                      "val_end": iso_us(group.ts[plan["val_end_i"]]),
                      "test_start": iso_us(group.ts[plan["test_start_i"]])})
    return plan, global_counts, moves, examined


def bucket_role(plan: dict, index: int, invalid: int) -> str:
    if invalid:
        return "embargo"
    if plan["mode"] == "TRAIN_ONLY":
        return "train"
    if index <= plan["train_end_i"]:
        return "train"
    if plan["val_start_i"] <= index <= plan["val_end_i"]:
        return "val"
    if index >= plan["test_start_i"]:
        return "test"
    return "embargo"


def role_targets(total: int) -> dict[str, int]:
    train = total * 60 // 100
    val = total * 20 // 100
    return {"train": train, "val": val, "test": total - train - val}


def flow_allocation(category_caps: dict[str, int],
                    supplies: dict[tuple[str, str], int],
                    targets: dict[str, int]) -> tuple[int, dict[tuple[str, str], int]]:
    """Tiny 8-by-3 integer max-flow; no flow rows are selected or copied."""
    source = 0
    cats = {cat: i + 1 for i, cat in enumerate(CLASS_BUDGET)}
    roles = {role: len(cats) + i + 1 for i, role in enumerate(ROLES)}
    sink = max(roles.values()) + 1
    graph = [[] for _ in range(sink + 1)]
    edges = {}

    def add_edge(u: int, v: int, cap: int):
        forward = [v, cap, len(graph[v])]
        reverse = [u, 0, len(graph[u])]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for cat in CLASS_BUDGET:
        add_edge(source, cats[cat], category_caps[cat])
        for role in ROLES:
            edge = add_edge(cats[cat], roles[role],
                            supplies.get((cat, role), 0))
            edges[(cat, role)] = (edge, supplies.get((cat, role), 0))
    for role in ROLES:
        add_edge(roles[role], sink, targets[role])

    flow = 0
    while True:
        parent = [None] * len(graph)
        parent[source] = (-1, -1)
        queue = [source]
        for u in queue:
            for index, edge in enumerate(graph[u]):
                if edge[1] > 0 and parent[edge[0]] is None:
                    parent[edge[0]] = (u, index)
                    queue.append(edge[0])
        if parent[sink] is None:
            break
        increment = 1 << 60
        v = sink
        while v != source:
            u, index = parent[v]
            increment = min(increment, graph[u][index][1])
            v = u
        v = sink
        while v != source:
            u, index = parent[v]
            edge = graph[u][index]
            edge[1] -= increment
            graph[v][edge[2]][1] += increment
            v = u
        flow += increment
    used = {key: capacity - edge[1]
            for key, (edge, capacity) in edges.items()}
    return flow, used


def source_flow_allocation(inputs: dict, source_supplies: dict[tuple[int, str], int],
                           targets: dict[str, int]) -> tuple[int, dict[tuple[int, str], int]]:
    """Category -> selected CSV -> role max-flow with Stage 2.1 per-file caps."""
    categories = {cat: i + 1 for i, cat in enumerate(CLASS_BUDGET)}
    source_ids = sorted(inputs["source_budgets"])
    source_nodes = {sid: 1 + len(categories) + i for i, sid in enumerate(source_ids)}
    role_nodes = {role: 1 + len(categories) + len(source_ids) + i
                  for i, role in enumerate(ROLES)}
    sink = max(role_nodes.values()) + 1
    graph = [[] for _ in range(sink + 1)]
    source = 0
    edges = {}

    def add_edge(u: int, v: int, cap: int):
        forward = [v, cap, len(graph[v])]
        reverse = [u, 0, len(graph[u])]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for category, budget in CLASS_BUDGET.items():
        add_edge(source, categories[category], budget)
    for sid in source_ids:
        budget = inputs["source_budgets"][sid]
        add_edge(categories[inputs["source_categories"][sid]], source_nodes[sid], budget)
        for role in ROLES:
            cap = min(budget, source_supplies.get((sid, role), 0))
            edges[(sid, role)] = (add_edge(source_nodes[sid], role_nodes[role], cap), cap)
    for role in ROLES:
        add_edge(role_nodes[role], sink, targets[role])

    flow = 0
    while True:
        parent = [None] * len(graph)
        parent[source] = (-1, -1)
        queue = [source]
        for u in queue:
            for i, edge in enumerate(graph[u]):
                if edge[1] > 0 and parent[edge[0]] is None:
                    parent[edge[0]] = (u, i)
                    queue.append(edge[0])
        if parent[sink] is None:
            break
        delta = 1 << 60
        v = sink
        while v != source:
            u, i = parent[v]
            delta = min(delta, graph[u][i][1])
            v = u
        v = sink
        while v != source:
            u, i = parent[v]
            edge = graph[u][i]
            edge[1] -= delta
            graph[v][edge[2]][1] += delta
            v = u
        flow += delta
    used = {key: cap - edge[1] for key, (edge, cap) in edges.items()}
    return flow, used


def subset_review(inputs: dict, source_supplies: dict[tuple[int, str], int]) -> dict:
    full = sum(CLASS_BUDGET.values())
    exact_targets = role_targets(full)
    exact_flow, exact_plan = source_flow_allocation(inputs, source_supplies, exact_targets)
    upper = sum(min(inputs["source_budgets"][sid],
                    sum(source_supplies.get((sid, role), 0) for role in ROLES))
                for sid in inputs["source_budgets"])
    low, high = 0, upper
    best_plan = {}
    while low <= high:
        middle = (low + high) // 2
        achieved, allocation = source_flow_allocation(
            inputs, source_supplies, role_targets(middle))
        if achieved == middle:
            best_plan = allocation
            low = middle + 1
        else:
            high = middle - 1
    max_total = high
    if max_total and not best_plan:
        achieved, best_plan = source_flow_allocation(
            inputs, source_supplies, role_targets(max_total))
        require(achieved == max_total, "Subset max-flow inconsistency")
    by_category = {}
    for cat, budget in CLASS_BUDGET.items():
        ids = [sid for sid, category in inputs["source_categories"].items()
               if category == cat]
        available = {role: sum(source_supplies.get((sid, role), 0) for sid in ids)
                     for role in ROLES}
        by_category[cat] = {
            "stage21_budget": budget, "available_after_embargo": available,
            "available_total": sum(available.values()),
            "budget_shortfall": max(0, budget - sum(available.values())),
            "exact_budget_plan": {
                role: sum(exact_plan[(sid, role)] for sid in ids)
                for role in ROLES},
            "max_ratio_plan": {
                role: sum(best_plan.get((sid, role), 0) for sid in ids)
                for role in ROLES},
        }
    return {
        "target_subset_flows": full, "target_retained_ratio": [60, 20, 20],
        "exact_category_budgets_and_ratio_feasible": exact_flow == full,
        "exact_budget_flow_allocated": exact_flow,
        "maximum_total_with_category_caps_and_60_20_20": max_total,
        "gap_from_target": full - max_total,
        "category_supply_and_plan": by_category,
        "per_source_stage21_caps_enforced": True,
        "important_limit": "Capacity-only feasibility with 133 Stage 2.1 per-CSV caps: no source rows extracted; chronology, deduplication and within-role sampling not yet verified.",
    }


def calculate_plans(inputs: dict, checkpoint_dir: Path):
    gids = sorted(inputs["groups"])
    plans = {}
    baseline_counts = {}
    group_totals = {}
    for gid in gids:
        group = GroupData(checkpoint_dir / f"{gid}.buckets.csv", gid,
                          inputs["summary"][gid]["category"])
        plan = baseline_plan(group)
        require(validate_group_plan(group, plan),
                f"Unsafe quantile baseline unexpectedly accepted: {gid}")
        plans[gid] = plan
        baseline_counts[gid] = group.role_counts(plan)
        group_totals[gid] = group.total
        print(f"[baseline {gid}] {plan['mode']} {baseline_counts[gid]}", flush=True)
    global_counts = tuple(sum(counts[i] for counts in baseline_counts.values())
                          for i in range(3))
    minimum_retained = sum(global_counts)
    moves = []
    examined = 0
    for gid in sorted(gids, key=lambda x: (-group_totals[x], x)):
        group = GroupData(checkpoint_dir / f"{gid}.buckets.csv", gid,
                          inputs["summary"][gid]["category"])
        revised, global_counts, changed, checked = optimize_one_group(
            group, plans[gid], global_counts, minimum_retained)
        plans[gid] = revised
        moves.extend(changed)
        examined += checked
        print(f"[optimize {gid}] candidates={checked} accepted={len(changed)}",
              flush=True)
    return plans, {
        "baseline_retained": minimum_retained,
        "baseline_counts": dict(zip(ROLES, (
            sum(counts[i] for counts in baseline_counts.values()) for i in range(3)))),
        "candidates_examined": examined, "accepted_moves": moves,
        "optimized_counts": dict(zip(ROLES, global_counts)),
    }


def evaluate_plan(inputs: dict, checkpoint_dir: Path, plans: dict[str, dict]):
    scenario_rows = []
    cutoff_rows = []
    supplies = defaultdict(int)
    source_supplies = defaultdict(int)
    next_source_rank = defaultdict(int)
    role_totals = Counter()
    label_roles = defaultdict(lambda: Counter())
    embargo = Counter()
    checks = {"whole_gid_timestamp_bucket": "PASS_BY_ONE_ROLE_PER_GLOBAL_BUCKET",
              "source_row_identity": "PASS_RECHECKED_STAGE25_INDEX_AND_GLOBAL_RANKS",
              "source_local_chronology": "PASS_RECHECKED_STAGE25_INDEX",
              "connection_key_value_equivalence": "NOT_VERIFIED",
              "exact_cross_split_duplicates": "NOT_VERIFIED",
              "capture_session_independence": "NOT_VERIFIED",
              "cross_source_timezone": "ASSUMED_NOT_VERIFIED",
              "duration_unit": "ASSUMED_MICROSECONDS_NOT_VERIFIED"}
    activity_crossings = 0
    failed_groups = []
    for gid in sorted(inputs["groups"]):
        category = inputs["summary"][gid]["category"]
        group = GroupData(checkpoint_dir / f"{gid}.buckets.csv", gid, category)
        plan = plans[gid]
        counts = Counter()
        local_embargo = Counter()
        crossings = 0
        for i in range(len(group.ts)):
            role = bucket_role(plan, i, int(group.invalid[i]))
            count = int(group.n[i])
            counts[role] += count
            if role == "embargo":
                local_embargo["metadata_invalid_whole_bucket" if group.invalid[i]
                              else "temporal_gap"] += count
            if plan["mode"] == "THREE_WAY" and role in {"train", "val"}:
                next_start = (group.ts[plan["val_start_i"]] if role == "train"
                              else group.ts[plan["test_start_i"]])
                if group.end[i] >= next_start:
                    crossings += count
        # Attribution uses only Stage 2.8 bucket metadata (source ID plus
        # Stage 2.5 sorted-rank intervals), never the raw feature CSV.
        with (checkpoint_dir / f"{gid}.buckets.csv").open(
                "r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for i, row in enumerate(reader):
                role = bucket_role(plan, i,
                                   as_int(row["raw_metadata_invalid_rows"], gid))
                members_total = 0
                for token in row["source_members"].split(";"):
                    parts = token.split(":")
                    require(len(parts) == 3, f"Invalid source member token: {gid}")
                    sid, first, last = (as_int(value, gid) for value in parts)
                    require(sid in inputs["source_budgets"]
                            and inputs["source_categories"][sid] == category,
                            f"Unexpected source ID in global bucket: {gid} {sid}")
                    require(first == next_source_rank[sid] and last >= first,
                            f"Source rank gap/overlap in global bucket: {gid} {sid}")
                    next_source_rank[sid] = last + 1
                    n_source = last - first + 1
                    members_total += n_source
                    source_supplies[(sid, role)] += n_source
                require(members_total == as_int(row["flow_count"], gid),
                        f"Source-member flow count mismatch: {gid}")
        require(sum(counts.values()) == group.total,
                f"Split roles do not conserve global group: {gid}")
        activity_crossings += crossings
        for role in (*ROLES, "embargo"):
            role_totals[role] += counts[role]
            label_roles[role]["BENIGN" if category == "BENIGN" else "ATTACK"] += counts[role]
            if role in ROLES:
                supplies[(category, role)] += counts[role]
            scenario_rows.append({
                "gid": gid, "category": category, "scenario": inputs["summary"][gid]["scenario_candidate"],
                "split": role, "flows": counts[role], "scenario_total": group.total,
                "plan_mode": plan["mode"],
            })
        embargo.update(local_embargo)
        if plan["mode"] == "THREE_WAY":
            ti, vj, ve, xj = (plan[k] for k in ("train_end_i", "val_start_i",
                                                 "val_end_i", "test_start_i"))
            guard_left = group.prefix(group.cum_all, vj - 1) - group.prefix(group.cum_all, ti)
            guard_right = group.prefix(group.cum_all, xj - 1) - group.prefix(group.cum_all, ve)
            safe = validate_group_plan(group, plan)
            if not safe:
                failed_groups.append(gid)
            times = [iso_us(group.ts[x]) for x in (ti, vj, ve, xj)]
        else:
            guard_left = guard_right = 0
            times = ["", "", "", ""]
            if category == "BENIGN":
                failed_groups.append(gid)
        cutoff_rows.append({
            "gid": gid, "category": category,
            "scenario": inputs["summary"][gid]["scenario_candidate"],
            "mode": plan["mode"], "train_end": times[0], "val_start": times[1],
            "val_end": times[2], "test_start": times[3],
            "train_val_embargo_flows": guard_left,
            "val_test_embargo_flows": guard_right,
            "train_flows": counts["train"], "val_flows": counts["val"],
            "test_flows": counts["test"], "embargo_flows": counts["embargo"],
            "activity_crossing_flows": crossings, "reason": plan["reason"],
        })
    retained = sum(role_totals[role] for role in ROLES)
    require(sum(role_totals.values()) == inputs["total"]["flows"],
            "Global split count conservation failed")
    require(retained + role_totals["embargo"] == inputs["total"]["flows"],
            "Retained + embargo conservation failed")
    for members in inputs["groups"].values():
        for item in members:
            sid = int(item["source_id"])
            require(next_source_rank[sid] == int(item["ordered_rows"]),
                    f"Source sorted-rank coverage failed: {sid}")
    require(all(sum(source_supplies.get((sid, role), 0)
                    for sid, category in inputs["source_categories"].items()
                    if category == cat) == supplies[(cat, role)]
                for cat in CLASS_BUDGET for role in ROLES),
            "Per-source and per-category role supplies disagree")
    checks["source_flow_conservation"] = "PASS"
    checks["applicable_511_boundary_guards"] = "FAIL" if failed_groups else "PASS"
    checks["activity_end_crossings_under_assumed_us"] = (
        "FAIL" if activity_crossings else "PASS_ASSUMED_DURATION")
    checks["benign_attack_each_retained_role"] = (
        "PASS" if all(label_roles[role]["BENIGN"] > 0 and label_roles[role]["ATTACK"] > 0
                      for role in ROLES) else "FAIL")
    ratios = {role: (100 * role_totals[role] / retained if retained else 0)
              for role in ROLES}
    integrity = {
        "stage": "2.8",
        "status": ("DRY_RUN_PLAN_WITH_FAILED_GATES" if failed_groups or activity_crossings
                   or checks["benign_attack_each_retained_role"] == "FAIL"
                   else "DRY_RUN_PLAN_ONLY"),
        "assumption_profile": "CICFLOW_ADAPTED_QUANTILE_BASELINE_ASSUMED_MICROSECONDS",
        "source_flows": inputs["total"]["flows"],
        "valid_metadata_flows": inputs["total"]["flows"] - EXPECTED_INVALID_DURATION,
        "whole_bucket_eligible_flows": sum(role_totals.values()) - embargo["metadata_invalid_whole_bucket"],
        "train_flows": role_totals["train"], "validation_flows": role_totals["val"],
        "test_flows": role_totals["test"], "embargo_flows": role_totals["embargo"],
        "retained_flows": retained, "actual_retained_ratio_percent": ratios,
        "binary_proxy_by_split": {
            role: dict(label_roles[role]) for role in (*ROLES, "embargo")},
        "activity_crossing_violating_flows": activity_crossings,
        "reference_connection_key_rule": "C+ trims Src/Dst IP, try_cast ports/protocol, checks ranges, and builds a lexicographic bidirectional key",
        "stage25_invalid_key_input_rows": 0,
        "connection_key_values_indexed": False,
        "reference_duplicate_rule": "C+ hashes all original string fields, then confirms candidate duplicates by full-field equality",
        "exact_duplicate_membership_indexed": False,
        "failed_or_infeasible_groups": failed_groups,
        "checks": checks,
        "cutoff_plan_computed": not failed_groups and activity_crossings == 0
        and checks["benign_attack_each_retained_role"] == "PASS",
        "partial_cutoff_results_available": True,
        "feature_csvs_exported": False, "subset_rows_selected": False,
        "materialization_ready": False,
        "materialization_blockers": [
            "Exact C+ full-record duplicate crossing check NOT_VERIFIED",
            "Exact C+ connection-key value/session leakage audit NOT_VERIFIED",
            "Research-assumed duration and cross-source naive clock, not independently verified",
        ] + (["One or more split integrity checks failed"] if failed_groups or activity_crossings
             or checks["benign_attack_each_retained_role"] == "FAIL" else []),
    }
    embargo_review = {
        "total": role_totals["embargo"],
        "raw_invalid_duration_rows": EXPECTED_INVALID_DURATION,
        "whole_bucket_invalid_embargo_flows": embargo["metadata_invalid_whole_bucket"],
        "temporal_gap_embargo_flows": embargo["temporal_gap"],
        "note": "A mixed valid/invalid (gid,timestamp) bucket is embargoed whole, not repaired or split.",
    }
    return scenario_rows, cutoff_rows, embargo_review, integrity, dict(source_supplies)


def render_report(inputs: dict, optimization: dict, integrity: dict,
                  embargo: dict, subset: dict, cutoff_rows: list[dict],
                  preflight_result: dict) -> str:
    ratios = integrity["actual_retained_ratio_percent"]
    checks = integrity["checks"]
    lines = [
        "# STAGE 2.8 — CICFLOW C+ adapted split dry-run", "",
        "## Decision", "",
        "**CICFLOW-adapted temporal C+ plan**, not a verified reproduction of the original CICIoMT24 cutoffs.",
        "No Train/Validation/Test CSVs, subset rows, sliding windows or models were created.",
        f"Cutoff plan computed: {integrity['cutoff_plan_computed']}. "
        f"Materialization ready: {integrity['materialization_ready']}.", "",
        "## Direct numerical results", "",
        "| Metric | Flows |",
        "|---|---:|",
        f"| All selected source rows | {integrity['source_flows']:,} |",
        f"| Raw metadata-valid rows | {integrity['valid_metadata_flows']:,} |",
        f"| Train | {integrity['train_flows']:,} |",
        f"| Validation | {integrity['validation_flows']:,} |",
        f"| Test | {integrity['test_flows']:,} |",
        f"| Embargo | {integrity['embargo_flows']:,} |",
        f"| Retained | {integrity['retained_flows']:,} |", "",
        f"Actual retained ratio: Train {ratios['train']:.4f}% / "
        f"Validation {ratios['val']:.4f}% / Test {ratios['test']:.4f}% "
        "(target 60/20/20; never forced through an unsafe boundary).", "",
        f"Five invalid-duration source rows are unchanged. Whole-bucket invalid embargo: "
        f"{embargo['whole_bucket_invalid_embargo_flows']:,} flows; temporal-gap embargo: "
        f"{embargo['temporal_gap_embargo_flows']:,} flows.", "",
        "## Binary IDS coverage", "",
        "| Role | BENIGN | ATTACK |",
        "|---|---:|---:|",
    ]
    for role in ROLES:
        values = integrity["binary_proxy_by_split"][role]
        lines.append(f"| {role} | {values.get('BENIGN', 0):,} | {values.get('ATTACK', 0):,} |")
    lines += [
        "", f"Both proxy classes in every retained role: "
        f"{checks['benign_attack_each_retained_role']}.",
        "Raw Label remains No Label; folder-derived binary labels are provisional.", "",
        "## Representative subset feasibility", "",
        f"Stage 2.1 budget: {subset['target_subset_flows']:,} flows. "
        f"Exact category budgets plus global 60/20/20 role capacities feasible: "
        f"{subset['exact_category_budgets_and_ratio_feasible']}.",
        f"Maximum feasible capacity-only subset with category caps and 60/20/20: "
        f"{subset['maximum_total_with_category_caps_and_60_20_20']:,} flows "
        f"(gap {subset['gap_from_target']:,}).",
        "This is a capacity plan enforcing all 133 Stage 2.1 per-CSV budgets, "
        "including 16 PARTIAL files; no source row was sampled or reused from embargo.", "",
        "| Category | Stage 2.1 budget | Available after embargo | Shortfall |",
        "|---|---:|---:|---:|",
    ]
    for category, item in subset["category_supply_and_plan"].items():
        lines.append(f"| {category} | {item['stage21_budget']:,} | "
                     f"{item['available_total']:,} | {item['budget_shortfall']:,} |")
    if not subset["exact_category_budgets_and_ratio_feasible"]:
        lines += [
            "",
            "The exact Stage 2.1 budget is infeasible from retained role capacities. "
            "Per-category shortfalls are shown above; if those are zero, the blocker "
            "is the distribution across Train/Validation/Test, not the category total.",
        ]
    lines += [
        "",
        "## C+ mechanics and disclosed adaptations", "",
        "- Reference CICIoMT24 C+: frozen scenario-specific B2 cutoffs, safe C baseline, then bounded +/-300-second coordinate optimization; whole (gid,timestamp) buckets, >=511 applicable guard flows and activity/duplicate audits.",
        "- CICFLOW has no transferable frozen B2/C timestamps. Researcher explicitly approved a new 60/20/20 quantile anchor. This script shortens the earlier boundary for safety, then applies the reference C+ local objective/search around that safe baseline.",
        "- 33 Stage 2.6 directory scenario candidates are used as experimental gids. They are neither verified PCAP identities nor borrowed G01..G19.",
        "- Mirai-udpplain15 (2) remains in its Stage 2.6 directory scenario candidate by explicit research assumption; its original PCAP relation remains UNKNOWN.",
        "- Sources sharing a gid and timestamp enter the same bucket; tie order is parsed timestamp, stable Stage 2.5 source ID and original row index.",
        "- Clock comparability is assumed naive; timezone and capture provenance are not verified. No capture-disjoint claim.",
        "- Flow Duration is assumed microseconds (Stage 2.5 data corroboration only); end_upper = timestamp + duration_us + 1 second.",
        "- A global bucket containing one of the five metadata-invalid rows is embargoed whole. This is an explicit CICFLOW policy adjustment needed to preserve bucket integrity.",
        "- Original six CICIoMT24 Train-only scenario names are not transferred. Attack scenarios that cannot satisfy all three >=512-role and two >=511-boundary guards are Train-only by feasibility; Benign Train-only is an integrity failure.",
        f"- Baseline retained rows: {optimization['baseline_retained']:,}; "
        f"C+ local candidate pairs examined: {optimization['candidates_examined']:,}; "
        f"accepted moves: {len(optimization['accepted_moves'])}.", "",
        "## Leakage/integrity checks", "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, status in checks.items():
        lines.append(f"| {name} | {status} |")
    lines += [
        "", f"Activity crossing violating flows under the assumed duration unit: "
        f"{integrity['activity_crossing_violating_flows']:,}.",
        "Exact connection-key value/session audit and full-record duplicate crossing audit are NOT_VERIFIED; source-local Stage 2.5 key-input checks are not full C+ equivalence.",
        "Temporal embargo cannot eliminate shared capture, host, near-duplicate or feature/label leakage.",
        "", "## Materialization gate", "",
    ]
    if integrity["materialization_blockers"]:
        lines.extend(f"- {reason}" for reason in integrity["materialization_blockers"])
    else:
        lines.append("- No known blocker (this stage still does not export data).")
    lines += [
        "", "Before materialization: implement exact duplicate/connection-key cross-split audits, review any failed scenario, and verify that raw CSVs still match Stage 2.4/2.5 SHA256 during a future streaming export.",
        "No Stage 2.9 was executed.", "",
        "## Resource and resume notes", "",
        f"Stage 2.8 replayed only Stage 2.5 indexes; original flow CSV bytes read: 0. "
        f"Disk estimate allowance: {preflight_result['recommended_output_free_bytes']:,} bytes "
        "(planning estimate, not a cap).",
        "Each scenario bucket file is checkpointed with an input binding and SHA256. Re-running the same full command verifies and skips completed scenarios, then recomputes final reports.",
        "A change to any bound Stage 2.5/2.6 metadata requires a fresh output directory; do not mix checkpoints.", "",
        "## Per-scenario boundary status", "",
        "| gid | Category | Mode | Train | Val | Test | Embargo | Train/Val guard | Val/Test guard |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cutoff_rows:
        lines.append(
            f"| {row['gid']} | {row['category']} | {row['mode']} | "
            f"{row['train_flows']:,} | {row['val_flows']:,} | "
            f"{row['test_flows']:,} | {row['embargo_flows']:,} | "
            f"{row['train_val_embargo_flows']:,} | {row['val_test_embargo_flows']:,} |")
    lines += ["", "Exact cutoff timestamps are in cplus_cutoffs.csv. No cutoff is claimed to be the original CICIoMT24 frozen cutoff.", ""]
    return "\n".join(lines)


def run(repo: Path, output: Path, execute: bool, accept_disk: bool) -> None:
    require(repo.is_dir(), f"Repository not found: {repo}")
    dataset = (repo / "CICFLOWMETER_FEATURE_CSV").resolve()
    require(output != repo and output != dataset and dataset not in output.parents,
            "Output cannot be repository root or inside raw dataset")
    for name in ("STAGE25_ORDERED_INDEX", "STAGE26_GROUP_MAPPING",
                 "STAGE21_SUBSET_PLAN", "code_CICFLOW", "huyen_code"):
        source_area = (repo / name).resolve()
        require(output != source_area and source_area not in output.parents,
                f"Output cannot be inside input/code area: {source_area}")
    inputs = load_inputs(repo)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    completed_buckets = 0
    if checkpoint_dir.is_dir():
        for gid in inputs["groups"]:
            p = checkpoint_dir / f"{gid}.checkpoint.json"
            g = checkpoint_dir / f"{gid}.buckets.csv"
            if p.is_file() and g.is_file():
                saved = read_json(p)
                if saved.get("input_binding") == group_checkpoint_binding(inputs, gid):
                    completed_buckets += int(saved.get("stats", {}).get("buckets", 0))
    disk = preflight(inputs, output)
    remaining = max(0, inputs["total"]["source_buckets"] - completed_buckets)
    disk["checkpointed_source_buckets"] = completed_buckets
    disk["recommended_output_free_bytes"] = (
        (remaining + inputs["total"]["source_buckets"]) * 220 + 1024 ** 3)
    disk["space_estimate_pass"] = disk["actual_free_bytes"] >= disk["recommended_output_free_bytes"]
    write_json_atomic(output / "stage28_preflight.json", disk)
    if not execute:
        print(json.dumps(disk, ensure_ascii=False, indent=2))
        print("Estimate only; no index replay or cutoff calculation. Use --execute --accept-disk-estimate.")
        return
    require(accept_disk, "Full dry-run requires --execute --accept-disk-estimate")
    require(disk["space_estimate_pass"],
            "Insufficient free disk for conservative checkpoint/output estimate; no index replay started")
    config = {
        "version": VERSION, "input_sha256": inputs["input_sha256"],
        "stage28_code_sha256": sha256_file(Path(__file__).resolve()),
        "assumptions": {
            "gid": "STAGE26_DIRECTORY_CANDIDATE_33",
            "clock": "COMMON_NAIVE_ASSUMED",
            "duration": "MICROSECONDS_ASSUMED",
            "baseline": "CICFLOW_QUANTILE_60_20_20_ADAPTED",
            "invalid_bucket": "WHOLE_BUCKET_EMBARGO",
        },
    }
    config_path = output / ".stage28_run_config.json"
    if config_path.is_file():
        saved_config = read_json(config_path)
        if saved_config != config:
            compatible = dict(saved_config)
            previous_code_sha = compatible.pop("stage28_code_sha256", None)
            expected_without_code_sha = dict(config)
            expected_without_code_sha.pop("stage28_code_sha256")
            require(previous_code_sha == ZERO_COUNT_FIX_PREVIOUS_SHA256
                    and compatible == expected_without_code_sha,
                    "Run configuration changed; use a fresh output directory")
            print("[resume] Accepting exact prior Stage 2.8 code fingerprint for "
                  "zero-count checkpoint fix; all input bindings remain unchanged.",
                  flush=True)
            write_json_atomic(config_path, config)
    else:
        require(not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()),
                "Checkpoint files exist without run configuration; manual review required")
        write_json_atomic(config_path, config)
    verify_indexes(inputs)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for i, gid in enumerate(sorted(inputs["groups"]), 1):
        checkpoints[gid] = build_one_group(inputs, gid, checkpoint_dir)
        print(f"[{i}/{EXPECTED_GROUPS}] {gid}: "
              f"{checkpoints[gid]['stats']['flows']:,} flows, "
              f"{checkpoints[gid]['stats']['buckets']:,} global buckets",
              flush=True)
    require(sum(c["stats"]["flows"] for c in checkpoints.values()) == EXPECTED_FLOWS,
            "Global buckets do not conserve all selected flows")
    require(sum(c["stats"]["raw_invalid"] for c in checkpoints.values())
            == EXPECTED_INVALID_DURATION, "Global invalid-duration count changed")
    require(sum(c["stats"].get("shared_timestamps", 0)
                for c in checkpoints.values())
            == EXPECTED_SHARED_TIMESTAMPS, "Global shared-timestamp count changed")
    plans, optimization = calculate_plans(inputs, checkpoint_dir)
    scenario_rows, cutoff_rows, embargo, integrity, source_supplies = evaluate_plan(
        inputs, checkpoint_dir, plans)
    subset = subset_review(inputs, source_supplies)
    global_counts = assemble_global(checkpoint_dir, output,
                                    sorted(inputs["groups"]), plans)
    require(all(global_counts[role] == integrity[
                    {"train": "train_flows", "val": "validation_flows",
                     "test": "test_flows", "embargo": "embargo_flows"}[role]]
                for role in (*ROLES, "embargo")),
            "Final global bucket role summary differs from scenario audit")
    integrity["optimization"] = optimization
    integrity["input_sha256"] = inputs["input_sha256"]
    integrity["subset_target_feasible_capacity_only"] = subset[
        "exact_category_budgets_and_ratio_feasible"]
    integrity["source_local_buckets"] = inputs["total"]["source_buckets"]
    integrity["global_buckets"] = sum(c["stats"]["buckets"]
                                      for c in checkpoints.values())
    integrity["shared_timestamp_global_buckets"] = EXPECTED_SHARED_TIMESTAMPS
    cutoff_fields = (
        "gid", "category", "scenario", "mode", "train_end", "val_start",
        "val_end", "test_start", "train_val_embargo_flows",
        "val_test_embargo_flows", "train_flows", "val_flows", "test_flows",
        "embargo_flows", "activity_crossing_flows", "reason",
    )
    scenario_fields = ("gid", "category", "scenario", "split", "flows",
                       "scenario_total", "plan_mode")
    write_csv_atomic(output / "cplus_cutoffs.csv", cutoff_fields, cutoff_rows)
    write_csv_atomic(output / "split_counts_by_scenario.csv",
                     scenario_fields, scenario_rows)
    write_json_atomic(output / "embargo_summary.json", embargo)
    write_json_atomic(output / "subset_feasibility.json", subset)
    write_json_atomic(output / "cplus_dryrun_integrity.json", integrity)
    report = render_report(inputs, optimization, integrity, embargo,
                           subset, cutoff_rows, disk)
    temp_report = output / "STAGE28_REPORT.md.stage28_tmp"
    temp_report.write_text(report, encoding="utf-8")
    os.replace(temp_report, output / "STAGE28_REPORT.md")
    print(f"Stage 2.8 dry-run complete: {output / 'STAGE28_REPORT.md'}")
    print("Materialization not run; inspect integrity checks before any next stage.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "STAGE28_CPLUS_PLAN")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-disk-estimate", action="store_true")
    args = parser.parse_args()
    try:
        run(args.repository.resolve(), args.output_root.resolve(),
            args.execute, args.accept_disk_estimate)
    except (DryRunError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"STAGE28 ERROR: {error}\n")


if __name__ == "__main__":
    main()
