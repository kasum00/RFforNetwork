"""Only local writes at per-k boundaries; no Drive dependency."""
import json
import math
import time
from pathlib import Path
from .common import Blocked, atomic, digest, read, redact, utc


def validate_rows(rows, ranked):
    if len(rows) > 100 or [r.get("k") for r in rows] != list(range(1, len(rows)+1)):
        raise Blocked("Completed k must be a contiguous prefix of 1..100")
    for row in rows:
        accuracy = row.get("accuracy")
        if (row.get("features") != ranked[:row["k"]] or isinstance(accuracy, bool)
                or not isinstance(accuracy, (float, int)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
            raise Blocked("Invalid k metrics/ranking prefix")
    return rows


def choose(rows, ranked, *, complete=False):
    validate_rows(rows, ranked)
    if not rows or (complete and len(rows) != 100):
        raise Blocked("Selection requires all k=1..100")
    best = rows[0]
    for row in rows[1:]:
        if row["accuracy"] > best["accuracy"]:
            best = row
    return best


class LocalProgress:
    def __init__(self, state, identity, *, check=None):
        self.state, self.identity = Path(state), identity
        self.state.mkdir(parents=True, exist_ok=True)
        self.guard = check or (lambda: None)

    def check(self):
        self.guard()

    def event(self, event, **details):
        row = {"utc": utc(), "event": event, "identity": self.identity["sha256"], **details}
        with (self.state / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(redact(json.dumps(row, allow_nan=False)) + "\n")
            f.flush()
        atomic(self.state / "progress.json", row)
        print(redact(json.dumps(row, allow_nan=False)), flush=True)


class KProgress:
    def __init__(self, state, identity, ranked, rfe_sha):
        self.root = Path(state) / "k"
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = {"identity": identity["sha256"], "rfe": rfe_sha, "ranked": digest(ranked)}
        self.ranked, self.rows, self.previous = ranked, [], None
        files = sorted(self.root.glob("k_*.json"))
        for k, path in enumerate(files, 1):
            record = read(path)
            raw = {key: value for key, value in record.items() if key != "sha256"}
            if (path.name != f"k_{k:03d}.json" or record.get("binding") != self.binding
                    or record.get("previous") != self.previous or digest(raw) != record.get("sha256")):
                raise Blocked("K checkpoint identity/checksum/gap mismatch")
            self.rows.append(record["row"])
            self.previous = record["sha256"]
        validate_rows(self.rows, ranked)

    def append(self, row):
        validate_rows(self.rows + [row], self.ranked)
        record = {"binding": self.binding, "previous": self.previous, "row": row}
        record["sha256"] = digest(record)
        atomic(self.root / f"k_{row['k']:03d}.json", record, exclusive=True)
        self.rows.append(row)
        self.previous = record["sha256"]
