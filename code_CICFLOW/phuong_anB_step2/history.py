"""Sparse policy-ordered candidates, independent of A1's contiguous k history."""
import math
from pathlib import Path
from .contract import Blocked, atomic, digest, info, read
from .policy import expected_sequence, fine_candidates, validate_policy, winner

METRICS = ("accuracy", "precision_attack", "recall_attack", "recall_benign", "f1_attack", "macro_f1", "balanced_accuracy")


def validate_rows(rows, ranked, policy, *, validation_count=None, complete=False):
    if len(ranked) != policy["feature_count"] or len(set(ranked)) != len(ranked):
        raise Blocked("Candidate ranking/policy feature count differs")
    required = {"phase", "k", "features", "fit", "predict_seconds", "unit_seconds",
                "classification_report", "label_origin", "started", "finished",
                "confusion_matrix_labels_0_benign_1_attack", *METRICS}
    if any(not isinstance(row, dict) or not required <= set(row)
           or not isinstance(row["fit"], dict)
           or not {"seconds", "converged", "warnings"} <= set(row["fit"]) for row in rows):
        raise Blocked("Incomplete candidate scientific fields")
    sequence = expected_sequence(rows, policy)
    if len(rows) > len(sequence) or (complete and len(rows) != len(sequence)):
        raise Blocked("Selection does not cover exactly the approved coarse/fine candidates")
    for ordinal, row in enumerate(rows):
        phase, k = sequence[ordinal]
        if row["phase"] != phase or type(row["k"]) is not int or row["k"] != k or row["features"] != ranked[:k]:
            raise Blocked("Candidate order/phase/k/feature prefix differs from policy")
        for key in METRICS:
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise Blocked("Missing/invalid measured Validation metric: " + key)
        for value in (row["fit"]["seconds"], row["predict_seconds"], row["unit_seconds"]):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise Blocked("Invalid measured candidate duration")
        if row["unit_seconds"] + 1e-8 < row["fit"]["seconds"] + row["predict_seconds"]:
            raise Blocked("Candidate wall time is shorter than recorded fit/predict")
        if type(row["fit"]["converged"]) is not bool or not isinstance(row["fit"]["warnings"], list):
            raise Blocked("Candidate convergence/warning diagnostics missing")
        matrix = row["confusion_matrix_labels_0_benign_1_attack"]
        if (not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(r, list) or len(r) != 2 for r in matrix)
                or any(type(v) is not int or v < 0 for r in matrix for v in r)):
            raise Blocked("Invalid binary Validation confusion matrix")
        count = sum(map(sum, matrix))
        if (count <= 0 or validation_count is not None and count != validation_count
                or not math.isclose((matrix[0][0]+matrix[1][1])/count, row["accuracy"], abs_tol=1e-12)):
            raise Blocked("Validation matrix/count/Accuracy mismatch")
    if rows:
        winner(rows)  # Enforce global uniqueness and the measured-score contract.
    if complete and len(rows) < len(policy["coarse_grid"]):
        raise Blocked("Coarse phase is incomplete")
    return rows


class History:
    def __init__(self, state, identity, ranked, rfe_sha):
        self.root = Path(state) / "candidates"
        self.identity, self.ranked = identity, ranked
        self.policy = identity["search_policy"]
        validate_policy(self.policy)
        self.binding = {"identity": identity["sha256"], "rfe": rfe_sha,
                        "ranked": digest(ranked), "policy": digest(self.policy), "rfe_step": 2}
        self.rows, self.records, self.previous = [], [], None
        # Reading a missing directory does not create candidate state during verify.
        for sequence, path in enumerate(sorted(self.root.glob("candidate_*.json")), 1):
            record = read(path)
            if not isinstance(record, dict) or not {"binding", "previous", "sequence", "sha256", "row"} <= set(record):
                raise Blocked("Incomplete candidate checkpoint record")
            if not isinstance(record["row"], dict) or type(record["row"].get("k")) is not int:
                raise Blocked("Incomplete candidate k")
            if (record["binding"] != self.binding or record["previous"] != self.previous
                    or record["sequence"] != sequence
                    or record["sha256"] != digest({k: v for k, v in record.items() if k != "sha256"})
                    or path.name != self.filename(sequence, record["row"]["k"])):
                raise Blocked("Candidate checkpoint identity/hash/sequence mismatch")
            self.rows.append(record["row"])
            self.records.append(record)
            self.previous = record["sha256"]
        validate_rows(self.rows, ranked, self.policy, validation_count=identity["dataset"]["splits"]["val"]["rows"])

    @staticmethod
    def filename(sequence, k):
        return f"candidate_{sequence:03d}_k{k:03d}.json"

    def path(self, sequence):
        return self.root / self.filename(sequence, self.rows[sequence-1]["k"])

    def stage(self, sequence):
        return self.path(sequence).stem

    def append(self, row):
        validate_rows(self.rows+[row], self.ranked, self.policy,
                      validation_count=self.identity["dataset"]["splits"]["val"]["rows"])
        sequence = len(self.rows)+1
        record = {"binding": self.binding, "sequence": sequence, "previous": self.previous, "row": row}
        record["sha256"] = digest(record)
        atomic(self.root / self.filename(sequence, row["k"]), record, exclusive=True)
        self.rows.append(row); self.records.append(record); self.previous = record["sha256"]

    def fine_plan(self):
        count = len(self.policy["coarse_grid"])
        if len(self.rows) < count:
            raise Blocked("Fine plan requires complete measured coarse results")
        coarse = self.rows[:count]
        best = winner(coarse)
        value = {"binding": self.binding, "coarse_results_sha256": digest(coarse),
                 "coarse_best_k": best["k"], "coarse_best_validation_accuracy": best["accuracy"],
                 "fine_radius": self.policy["fine_radius"],
                 "coarse_anchor_k": [best["k"]], "fine_grid": fine_candidates(coarse, self.policy)}
        value["sha256"] = digest(value)
        return value


def check_committed_candidates(store, history):
    durable = [r for r in store.rows if r["record"]["stage"].startswith("candidate_")]
    if len(durable) > len(history.rows):
        raise Blocked("Local candidate history is behind committed state")
    for sequence, checkpoint in enumerate(durable, 1):
        path = history.path(sequence)
        relative = path.relative_to(store.state).as_posix()
        if (checkpoint["record"]["stage"] != history.stage(sequence)
                or checkpoint["record"]["files"].get(relative) != info(path)
                or checkpoint["record"]["metadata"]["phase"] != history.rows[sequence-1]["phase"]
                or checkpoint["record"]["metadata"]["k"] != history.rows[sequence-1]["k"]):
            raise Blocked("Committed candidate prefix differs from verified local records")
    return len(durable)


def check_committed_search(store, history):
    """Completed search must bind the frozen plan between coarse and fine commits."""
    count = check_committed_candidates(store, history)
    validate_rows(history.rows, history.ranked, history.policy,
                  validation_count=history.identity["dataset"]["splits"]["val"]["rows"], complete=True)
    if count != len(history.rows):
        raise Blocked("Every completed candidate must be COMMITTED before publication")
    rows = store.rows
    plan = next((row for row in rows if row["record"]["stage"] == "fine_plan"), None)
    coarse_count = len(history.policy["coarse_grid"])
    coarse_last = next(row for row in rows if row["record"]["stage"] == history.stage(coarse_count))
    first_fine = next((row for row in rows if row["record"]["stage"] == history.stage(coarse_count+1)), None)
    if (plan is None or plan["record"]["files"].get("fine_plan.json") != info(store.state / "fine_plan.json")
            or plan["record"]["sequence"] <= coarse_last["record"]["sequence"]
            or first_fine is None or first_fine["record"]["sequence"] <= plan["record"]["sequence"]):
        raise Blocked("Fine plan was not COMMITTED after all coarse measurements and before fine candidates")
    return count
