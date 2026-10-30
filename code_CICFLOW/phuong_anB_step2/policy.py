"""Exact coarse/fine policy from the user-supplied run_phuong_anB.py.

RFE_STEP=2 controls RF elimination only. It never defines the coarse SVM grid.
"""
import math
from .contract import Blocked

POLICY_NAME = "phuong_anB_coarse_to_fine_v1"
COARSE = (10, 20, 30, 50, 75, 100)
RADIUS = 5


def coarse_grid():
    return list(COARSE)


def policy_spec():
    return {"version": "B.step2.coarse6.radius5.v2", "name": POLICY_NAME,
            "reference": "user-supplied run_phuong_anB.py",
            "feature_count": 100, "coarse_grid": coarse_grid(), "fine_radius": RADIUS,
            "fine_rule": "inclusive [max(1,coarse_best_k-5),min(100,coarse_best_k+5)] excluding tried k",
            "coarse_anchor_tie": "highest Validation Accuracy; ties smallest k",
            "fine_order": "ascending k", "selection": "highest Validation Accuracy; ties smallest k",
            "max_unique_svm_fits": 16,
            "exhaustive_1_to_100_required": False, "test_used": False}


def validate_policy(value):
    if value != policy_spec():
        raise Blocked("B requires the original six coarse k and radius5 policy; draft policies are incompatible")
    return value


def approved_policy(request):
    if request.get("rfe_step") != 2 or type(request.get("rfe_step")) is not int:
        raise Blocked("B is bound to integer RFE_STEP=2")
    supplied_grid = request.get("coarse")
    if (request.get("search_policy") != POLICY_NAME or not isinstance(supplied_grid, list)
            or any(type(k) is not int for k in supplied_grid) or supplied_grid != list(COARSE)
            or type(request.get("radius")) is not int or request["radius"] != RADIUS):
        raise Blocked("Use original COARSE=(10,20,30,50,75,100), RADIUS=5; RFE step2 is separate")
    return policy_spec()  # Already explicitly specified by the user; no new policy approval prompt.


def winner(rows):
    if not rows:
        raise Blocked("No measured Validation candidates")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"k", "accuracy"} <= set(row):
            raise Blocked("Candidate lacks k/Validation Accuracy")
        k, accuracy = row["k"], row["accuracy"]
        if (type(k) is not int or not 1 <= k <= 100 or k in seen
                or type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
            raise Blocked("Invalid/duplicate measured candidate")
        seen.add(k)
    return min(rows, key=lambda row: (-row["accuracy"], row["k"]))


def fine_grid(best_k, tried):
    """Same inclusive radius and ascending visit order as original fine_grid()."""
    if type(best_k) is not int or not 1 <= best_k <= 100:
        raise Blocked("Invalid coarse_best_k")
    tried = set(tried)
    if any(type(k) is not int or not 1 <= k <= 100 for k in tried):
        raise Blocked("Invalid tried k")
    return [k for k in range(max(1, best_k-RADIUS), min(100, best_k+RADIUS)+1)
            if k not in tried]


def fine_candidates(coarse_rows, policy):
    validate_policy(policy)
    grid = policy["coarse_grid"]
    winner(coarse_rows)  # Validate numeric k/scores before comparing the ordered coarse grid.
    if [r["k"] for r in coarse_rows] != grid:
        raise Blocked("Fine search requires the complete policy-defined coarse grid")
    best = winner(coarse_rows)
    return fine_grid(best["k"], grid)


def expected_sequence(rows, policy):
    validate_policy(policy)
    grid = policy["coarse_grid"]
    if len(rows) < len(grid):
        return [("COARSE", k) for k in grid]
    return [("COARSE", k) for k in grid] + [("FINE", k) for k in fine_candidates(rows[:len(grid)], policy)]
