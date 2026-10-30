"""Policy excerpt transcribed from run_phuong_anB.py pasted by the user.

Only pure search helpers/constants are retained here for regression comparison.
This is not the complete original runner and contains no data/training entrypoint.
"""
COARSE = (10, 20, 30, 50, 75, 100)
RADIUS = 5


def fine_grid(best_k: int, tried) -> list[int]:
    return [k for k in range(max(1, best_k - RADIUS), min(100, best_k + RADIUS) + 1) if k not in set(tried)]


def choose_best(rows: list[dict]) -> dict:
    best = None
    for row in sorted(rows, key=lambda r: r["k"]):
        if best is None or row["metrics"]["accuracy"] > best["metrics"]["accuracy"]:
            best = row
    return best
