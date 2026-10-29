"""RFE step5/target1 with an audited total order inside sklearn rank groups."""
import math
import time
import numpy as np
from .contract import Blocked, TIE_POLICY, digest


def validate_steps(steps, count):
    support, ranking = np.ones(count, bool), np.ones(count, int)
    for number, row in enumerate(steps, 1):
        required = {"step", "active_indices", "raw_importances", "squared_importances", "removed_indices", "support", "ranking"}
        if not isinstance(row, dict) or not required <= set(row):
            raise Blocked("Incomplete RFE checkpoint fields")
        active = np.flatnonzero(support)
        raw = np.asarray(row["raw_importances"], dtype=np.float64)
        squared = raw**2
        if (row["step"] != number or row["active_indices"] != active.tolist()
                or raw.shape != (len(active),) or not np.isfinite(raw).all() or (raw < 0).any()
                or row["squared_importances"] != squared.tolist() or len(active) <= 1):
            raise Blocked("RFE checkpoint dimensions/importances differ")
        removed = active[np.argsort(squared)[:min(5, len(active)-1)]]
        support[removed] = False
        ranking[~support] += 1
        if (row["removed_indices"] != removed.tolist() or row["support"] != support.tolist()
                or row["ranking"] != ranking.tolist()):
            raise Blocked("RFE step5 update differs from sklearn semantics")
    return support, ranking


def total_order(names, ranking, steps):
    importance, round_removed = {}, {}
    for row in steps:
        scores = dict(zip(row["active_indices"], row["squared_importances"]))
        for index in row["removed_indices"]:
            importance[index], round_removed[index] = scores[index], row["step"]
    if sorted(importance) != sorted(i for i, rank in enumerate(ranking) if rank != 1):
        raise Blocked("Missing elimination-round importance for tied ranks")
    order = sorted(range(len(names)), key=lambda i: (int(ranking[i]), -importance.get(i, 0.0), i))
    records = [{"feature": names[i], "mi_position": i, "sklearn_rfe_rank": int(ranking[i]),
                "total_order_position": position+1, "removal_round": round_removed.get(i),
                "removal_squared_importance": importance.get(i)} for position, i in enumerate(order)]
    return records


def fit_step5(X, y, names, estimator, *, fit_role="train", steps=(), checkpoint=lambda row: None,
              check=lambda: None, event=lambda *args, **kwargs: None):
    from sklearn.base import clone
    if fit_role != "train" or X.ndim != 2 or X.shape[1] != len(names) or len(X) != len(y):
        raise Blocked("RF-RFE must fit aligned Train data only")
    if len(set(names)) != len(names) or len(names) < 2:
        raise Blocked("Invalid ordered MI features")
    steps = list(steps)
    support, ranking = validate_steps(steps, len(names))
    while support.sum() > 1:
        check()
        active = np.flatnonzero(support)
        model = clone(estimator)
        started = time.perf_counter()
        event("rfe_fit_started", remaining_features=len(active), elimination_step=len(steps)+1)
        model.fit(X[:, active], y)
        fit_seconds = time.perf_counter()-started
        event("rfe_fit_completed", remaining_features=len(active), fit_seconds=fit_seconds)
        raw = np.asarray(model.feature_importances_, dtype=np.float64)
        if raw.shape != (len(active),) or not np.isfinite(raw).all() or (raw < 0).any():
            raise Blocked("Invalid RF feature importances")
        squared = raw**2
        removed = active[np.argsort(squared)[:min(5, len(active)-1)]]
        support[removed] = False
        ranking[~support] += 1
        row = {"step": len(steps)+1, "active_indices": active.tolist(),
               "raw_importances": raw.tolist(), "squared_importances": squared.tolist(),
               "removed_indices": removed.tolist(), "support": support.tolist(), "ranking": ranking.tolist(),
               "fit_seconds": fit_seconds}
        del model
        checkpoint(row)
        steps.append(row)
        event("rfe_step_completed", remaining_features=int(support.sum()), elimination_step=len(steps))
    check()
    # sklearn RFE also fits an estimator on the final surviving feature.
    event("rfe_final_estimator_fit_started", remaining_features=1)
    model = clone(estimator)
    final_started = time.perf_counter()
    model.fit(X[:, np.flatnonzero(support)], y)
    final_seconds = time.perf_counter()-final_started
    event("rfe_fit_completed", remaining_features=1, fit_seconds=final_seconds)
    del model
    records = total_order(names, ranking, steps)
    return {"names": [r["feature"] for r in records], "ranking": ranking.tolist(),
            "records": records, "steps": steps, "tie_policy": TIE_POLICY,
            "step": 5, "target": 1, "fit_role": "train", "final_estimator_fit_performed": True,
            "forest_fits_in_completed_rfe": len(steps)+1,
            "final_estimator_fit_seconds": final_seconds,
            "unique_sklearn_rank_count": len(set(ranking.tolist())),
            "rank_is_not_a_unique_1_to_100_permutation": True}


def validate_result(result, names):
    if (result.get("step") != 5 or result.get("target") != 1 or result.get("fit_role") != "train"
            or result.get("tie_policy") != TIE_POLICY or result.get("final_estimator_fit_performed") is not True):
        raise Blocked("Not an A1 step5 target1 RFE artifact")
    support, ranking = validate_steps(result["steps"], len(names))
    records = total_order(names, ranking, result["steps"])
    if (support.sum() != 1 or result["ranking"] != ranking.tolist() or result["records"] != records
            or result["names"] != [r["feature"] for r in records]
            or result["forest_fits_in_completed_rfe"] != len(result["steps"])+1
            or result.get("unique_sklearn_rank_count") != len(set(ranking.tolist()))):
        raise Blocked("Incomplete/non-deterministic A1 ranking")
    return result["names"]
