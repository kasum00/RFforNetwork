"""Policy-defined sparse SVM evaluations; no Test and no interrupted-fit resume."""
import gc
import time
from .contract import Blocked, atomic, read, utc
from .history import validate_rows
from .policy import winner
from mi_rfe_svm_colab.dataset import svc_prefix


def ensure_fine_plan(state, history, checkpoint):
    expected = history.fine_plan()
    path = state / "fine_plan.json"
    if path.exists():
        if read(path) != expected:
            raise Blocked("Frozen fine grid/coarse scores differ; no overwrite")
    else:
        atomic(path, expected, exclusive=True)
    checkpoint("fine_plan", {"fine_plan.json": path}, phase="FINE_PLAN")
    return expected["fine_grid"]


def search(state, history, X, y, V, vy, progress, *, model_factory, fit_fn,
           metrics_fn, commit_candidates, checkpoint, rss=lambda: None):
    policy = history.policy
    commit_candidates(history)  # Recommit only verified local records before any new fit.
    progress.restore_history(history.rows, history.ranked)
    for phase in ("COARSE", "FINE"):
        grid = policy["coarse_grid"] if phase == "COARSE" else ensure_fine_plan(state, history, checkpoint)
        progress.set_phase(phase, len(grid), [r["unit_seconds"] for r in history.rows if r["phase"] == phase])
        for k in grid:
            if any(row["k"] == k for row in history.rows):
                continue
            progress.check()
            progress.event("candidate_started", phase=phase, k=k, phase_total=len(grid))
            started, clock = utc(), time.perf_counter()
            train_k, val_k = svc_prefix(X, k), svc_prefix(V, k)
            rss_before = rss()
            model = model_factory()
            model, measurement = fit_fn(model, train_k, y)
            prediction_started = time.perf_counter()
            prediction = model.predict(val_k)
            predict_seconds = time.perf_counter()-prediction_started
            metrics = metrics_fn(vy, prediction)
            row = {"phase": phase, "k": k, "features": history.ranked[:k], **metrics,
                   "fit": measurement, "predict_seconds": predict_seconds,
                   "unit_seconds": time.perf_counter()-clock,
                   "started": started, "finished": utc(),
                   "rss_before_fit_bytes": rss_before, "rss_after_predict_bytes": rss()}
            del model, train_k, val_k, prediction
            gc.collect()
            history.append(row)  # Fully validated, hash chained, atomic no-clobber.
            progress.restore_history(history.rows, history.ranked)
            best = winner(history.rows)
            progress.event("candidate_completed", phase=phase, k=k,
                validation_accuracy=row["accuracy"], best_k=best["k"],
                best_validation_accuracy=best["accuracy"], local_completed_candidates=len(history.rows))
            commit_candidates(history)  # No next fit before this candidate is COMMITTED/readback verified.
            progress.check()  # Safe stop after the durable candidate, never mid-SVC.
        if phase == "COARSE":
            ensure_fine_plan(state, history, checkpoint)
    validate_rows(history.rows, history.ranked, policy,
                  validation_count=len(vy), complete=True)
    return history
