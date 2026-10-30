import csv
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from phuong_anB_step2.contract import atomic, core, digest, info, sha
from phuong_anB_step2.history import History, check_committed_candidates
from phuong_anB_step2.rfe import fit_step2
from phuong_anB_step2.search import search
from .fixtures import SOURCE, small_identity


class TiedForest(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        return self


class FakeSVC:
    """Predict actual binary fixtures; fit/predict calls are observable."""
    def __init__(self, calls, predictions, scores=None):
        self.calls, self.predictions, self.scores = calls, predictions, scores or {}

    def fit(self, X, y):
        self.width = X.shape[1]
        self.calls.append(self.width)
        self.fit_status_ = 0
        self.n_iter_ = np.array([1])
        self.support_ = np.array([0, 1])
        return self

    def predict(self, X):
        self.predictions.append(self.width)
        labels = np.array([0, 1] * (len(X)//2))
        correct = self.scores.get(self.width, len(X)//2)
        labels[correct:] = 1-labels[correct:]
        return labels


class Progress:
    def __init__(self, stop_after=None):
        self.events, self.stop_after, self.rows = [], stop_after, []

    def check(self):
        from phuong_anB_step2.telemetry import Pause
        if self.stop_after is not None and len(self.rows) >= self.stop_after:
            raise Pause("Synthetic safe stop")

    def set_phase(self, phase, total, durations=()):
        self.phase, self.total = phase, total

    def event(self, event, **details):
        self.events.append((event, details))

    def restore_history(self, rows, ranked):
        self.rows = list(rows)


def history_fixture(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    identity = small_identity()
    count = 100
    names = identity["dataset"]["names"][:count]
    atomic(state / "rfe_state.json", {"fixture_ranking": names}, exclusive=True)
    return identity, names, History(state, identity, names, sha(state / "rfe_state.json"))


def run_search(state, history, *, calls, predictions, scores=None, progress=None, commit=None, checkpoint=None):
    module = core(SOURCE)
    count = len(history.ranked)
    X, V = np.ones((8, count), np.float32), np.ones((20, count), np.float32)
    y, vy = np.array([0, 1]*4), np.array([0, 1]*10)
    return search(Path(state), history, X, y, V, vy, progress or Progress(),
        model_factory=lambda: FakeSVC(calls, predictions, scores),
        fit_fn=module.timed_fit, metrics_fn=module.metrics,
        commit_candidates=commit or (lambda current: None),
        checkpoint=checkpoint or (lambda *args, **kwargs: None))


def commit_history(store, history):
    durable = check_committed_candidates(store, history)
    for sequence in range(durable+1, len(history.rows)+1):
        path, row = history.path(sequence), history.rows[sequence-1]
        store.commit(history.stage(sequence), {path.relative_to(store.state).as_posix(): path},
                     metadata={"phase": row["phase"], "k": row["k"]})


def complete_state(state, identity=None):
    """Small measured fixtures; no real SVC/RF and no production dataset."""
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    identity = identity or small_identity()
    names, top = identity["dataset"]["names"], identity["dataset"]["names"][:100]
    atomic(state / "run_identity.json", identity, exclusive=True)
    with (state / "mi_feature_scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "mi_score"])
        writer.writerows((name, 1.0-index/1000) for index, name in enumerate(names))
    atomic(state / "mi_top100.json", top, exclusive=True)
    atomic(state / "mi_state.json", {"identity": identity["sha256"], "seconds": 0.1,
        "files": {n: info(state / n) for n in ("mi_feature_scores.csv", "mi_top100.json")}}, exclusive=True)
    (state / "top100_scaler.joblib").write_bytes(b"fixture scaler: never deserialized")
    atomic(state / "top100_scaler_state.json", {
        "binding": {"identity": identity["sha256"], "mi": sha(state / "mi_state.json"), "top": top},
        "files": {"top100_scaler.joblib": info(state / "top100_scaler.joblib")},
        "fit_role": "train", "rows": 8}, exclusive=True)
    result = fit_step2(np.ones((8, 100), np.float32), np.array([0, 1]*4), top, TiedForest())
    with (state / "rfe_feature_ranking.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["records"][0]))
        writer.writeheader(); writer.writerows(result["records"])
    atomic(state / "rfe_state.json", {"binding": {"identity": identity["sha256"],
        "mi": sha(state / "mi_state.json"), "scaler": sha(state / "top100_scaler_state.json"), "top": digest(top)},
        "result": result, "files": {"rfe_feature_ranking.csv": info(state / "rfe_feature_ranking.csv")}}, exclusive=True)
    history = History(state, identity, result["names"], sha(state / "rfe_state.json"))
    calls, predictions = [], []
    run_search(state, history, calls=calls, predictions=predictions, scores={50: 18, 45: 19, 47: 20})
    receipt = {"identity": identity["sha256"], "attempt": "unit",
        "outcome": "SELECTION_WORK_COMPLETE", "started": "synthetic", "finished": "synthetic",
        "wall_seconds": 2.0, "rf_started": 51, "rf_completed": 51,
        "svm_started": len(calls), "svm_completed": len(calls),
        "peak_rss_bytes": 1024, "prior_receipts": {}}
    receipt["sha256"] = digest(receipt)
    atomic(state / "sessions/session_unit.json", receipt, exclusive=True)
    from phuong_anB_step2.results import create_selection
    create_selection(state, identity, history, core(SOURCE))
    return identity, history
