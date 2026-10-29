"""Explicit MI/RFE import from the unchanged CICFLOW reference-layout SVM path."""
import csv
import math
import shutil
from pathlib import Path
from .common import Blocked, atomic, check_files, core, digest, info, read

FILES = ("run_identity.json", "mi_state.json", "mi_top100.json", "mi_feature_scores.csv",
         "rfe_state.json", "rfe_feature_ranking.csv")


def validate_selection(state, identity):
    state = Path(state)
    names = identity["dataset"]["names"]
    top, mi, rfe = read(state / "mi_top100.json"), read(state / "mi_state.json"), read(state / "rfe_state.json")
    if mi["identity"] != identity["sha256"] or rfe["binding"] != {
            "identity": identity["sha256"], "mi_state": digest(mi), "top_order": digest(top)}:
        raise Blocked("MI/RFE binding mismatch")
    check_files(state, mi["files"])
    check_files(state, rfe["files"])
    with (state / "mi_feature_scores.csv").open(newline="", encoding="utf-8") as f:
        scores = list(csv.reader(f))[1:]
    with (state / "rfe_feature_ranking.csv").open(newline="", encoding="utf-8") as f:
        ranks = list(csv.reader(f))[1:]
    if (len(names) != 380 or len(set(names)) != 380 or len(scores) != 380
            or set(r[0] for r in scores) != set(names) or [r[0] for r in scores[:100]] != top
            or len(top) != 100 or len(set(top)) != 100 or not set(top) <= set(names)):
        raise Blocked("MI feature mapping/order invalid")
    values = [float(r[1]) for r in scores]
    if any(not math.isfinite(v) or v < 0 for v in values) or values != sorted(values, reverse=True):
        raise Blocked("MI score order invalid")
    ranked = rfe["names"]
    if (len(ranked) != 100 or set(ranked) != set(top) or [r[0] for r in ranks] != ranked
            or [int(r[1]) for r in ranks] != list(range(1, 101))
            or sorted(rfe["ranking"]) != list(range(1, 101))
            or [top.index(n) for n in ranked] != sorted(range(100), key=lambda i: rfe["ranking"][i])
            or rfe.get("completed_eliminations") != 99 or rfe.get("final_estimator_fit_performed") is not True):
        raise Blocked("SVM full RFE ranking invalid/incomplete")
    return top, ranked


def inspect_import(request, audit, environment):
    if not request.get("reuse_from"):
        return None
    directory = Path(request["reuse_from"]).resolve()
    old = read(directory / "run_identity.json")
    module = core(request["source"])
    from sklearn.ensemble import RandomForestClassifier
    resolved = {"classifier": module.make_model("svm").get_params(),
                "rfe": RandomForestClassifier(**module.RFE_PARAMS["svm"]).get_params()}
    if (old.get("format") != module.FORMAT or old.get("spec", {}).get("experiment") != "mi_rfe_svm"
            or old["spec"]["config"] != request["config"] or old.get("contract") != module.contract("svm")
            or old.get("dataset") != audit["binding"] or old.get("environment", {}).get("versions") != environment["versions"]
            or old["environment"].get("python") != environment["python"]
            or old.get("source") != module.source_snapshot(request["source"])
            or old.get("resolved_estimators") != resolved
            or digest({k: v for k, v in old.items() if k != "sha256"}) != old.get("sha256")):
        raise Blocked("Import is not the same SVM/reference MI/data/environment variant; no silent recompute")
    validate_selection(directory, old)
    return {"directory": str(directory), "source_identity": old["sha256"],
            "git_commit_at_origin": old.get("git_commit"),
            "files": {p: info(directory / p) for p in FILES}}


def import_selection(state, identity, binding):
    if binding is None or (Path(state) / "rfe_state.json").exists():
        return
    source, state = Path(binding["directory"]), Path(state)
    check_files(source, binding["files"])
    for name in ("mi_top100.json", "mi_feature_scores.csv", "rfe_feature_ranking.csv"):
        if (state / name).exists():
            if info(state / name) != info(source / name):
                raise Blocked("Partial imported artifact differs; do not overwrite")
            continue
        with (source / name).open("rb") as src, (state / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
    old_mi, old_rfe = read(source / "mi_state.json"), read(source / "rfe_state.json")
    mi = {"identity": identity["sha256"], "seconds": old_mi.get("seconds"),
          "origin": binding, "reused_without_fit": True,
          "files": {p: info(state / p) for p in ("mi_top100.json", "mi_feature_scores.csv")}}
    if (state / "mi_state.json").exists():
        if read(state / "mi_state.json") != mi:
            raise Blocked("Partial imported MI state differs")
    else:
        atomic(state / "mi_state.json", mi, exclusive=True)
    top = read(state / "mi_top100.json")
    rfe = {**old_rfe, "binding": {"identity": identity["sha256"], "mi_state": digest(mi), "top_order": digest(top)},
           "origin": binding, "reused_without_fit": True,
           "files": {"rfe_feature_ranking.csv": info(state / "rfe_feature_ranking.csv")}}
    atomic(state / "rfe_state.json", rfe, exclusive=True)
    validate_selection(state, identity)
