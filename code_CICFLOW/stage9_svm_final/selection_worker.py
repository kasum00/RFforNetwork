"""Isolated original-source verifier; never calls run(), fit(), selection or Test."""
import base64
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from .common import Blocked, METHODS, digest, files_checked, info, intact, once, read, selection_path, sha


def frozen_evidence(request, package):
    if request.get("evidence_root"):
        return request["evidence_root"]
    source = Path(request["selection_source"])
    bundle = read(source / package / "evidence/cicflow_split_evidence.json")["files"]
    pre = read(Path(request["dataset"]) / request["config"] / ".stage3_preflight.json")
    root = Path(request["local_root"]) / "evidence" / request["method"] / digest(pre["binding"])
    v2 = None
    for key in ("validation", "v2_preflight", "manifest", "v2_binding", "label_producer"):
        record = bundle[key]
        expected = v2["v2_code_sha256"] if key == "label_producer" else pre["binding"]["input_sha256"][key]
        relative = ("code_CICFLOW/stage29_v2_allocation.py" if key == "label_producer"
                    else pre["binding"]["input_paths"][key].replace("\\", "/"))
        name = relative.rsplit("/", 1)[-1]
        if (name != record["source_relative_path"].replace("\\", "/").rsplit("/", 1)[-1]
                or record["sha256"] != expected or not 0 < record["bytes"] <= 8*1024**2):
            raise Blocked("Frozen evidence mapping mismatch")
        raw = gzip.decompress(base64.b64decode(record["gzip_base64"], validate=True))
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != expected:
            raise Blocked("Frozen evidence byte/SHA mismatch")
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        if target.exists():
            if target.read_bytes() != raw:
                raise Blocked("Existing local evidence differs; preserve it")
        else:
            with target.open("xb") as handle:
                handle.write(raw)
        if key == "v2_binding":
            v2 = json.loads(raw.decode("utf-8-sig"))
    return str(root)


def verify(request):
    directory = selection_path(request)
    if not (directory / "results/SELECTION_COMPLETE.json").is_file():
        raise Blocked("BLOCKED_SELECTION_INCOMPLETE")
    identity = intact(read(directory / "run_manifest.json")["identity"])
    package, algorithm, step = METHODS[request["method"]]
    if (identity["algorithm"] != algorithm or identity["config"] != request["config"]
            or identity["run_id"] != request["selection_run_id"] or identity["contract"]["rfe"]["step"] != step):
        raise Blocked("Selection binding mismatch")
    source = Path(request["selection_source"]).resolve()
    repo = source.parent
    for args, expected in ((("rev-parse", "HEAD"), identity["git_commit"]), (("status", "--porcelain"), "")):
        actual = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
        if actual != expected:
            raise Blocked("Original selection source checkout must be clean and at its saved commit")
    # Check every recorded source byte BEFORE importing that checkout.
    recorded_root = repo if request["method"] == "B" else source
    records = identity["source"]["files"]
    if identity["source"]["sha256"] != digest(records):
        raise Blocked("Selection source digest differs")
    for name, expected in records.items():
        p = Path(name)
        target = recorded_root / p
        if (p.is_absolute() or ".." in p.parts or target.is_symlink()
                or recorded_root not in target.resolve().parents or sha(target) != expected):
            raise Blocked("Selection source byte mismatch: " + name)
    for name, version in identity["environment"]["versions"].items():
        if importlib.metadata.version(name) != version:
            raise Blocked("Selection environment version mismatch: " + name)
    if platform.python_version() != identity["environment"]["python"]:
        raise Blocked("Selection Python version mismatch; do not bypass joblib/environment binding")
    sys.path.insert(0, str(source))
    contract = importlib.import_module(package + ".contract")
    if contract.source_snapshot(source) != identity["source"]:
        raise Blocked("Original selection source snapshot differs")
    if identity["format"] != contract.FORMAT:
        raise Blocked("Selection format differs from its archived source")
    if request["method"] == "B":
        policy = importlib.import_module(package + ".policy")
        policy.validate_policy(identity["search_policy"])
        expected_contract = contract.scientific(source, identity["search_policy"])
    else:
        expected_contract = contract.scientific(source)
    if identity["contract"] != expected_contract:
        raise Blocked("Selection scientific contract differs from its archived implementation")
    module = contract.core(source)
    from sklearn.ensemble import RandomForestClassifier
    if identity["resolved_estimators"] != {
            "svm": module.make_model("svm").get_params(),
            "rfe": RandomForestClassifier(**module.RFE_PARAMS["svm"]).get_params()}:
        raise Blocked("Selection resolved estimator parameters differ")
    pipeline = importlib.import_module(package + ".pipeline")
    results = importlib.import_module(package + ".results")
    store = importlib.import_module(package + ".store")
    temporary = Path(request["local_root"]) / "verify_selection"
    verified = pipeline.verify(directory, identity, temporary)
    rows = store.ledger(directory, identity)
    artifacts = {}
    for row in rows:
        for name in row["record"]["files"]:
            artifacts[name] = row["path"] / name
    # Do not accept pre-existing final/Test work in the selection namespace.
    forbidden = {"TEST_CLAIM.json", "test_result.json", "test_metrics.json", "test_ledger.json",
                 "TEST_COMPLETE.json", "FINAL_COMPLETE.json", "final_state.json", "model.joblib"}
    if (any(Path(name).name in forbidden for name in artifacts)
            or any((directory / sub / name).exists() for sub in ("", "state", "results", "final_model") for name in forbidden)):
        raise Blocked("Existing final/Test work requires separate reconciliation; no re-evaluation")
    # Additional durable k-prefix checks for A1 (including migrated k=1).
    if request["method"] == "A1":
        for k in range(1, 101):
            name = f"k/k_{k:03d}.json"
            owners = [i for i, row in enumerate(rows) if name in row["record"]["files"]]
            selection_index = next(i for i, row in enumerate(rows) if row["record"]["stage"] == "selection")
            if len(owners) != 1 or owners[0] >= selection_index:
                raise Blocked("A1 k history was not individually committed before selection")
    selection = read(directory / "results/selection.json")
    frozen = read(artifacts["selection_frozen.json"])
    rfe = read(artifacts["rfe_state.json"])["result"]
    import joblib
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(artifacts["top100_scaler.joblib"])
    if (type(scaler) is not StandardScaler or scaler.get_params() != StandardScaler().get_params()
            or scaler.n_features_in_ != 100 or int(scaler.n_samples_seen_) != identity["dataset"]["splits"]["train"]["rows"]):
        raise Blocked("Selection scaler fitted dimensions/count differ")
    if module.contract("svm")["final_scaler_and_classifier_fit_role"] != "Train+Validation after frozen selection":
        raise Blocked("BLOCKED_SCALER_PROTOCOL_UNRESOLVED")
    data = importlib.import_module(package + ".dataset")
    effective = {**request, "source": str(source), "evidence_root": frozen_evidence(request, package)}
    audit = data.inspect(effective)  # Train/Val full hashes; Test STAT ONLY.
    if audit != identity["audit"]:
        raise Blocked("Dataset/split/schema identity changed since selection")
    return dict(identity=identity, verification=verified, audit=audit, best_k=selection["best_k"],
        features=selection["selected_features"], ranked=rfe["names"], frozen=frozen,
        checkpoint_sha256=read(directory / "results/SELECTION_COMPLETE.json")["checkpoint"],
        publication=info(directory / "results/artifact_manifest.json"),
        selection_scaler=info(artifacts["top100_scaler.joblib"]),
        verified_helpers={name: sha(source / name) for name in
                          ("colab/cicflow_reference_pipeline.py", "mi_rfe_svm_colab/common.py")},
        reference_final_role=module.contract("svm")["final_scaler_and_classifier_fit_role"])


if __name__ == "__main__":
    try:
        print("__STAGE9_SELECTION__" + json.dumps(verify(json.load(sys.stdin)), allow_nan=False), flush=True)
    except Exception as exc:
        print(f"BLOCKED_SELECTION_NOT_VERIFIED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
