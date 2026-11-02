from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from stage9_svm_final.common import (FORMAT, SCALER_PROTOCOL, SVM_PARAMS, digest, info, paths, once)

SOURCE = Path(__file__).resolve().parents[2]


def fixture(root, method="A1"):
    root = Path(root)
    names = [f"feature_{i}__mean" for i in range(380)]
    features = [names[7], names[2], names[13]]
    dataset = root / "CICFLOWMETER_FEATURE_CSV"
    folder = dataset / "FW256_FS8"
    folder.mkdir(parents=True, exist_ok=True)
    splits, histograms, matrices = {}, {}, {}
    for role, n in (("train", 8), ("val", 4), ("test", 4)):
        X = np.asarray(np.arange(n*380).reshape(n, 380)+{"train": 0, "val": 100, "test": 200}[role], dtype=np.float32)
        y = np.asarray([0, 1]*(n//2), dtype=np.int8)
        table = pa.table({**{name: X[:, i] for i, name in enumerate(names)}, "Window_Label": y})
        path = folder / (role + "_windows.parquet")
        pq.write_table(table, path)
        splits[role] = dict(path=str(path), rows=n, declared_sha256=info(path)["sha256"])
        histograms[role] = {"0": n//2, "1": n//2}
        matrices[role] = X[:, [7, 2, 13]], y
    audit = dict(binding=dict(names=names, splits=splits), label_histograms=histograms,
        label_mapping={"0": "Benign_folder_proxy", "1": "Attack_folder_proxy", "status": "PROVISIONAL_FOLDER", "producer_sha256": "a"*64},
        split_evidence={"label_producer": {"sha256": "a"*64}})
    audit["fingerprint"] = digest(audit)
    from sklearn.svm import SVC
    identity = dict(format=FORMAT, method=method, config="FW256_FS8", run_id="RUN_S9_UNIT",
        selection_run_id="RUN_A1_UNIT" if method == "A1" else "RUN_B2_UNIT",
        selection_identity="b"*64, selection_checkpoint_sha256="c"*64,
        features=features, best_k=3, rfe_step=5 if method == "A1" else 2,
        svm=SVM_PARAMS, svm_resolved=SVC(**SVM_PARAMS).get_params(), scaler_protocol=SCALER_PROTOCOL,
        labels={"benign": 0, "attack": 1}, dataset=audit)
    frozen = dict(features=features, best_k=3, identity="b"*64)
    identity["frozen_digest"] = digest(frozen)
    identity["sha256"] = digest(identity)
    request = dict(source=str(SOURCE), dataset=str(dataset), output=str(root / "experiment_results"),
        local_root=str(root / "local"), production=False, method=method, config="FW256_FS8",
        run_id=identity["run_id"], selection_run_id=identity["selection_run_id"], heartbeat_seconds=15,
        budget=dict(min_available_ram_bytes=2*1024**3, max_process_rss_bytes=40*1024**3))
    return request, dict(status="PASS", identity=identity, selected=dict(frozen=frozen)), matrices


def final_bundle(request, plan):
    """Use real reference scaler/dumping, replace only costly synthetic SVC.fit."""
    from unittest.mock import patch
    from mi_rfe_svm_colab.common import core
    from stage9_svm_final.common import confirmation
    from stage9_svm_final.pipeline import fit_final
    from stage9_svm_final.telemetry import Heartbeat
    def fake_fit(model, X, y):
        model.n_features_in_ = X.shape[1]
        model.classes_ = np.unique(y)
        return model, dict(seconds=0.01, converged=True, warnings=[], fit_status=0)
    request = {**request, "enable_final_fit": True, "final_confirmation": confirmation(request, "FINAL_FIT")}
    with patch.object(core(request["source"]), "timed_fit", side_effect=fake_fit), patch.object(Heartbeat, "emit"):
        result = fit_final(request, plan)
    return result
