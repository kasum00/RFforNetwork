# Phương án A1 — MI–RF-RFE(step5,target1)–SVM

**EXPERIMENTAL_NON_OFFICIAL**. Written only; Codex has not executed Python, shell, tests, preflight, training, Git, packages, benchmarks or notebook cells.

Open `notebooks/PHUONG_ANA_A1_COLAB.ipynb`. Detailed instructions: `reports/COLAB_GUIDE.md`. Scientific/durability contract: `reports/A1_DESIGN.md`. Comparison template: `reports/A0_A1_COMPARISON.md`.

A1 reuses the existing A0 CICFLOW MI, cleaning, Train Top100 scaler, SVC constructor/timing/metrics, ranked workspaces, exhaustive k loop and best-k rule. Only RF-RFE step changes scientifically:1→5, target remains1. A1 adds an explicit total order inside tied sklearn rank groups, with removal-round squared importance descending then original MI position ascending. The actual sklearn group ranks are retained separately.

RF-RFE estimator remains100 trees, depth15, balanced, seed42, n_jobs=-1. SVM remains RBF, C10, gamma scale, probabilityFalse, cache10000, shrinkingTrue, max_iter20000, balanced, seed42. Train/Validation rows and all k=1..100 are preserved. This experiment ends at frozen Validation selection. No final Train+Validation refit or Test access is implemented.

Exact notebook paths:

```python
DRIVE = Path('/content/drive/MyDrive')
DATASET_ROOT = DRIVE / 'CICFLOWMETER_FEATURE_CSV'
OUTPUT_ROOT = DRIVE / 'experiment_results'
LOCAL_ROOT = Path('/content/mi_rfe_svm_5h')
```

Remote root: `experiment_results/phuong_anA/mi_rfe_svm/<CONFIG>/<RUN_ID>/`. Local root appends the same namespace/config/ID under LOCAL_ROOT. A0 outputs and phuong_anB are not edited or reused as checkpoints.

Checkpoint every stage and completed k, including RFE elimination steps. No SVC mid-fit recovery. The `5h` pathname does not define a runtime limit; the notebook exposes session budget/margin. A1 uses periodic measured heartbeat and per-k durable checkpoints; A0's five-hour Drive policy remains unchanged.

All scientific source/dataset/split/config/order/scaler/parameters/algorithm identities must match on resume. Missing or invalid original manifest evidence is BLOCKED. The included evidence snapshot contains only original small CICFLOW lineage manifests and frozen label-producer source bytes, verified against each selected config's Stage3 hashes. It does not create a split/label or import a CICIoMT2024 row count. Full Train/Validation file SHA256 is checked. Test is never opened or hashed; its declared manifest SHA/stat is bound for future review.

The dataset reports folder-derived proxy labels and unresolved capture/session/near-duplicate provenance. Those limits are retained; technical PASS never becomes official reproduction. Neither runtime speedup nor A0/A1 ranking or metric equality has been demonstrated.
