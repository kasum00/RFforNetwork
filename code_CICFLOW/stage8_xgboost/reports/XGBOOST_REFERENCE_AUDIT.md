# XGBoost reference audit — CICFLOW — 2026-10-29

## Decision

Implement **baseline_xgb** for four independent CICFLOW configurations.
MI-RFE-XGB remains **BLOCKED_PENDING_SEPARATE_RFE_POLICY_APPROVAL**. This is
a research adaptation, not an official CTU-13 reproduction. No RF output is
loaded by this implementation.

## Actual source reviewed

The complete 14-cell notebook
`D:\COMPACT-MI-RFE\notebooks\STAGE8_XGBOOST_COLAB.ipynb` was read as JSON/source.
Its Cell8 permits only baseline_xgb; choosing mi_rfe_xgb explicitly raises.
It uses one configuration per run, not a four-configuration batch. The source
module contains a proposed compact extension, but both notebook and official
checkpointed worker reject it until a separate policy is approved.

The current `configs/stage8_xgboost_policy.BLOCKED.json` says approved=false.
Its proposal reuses the RF-branch RF-RFE estimator; that is an extension, not
an independently established CTU-13 XGB ranking contract. Reading five CTU-13
scripts does not establish an approved standalone XGBoost classifier/ranking
path. A comment mentioning XGB in the RF script is not such a contract.

Relevant reference symbols in `huyen_code/ciciomt24_xgboost.py`:
XGB_REQUIRED_PARAMS, XGB_FIXED_POLICY, make_xgb, effective_xgb_parameters,
load_rfe_policy, validation_baseline_xgb, fit_final_xgb, run_worker.
The independent checkpoint source was read in full and copied without changes
into this Stage8 package, preserving its API and failure semantics. Its pure
fake-model checkpoint/protocol tests were adapted to the new import path;
legacy notebook-inspection tests were excluded because they inspect another
notebook/dataset.

Local reference SHA256 at inspection:

| Source | SHA256 |
|---|---|
| notebooks/STAGE8_XGBOOST_COLAB.ipynb | 984ce47e0d0eeed04be82eb3e63fe9d33dc6ce36681a41f5c9562c5afaff8350 |
| huyen_code/ciciomt24_xgboost.py | b4d6eef8044c25304c7a1f22186674d98a28de957dd9fda7587decbccb04fc1f |
| huyen_code/ciciomt24_xgboost_checkpoint.py | fd3e8090ebb52f282779e3a97da6f9a808d53fe632b08232c400a8196f7324cd |
| configs/stage8_xgboost_policy.BLOCKED.json | 778de88d604ef834ed8bc1ecafdea115239e2ca094b238ab19e678a0c82cefa3 |

## Dataset/preprocessing differences

Reference: CICIoMT2024 C+ WINDOWS_CPLUS CSV, project root beneath
MyDrive/COMPACT-MI-RFE, canonical names FW_256_FS_8 etc., and source C+ counts.
New adapter: MyDrive/CICFLOWMETER_FEATURE_CSV Parquet, canonical names
FW256_FS8 etc., exactly the following independent split counts:

| Config | Train | Validation | Test |
|---|---:|---:|---:|
| FW256_FS8 | 326157 | 104147 | 103494 |
| FW256_FS16 | 163203 | 52145 | 51824 |
| FW512_FS8 | 311423 | 96408 | 95865 |
| FW512_FS16 | 155814 | 48258 | 47991 |

The actual local dataset layout was inspected by reading directory names and
small JSON sidecars only, not Parquet values. Each folder has existing
window_integrity.json and window_feature_schema.json. Their split counts,
label histograms, aggregate_feature_count380/output_labelWindow_Label and
PROVISIONAL_FOLDER status were inspected. These sidecars must be present on
Drive; the adapter will not regenerate them or repair/re-split data.

Metadata admission binds ordered schema, exact counts, file size/mtime, footer
schema and small sidecar hashes. Actual binary labels must match bound
histograms when each role is opened. The fingerprint is NOT a full Parquet
content hash. Sidecar consistency is not proof of label meaning or independent
captures/sessions.

Preprocessing follows the existing CICFLOW and training-core contract:
numeric float64, NaN/+Inf/-Inf→zero, cast finite float32, preserve all rows.
Validation: StandardScaler fit Train380; Validation transform only. Baseline
has no MI/RFE/k-search: feature_selection.status=NOT_APPLICABLE, best_k=null,
all380 frozen. Final refit: new scaler and XGBoost on C-order float32
Train+Validation. Test values are opened only after final artifacts and durable
protocol claim; one evaluation per configuration.

## Scientific parameters and backend

Keep source/user classifier200/depth10/min_child_weight3, binary:logistic,
logloss, seed42, n_jobs=-1, hist. No automatic class balancing, scale_pos_weight
tuning, early stopping, sampling or parameter search. Booster, learning rate,
subsample, colsample variants and regularization retain pinned XGBoost3.4.1
defaults and are resolved from native configuration, not guessed in code.
Unresolvable defaults BLOCK training. All constructor and native settings are
stored in configuration.json and final booster_config.json.

The source dependency profile is preserved: numpy2.2.6, pandas2.3.3,
scipy1.15.3, sklearn1.7.2, joblib1.5.3, psutil5.9.5, XGBoost3.4.1;
CUDA additionally needs nvidia-ml-py13.580.82. PyArrow is an additional Parquet
dependency and its actual version is identity-bound. No automatic pip occurs.
Missing/unavailable profiles are BLOCKED, not silently replaced.

CPU is the reference/default backend. CUDA is explicit, requires compatible
build/NVML/GPU/VRAM, and actual configured device is checked before/after fit
and at inference. No CPU fallback and no CPU/CUDA bitwise-equivalence claim.

## Resource adaptation, not scientific-parameter alteration

The C+ notebook's review budget is44GiB start RAM/40GiB disk with matrix cache.
The new Parquet CICFLOW notebook declares a separate explicit budget:
8GiB start RAM,4GiB runtime available RAM,40GiB process-tree RSS cap,
4GiB free disk,4GiB free GPU VRAM when CUDA,43200seconds per configuration.
Admission also applies a dataset-size-based RAM estimate. No matrix cache or
Parquet copy is made. These are admission heuristics, not benchmarks or a
promise of runtime/fit success. The budget is visible, included in confirmation
and checkpoint identity, and never lowered automatically to achieve PASS.

## MI/RF evidence is not transferred

The existing MI_REFERENCE_COMPATIBILITY_AUDIT.md was read. Its per-feature
seed42+index/no-external-scaler MI is PROVISIONAL_REFERENCE_UNVERIFIED, unlike
a scaled full380 one-call reference. The corrected RF notebook has a distinct
MI variant, but neither its ranking nor RF-selected best-k establishes an
XGB selection policy. RF verify PASS cannot authorize MI-RFE-XGB or become an
XGB checkpoint. All mi_rfe_rf_* folders and old Stage4/5 artifacts are preserved.

## Checkpoint and scientific limitations

Checkpointed validation_fit/final_refit use10-round continuation blocks to
exactly200 boosted rounds; only the latest two model+manifest pairs per stage
are retained by the copied reference manager. No partial fit inside a block
is resumable. Compatibility binds ordered features, dataset, source, commit,
library/environment, hyperparameters, backend, budget and run identity.
Corrupt newest/orphan checkpoints block rather than silently falling back.

The copied protocol uses a legacy field official_status=COMPLETED to denote
protocol completion. Every new run/report/completion marker separately keeps
scientific_status=PROVISIONAL_RESEARCH and official_status=OFFICIAL_BLOCKED.
An internal legacy field never promotes scientific status.

No new test, preflight, training or numeric equivalence evidence was produced
by Codex. Historical source test claims are not transferred to this adapter.
Resume numerical parity with a one-call200-round fit is not newly proven;
the approved reference checkpointed algorithm is retained and its actual
behavior must be tested by the user. Proxy label validity and independent
capture/session evidence remain unresolved.
