# COMPACT-MI-RFE — Stage 5A RF-RFE Implementation Report

## Scope and status

Stage 5A is implemented under:

`D:\COMPACT-MI-RFE\code_CICFLOW\stage5_rfe`

It consumes the four verified-but-provisional Stage 4 MI runs supplied by the
user. It does not modify Stage 0–4 source, artifacts, checkpoints, dataset
files, or the five CTU-13 scripts.

Scientific status remains:

- `PROVISIONAL_RESEARCH`
- `PROVISIONAL_REFERENCE_UNVERIFIED`
- `official_reproduction=false`

No command, test, training, RFE fit, or benchmark was executed while creating
this implementation.

## Scientific contract audit

The existing Stage 4 audit records that exact CTU-13 reference equivalence is
not established. The current MI implementation uses per-feature calls and a
per-feature seed rule, so Stage 5 does not promote the MI outputs to official
reference data. Stage 5 binds to the exact Stage 4 run identity and refuses to
continue if the MI source identity changes.

The Stage 5 implementation does not silently change MI preprocessing,
StandardScaler behavior, `mutual_info_classif` invocation, seed handling, or
ranking semantics.

## Input validation

Before Train values are read, Stage 5 validates:

- `MI_COMPLETE.json` exists and is `COMPLETE`.
- `mi_manifest.json`, `top100_features.json`, and `mi_ranking.json` exist.
- The Stage 4 configuration matches the requested Stage 5 configuration.
- The MI manifest remains non-official.
- The current Train footer has exactly 380 features followed by
  `Window_Label`.
- The Train row count matches the Stage 3 configuration contract.
- The current source identity digest equals the Stage 4 manifest identity.
- The full MI ranking has 380 entries and matches `MI_COMPLETE.json`.
- Top-100 has exactly 100 unique names/indices and preserves MI order.
- Every Top-100 name maps to the same original 380-feature schema position.
- No provenance-like field or label is selected as a feature.

Only after these metadata checks and the resource gate does Stage 5 read the
selected columns from the corresponding Train Parquet. Validation and Test
Parquet files are never opened by Stage 5A.

## RFE branches

### RF branch

```text
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

### SVM-RFE branch

The project branch label is `svm`, but the requested RFE estimator is the
specified RandomForestClassifier:

```text
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

`min_samples_split` and `max_features` were not specified for this branch and
are therefore recorded as unresolved. The implementation uses the sklearn
constructor defaults provisionally and refuses to call this reference-
compatible. These values must be reconciled with source before scientific
promotion; they were not altered to obtain a passing result.

### RFE semantics

- `n_features_to_select=1`.
- `step=1`.
- sklearn-compatible `importance_getter="auto"` behavior for the RF estimator.
- The same `np.argsort(importances)` call semantics used by sklearn RFE; no
  custom estimator tie-break is introduced.
- Full 100-feature ranking retained, with raw `rfe_rank`, selection order,
  original MI rank, original 380-feature index, name, and MI score.
- RF and SVM branches instantiate and fit separate estimators. Ranking from
  one branch is never reused for the other.
- XGBoost is `BLOCKED`; no estimator or ranking contract was guessed.

The manual loop exists to checkpoint after each elimination. The unit tests
compare its ranking/support output with public `sklearn.RFE(..., step=1)` on
synthetic fixtures.

## Checkpoint and resume

Each branch writes to a new exclusive directory:

`stage5_rfe/runs/<CONFIG>/<RUN_ID>/<BRANCH>/`

The run identity includes:

- configuration, branch, run ID;
- Stage 4 MI run ID and source identity digest;
- Stage 4 ranking digest and Top-100 order digest;
- dataset fingerprint;
- supplied git commit or explicit `NOT_CAPTURED`;
- branch estimator parameters;
- RFE parameters;
- scientific/reference status.

`rfe_progress.json` is atomically refreshed after every step-one elimination.
Resume requires an exact identity match and refuses completed runs. A resource
failure creates `BLOCKED.json`; a fit failure creates `FAILED.json`. The
`COMPLETE.json` marker is written last.

## Expected artifacts per branch

- `RUN_INCOMPLETE.json`
- `BLOCKED.json` or `FAILED.json` when applicable
- `rfe_progress.json`
- `rfe_ranking.json`
- `rfe_ranking.csv`
- `rfe_manifest.json`
- `resource_usage.json`
- `rfe_report.md`
- `COMPLETE.json` only after all artifacts are complete

## Stage 5B boundary

Stage 5B is specified separately in
[STAGE5B_CONTRACT.md](STAGE5B_CONTRACT.md). No k-search, final refit, Test
read, or evaluation was implemented as a runtime action in this task.

