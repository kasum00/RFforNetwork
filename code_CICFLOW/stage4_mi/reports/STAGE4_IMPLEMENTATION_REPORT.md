# COMPACT-MI-RFE — CICFLOW Stage 4 Implementation Report

## Scope

Stage 4 implements a guarded, Train-only Mutual Information (MI) selector for
the four existing CICFLOW sliding-window configurations:

- `FW256_FS8`
- `FW256_FS16`
- `FW512_FS8`
- `FW512_FS16`

All Stage 4 files are contained in:

`D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi`

Stage 0–3 artifacts are treated as read-only. No Validation or Test feature
values are read by the metadata/resource preflight or by the MI run path.

## Current scientific position

The user-run Stage 0 result supplied for this continuation is recorded as:

| Gate | Result |
|---|---|
| Technical readiness | `READY` |
| Provisional research readiness | `PROVISIONAL_RESEARCH` |
| Official reproduction | `BLOCKED` |
| Configuration coverage | 4 configurations, 8/8 technical checks |
| Resource evidence | Snapshot exists, but it is not sufficient evidence to assume a full MI run will fit in RAM |

The MI implementation therefore emits `PROVISIONAL_REFERENCE_UNVERIFIED` and
keeps `official_reproduction=false`. The five CTU-13 reference scripts and
the current feature-selection modules still require an explicit, user-run
reconciliation before any output can be called an official reproduction.

## Implemented controls

### Contract and schema

- Requires exactly 380 ordered statistical features and the final label column
  `Window_Label`.
- Accepts the Stage 3 Train Parquet footer as a metadata-only schema binding
  when a sidecar `feature_schema.json` is not present in the configuration
  directory; the footer must still contain exactly 380 features followed by
  `Window_Label`.
- Rejects duplicate feature names and provenance-like fields.
- Binds every run to a schema/source identity digest and the Train footer
  metadata.
- Checks the expected Stage 3 Train row count for each configuration.

### Preprocessing

- Reads numeric Train feature values only.
- Converts to numeric, maps `NaN`, `+Inf`, and `-Inf` to zero, then stores
  `float32`.
- Does not drop rows, alter labels, scale features, or read Validation/Test
  values.

### MI and ranking

- Uses `sklearn.feature_selection.mutual_info_classif`.
- Uses `discrete_features=False`, `n_neighbors=3`, and a documented base seed
  of `42` with a deterministic per-original-feature seed.
- Produces scores for all 380 features.
- Sorts by descending MI score, then original feature order for exact ties.
- Produces a Top-100 artifact; no validation accuracy, RFE, or k-search is
  performed in Stage 4.

### Safety and reproducibility

- Metadata and resource gates run before any Train values are loaded.
- A failed resource gate blocks MI without sampling, parameter reduction, or
  automatic retry.
- Run directories are exclusive. Final result artifacts are exclusive and
  cannot silently overwrite an earlier run.
- Checkpoints support resume after an interrupted batch.
- `MI_COMPLETE.json` is written last; incomplete or failed runs are not
  official results.
- `verify` validates result artifacts without fitting or reading dataset
  values.

## Files

- `mi_selection.py` — contract, preprocessing, MI, checkpoint, resume, and verification implementation.
- `mi_preflight.py` — metadata-only and resource-only gates.
- `configs/mi_contract.json` — machine-readable Stage 4 contract.
- `tests/test_mi_selection.py` — synthetic tests using 380-column fixtures.
- `reports/STAGE4_EXECUTION_GUIDE.md` — commands for the user to run.

## Verification status

The implementation and synthetic tests have been written, but they have not
been executed by this agent because the Stage 4 request explicitly reserves
Python execution, dataset access, resource checks, and MI execution for the
user's controlled environment. Stage 4 remains pending user-run verification.

No MI run is being claimed as complete in this report.

