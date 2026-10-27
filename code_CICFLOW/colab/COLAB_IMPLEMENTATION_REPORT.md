# CICFLOW MI-RFE Colab Implementation Report

## Created artifacts

- `D:\COMPACT-MI-RFE\notebooks\COMPACT_MI_RFE_CICFLOW_Colab.ipynb`
- `D:\COMPACT-MI-RFE\code_CICFLOW\colab\colab_pipeline.py`
- `D:\COMPACT-MI-RFE\code_CICFLOW\colab\tests\test_colab_pipeline.py`
- `D:\COMPACT-MI-RFE\code_CICFLOW\colab\COLAB_GUIDE.md`

The reference notebook `notebooks/COMPACT_MI_RFE_ColabPro.ipynb` was not
overwritten. The CTU-13 source scripts and Stage 0–5 artifacts were not
modified.

The Colab path contract is now aligned with the reference-style Drive layout:

- `DRIVE_ROOT=/content/drive/MyDrive`
- `SOURCE_ROOT=/content/drive/MyDrive/code_CICFLOW`
- `DATASET_ROOT=/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV`
- Fresh outputs under `COMPACT_MI_RFE_FRESH_RUNS/RUN_MI_<UTC_TIMESTAMP>`
- Stage 4 and Stage 5 run roots are created inside that fresh result root.

## Reference cell-purpose alignment

The reference notebook was read and its operational sequence was mapped into
the new CICFLOW notebook:

| Reference purpose | CICFLOW notebook cell |
|---|---|
| Runtime/resource information | Cell 1 |
| Google Drive mount and paths | Cell 2 |
| Dataset source resolution | Cell 3 |
| Safe source checkout | Cell 4 |
| Dependencies and tests | Cell 5 |
| Dataset/schema audit | Cell 6 |
| Dataset readiness gate | Cell 7 |
| Experiment/config selection | Cell 8 |
| Safe preflight | Cell 9 |
| Isolated MI/RFE run | Cell 10 |
| Artifact verification | Cell 11 |
| Summary and small-artifact backup | Cell 12 |

The legacy scaffold cells remain below the authoritative CICFLOW sequence for
traceability and are marked `LEGACY OLD ORCHESTRATOR CELLS — DO NOT RUN`.

## Operating modes

The new notebook defaults to `MODE = 'dry_run'`. The modes are:

- `dry_run`: path, root, schema/footer/resource, and contract checks; no MI or
  RFE fit.
- `full`: Stage 4 MI followed by Stage 5A RF/SVM-RFE, only after explicit
  provisional and reference-reconciliation flags are enabled.
- `resume`: resume only identity-matching incomplete runs.
- `verify`: verify existing Stage 4/5 markers and manifests without fitting.

Run IDs are configured explicitly for all four configurations. Existing
completed Windows MI runs are not overwritten; a new Run ID is required for a
new Colab MI variant.

## Scientific status

The notebook preserves `PROVISIONAL_REFERENCE_UNVERIFIED`. It does not claim
that the Windows MI per-feature implementation is CTU-13 reference-equivalent.
Stage 5A uses the separate RF/SVM branches already contracted in
`code_CICFLOW/stage5_rfe`. XGBoost remains blocked.

## Data protection

- Dataset root mixing is rejected.
- MI/RFE read only Train values.
- Validation/Test are not opened by the Stage 4/5A orchestration cells.
- Drive backup copies only small metadata/result artifacts and writes a backup
  marker last. It never copies Parquet files.
- Existing COMPLETE runs and checkpoints are protected from overwrite.

## Execution status

No notebook, Python, test, MI, RFE, or training execution was performed while
creating these artifacts.

