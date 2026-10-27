# Google Colab guide — CICFLOW MI-RFE

## Windows → Google Drive → Colab path map

| Role | Windows | Google Drive / Colab |
|---|---|---|
| Drive root | — | `/content/drive/MyDrive` |
| Source root | `D:\COMPACT-MI-RFE\code_CICFLOW` | `/content/drive/MyDrive/code_CICFLOW` |
| Dataset root | `D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV` | `/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV` |
| Fresh result root | new runtime folder | `/content/drive/MyDrive/COMPACT_MI_RFE_FRESH_RUNS/RUN_MI_<UTC_TIMESTAMP>` |
| Stage 4 runs | `stage4_mi\runs` | `COMPACT_MI_RFE_FRESH_RUNS/RUN_MI_<...>/stage4_mi/runs` |
| Stage 5A runs | `stage5_rfe\runs` | `COMPACT_MI_RFE_FRESH_RUNS/RUN_MI_<...>/stage5_rfe/runs` |
| Aggregate execution log | `code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log` | `/content/drive/MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log` |
| Small-artifact backup | new Drive directory | `COMPACT_MI_RFE_BACKUPS` |

The reference notebook remains:

`notebooks/COMPACT_MI_RFE_ColabPro.ipynb`

The new notebook is:

`notebooks/COMPACT_MI_RFE_CICFLOW_Colab.ipynb`

## Prepare source and data on Windows

Run these commands yourself. They are preparation examples only and were not
executed by Codex. Do not package the entire dataset if only Train is needed.

### Source package

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
if (Test-Path -LiteralPath 'D:\COMPACT-MI-RFE\COMPACT_MI_RFE_CICFLOW_COLAB_SOURCE.zip') { throw 'Refusing to overwrite an existing source package' }
Compress-Archive `
  -Path 'code_CICFLOW', 'notebooks\COMPACT_MI_RFE_CICFLOW_Colab.ipynb' `
  -DestinationPath 'D:\COMPACT-MI-RFE\COMPACT_MI_RFE_CICFLOW_COLAB_SOURCE.zip'
```

Do not include the reference notebook or CTU-13 scripts in a modified archive.

### Train-only data package

The exact Train filename must be confirmed from the Stage 3 metadata first.
Copy only the selected Train Parquet and required schema/manifest metadata for
each configuration. Do not copy Validation/Test Parquet for MI/RFE.

```powershell
$source = 'D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV'
$package = 'D:\COMPACT-MI-RFE\COLAB_INPUT_TRAIN_ONLY'

New-Item -ItemType Directory -Force -Path $package | Out-Null
foreach ($config in @('FW256_FS8','FW256_FS16','FW512_FS8','FW512_FS16')) {
  New-Item -ItemType Directory -Force -Path (Join-Path $package $config) | Out-Null
  Copy-Item -LiteralPath (Join-Path $source "$config\train_windows.parquet") -Destination (Join-Path $package "$config\train_windows.parquet")
}
```

If the actual Train filename is `train.parquet` or
`train_features.parquet`, substitute that exact filename. Do not use a wildcard
that could copy Validation/Test files.

## Run the new notebook

1. Upload/open `COMPACT_MI_RFE_CICFLOW_Colab.ipynb`.
2. Mount Drive. The notebook expects source at `/content/drive/MyDrive/code_CICFLOW`.
3. Confirm `SOURCE_ROOT`, `DATASET_ROOT`, and the four Run IDs.
4. Run the path/layout and metadata cells in `MODE='dry_run'`.
5. Review every configuration's schema, Train row count, resource gate, and
   Stage 4 identity before changing mode.

### Dry run

Keep:

```python
MODE = 'dry_run'
REFERENCE_RECONCILIATION_CONFIRMED = False
CONFIRM_PROVISIONAL_RESEARCH = False
```

Run All is safe in this mode: MI/RFE fitting is skipped.

### Full

Use new Run IDs if the listed Stage 4 run already exists. Then set:

```python
MODE = 'full'
REFERENCE_RECONCILIATION_CONFIRMED = True
CONFIRM_PROVISIONAL_RESEARCH = True
```

This starts MI first and RFE only after MI succeeds. It remains provisional.

### Resume

Set:

```python
MODE = 'resume'
REFERENCE_RECONCILIATION_CONFIRMED = True
CONFIRM_PROVISIONAL_RESEARCH = True
```

Also set `EXISTING_RESULT_ROOT` to the existing fresh result directory, for
example:

```python
EXISTING_RESULT_ROOT = '/content/drive/MyDrive/COMPACT_MI_RFE_FRESH_RUNS/RUN_MI_20261027T154149_761986Z'
```

Resume is allowed only for identity-matching incomplete runs. A COMPLETE run
is never resumed or overwritten. `full` creates a new fresh result root;
`resume` and `verify` never create a new one.

### Verify

Set:

```python
MODE = 'verify'
```

Verify reads artifacts/markers only and does not fit MI/RFE or read Test.

## Unit tests on Windows

Use the shared logging wrapper and do not run tests from an arbitrary working
directory:

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' `
  -Stage 'Colab' `
  -Task 'Colab orchestration unit tests' `
  -Experiment 'CICFLOW_COMPACT_MI_RFE' `
  -Configuration 'SYNTHETIC' `
  -RunId 'TEST_COLAB_PIPELINE' `
  -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
  -Command "python -m unittest discover -s 'D:\COMPACT-MI-RFE\code_CICFLOW\colab\tests' -p 'test_*.py' -v" `
  -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\colab\tests' `
  -ScientificStatus NOT_ASSESSED
```

The Stage 4 and Stage 5A unit-test commands remain in their respective
execution guides.

## Checkpoint and report locations

Stage 4:

`/content/drive/MyDrive/COMPACT-MI-RFE/code_CICFLOW/stage4_mi/runs/<CONFIG>/<RUN_ID>/`

Stage 5A:

`/content/drive/MyDrive/COMPACT-MI-RFE/code_CICFLOW/stage5_rfe/runs/<CONFIG>/<RUN_ID>/<BRANCH>/`

The notebook summary reports marker status for all four configurations and both
RF/SVM branches. Send back the cell output, manifests, resource reports,
checkpoint status, and any BLOCKED/FAILED marker; do not infer scientific PASS
from a zero process exit code.

