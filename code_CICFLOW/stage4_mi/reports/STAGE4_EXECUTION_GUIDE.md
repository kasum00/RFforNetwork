# Stage 4 — Execution Guide with aggregate logging

All executable commands in this guide must be launched through
`D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1`.
The wrapper displays stdout/stderr live and appends the structured result to
`D:\COMPACT-MI-RFE\code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log`. It preserves
the child exit code and does not infer a scientific PASS from that code.

Define the shared paths first:

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
$wrapper = 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1'
$datasetRoot = 'D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV'
$runsRoot = 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs'
$runId = 'RUN_MI_20261027_01'
```

Use the logged command forms below for every execution. The older direct
`python ...` examples later in this file are retained as reference syntax;
do not execute them directly.

## Logged command forms

### Unit tests

```powershell
& $wrapper -Stage '4' -Task 'Stage 4 synthetic unit tests' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'SYNTHETIC' -RunId 'TEST_STAGE4_MI' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command "python -m unittest discover -s 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\tests' -p 'test_*.py' -v" -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\tests' -ScientificStatus NOT_ASSESSED
```

### Metadata and resource preflight

```powershell
& $wrapper -Stage '4' -Task 'metadata preflight' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'FW256_FS8' -RunId $runId -Command "python -m stage4_mi.mi_preflight --mode metadata --config FW256_FS8 --dataset-root '$datasetRoot' --runs-root '$runsRoot' --run-id '$runId'" -OutputDirectory "$runsRoot\FW256_FS8\$runId" -ScientificStatus NOT_ASSESSED
& $wrapper -Stage '4' -Task 'resource preflight' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'FW256_FS8' -RunId $runId -Command "python -m stage4_mi.mi_preflight --mode resource --config FW256_FS8 --dataset-root '$datasetRoot'" -OutputDirectory "$runsRoot\FW256_FS8\$runId" -ScientificStatus NOT_ASSESSED
```

Repeat the same logged forms for `FW256_FS16`, `FW512_FS8`, and `FW512_FS16`.

### Full MI, resume, and verify

```powershell
& $wrapper -Stage '4' -Task 'MI full run' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'FW256_FS8' -RunId $runId -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command "python -m stage4_mi.mi_selection --mode full --config FW256_FS8 --dataset-root '$datasetRoot' --runs-root '$runsRoot' --run-id '$runId' --confirm-provisional-research" -OutputDirectory "$runsRoot\FW256_FS8\$runId" -ScientificStatus NOT_ASSESSED

& $wrapper -Stage '4' -Task 'MI resume' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'FW256_FS8' -RunId $runId -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command "python -m stage4_mi.mi_selection --mode resume --config FW256_FS8 --dataset-root '$datasetRoot' --runs-root '$runsRoot' --run-id '$runId' --confirm-provisional-research" -OutputDirectory "$runsRoot\FW256_FS8\$runId" -ScientificStatus NOT_ASSESSED

& $wrapper -Stage '4' -Task 'MI artifact verification' -Experiment 'CICFLOW_COMPACT_MI_RFE' -Configuration 'FW256_FS8' -RunId $runId -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -Command "python -m stage4_mi.mi_selection --mode verify --run-dir '$runsRoot\FW256_FS8\$runId'" -OutputDirectory "$runsRoot\FW256_FS8\$runId" -ScientificStatus NOT_ASSESSED
```

If a resource gate is blocked, stop. Do not sample, reduce MI parameters, or
retry automatically. Do not execute direct command examples later in this
file; use the wrapper forms above.

Run these commands in a controlled PowerShell session. The commands below are
instructions only; they were not executed while preparing this implementation.

## 1. Enter the project and expose the package

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
```

## 2. Run synthetic tests

```powershell
python -m unittest discover `
  -s 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\tests' `
  -p 'test_*.py' -v
```

These tests must use only synthetic data and must not scan the CICFLOW
dataset.

## 3. Run metadata-only preflight for every configuration

Use a fresh run ID. Do not reuse an existing run directory.

```powershell
$datasetRoot = 'D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV'
$runsRoot = 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs'
$runId = 'RUN_MI_20261027_01'

python -m stage4_mi.mi_preflight --mode metadata --config FW256_FS8 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId
python -m stage4_mi.mi_preflight --mode metadata --config FW256_FS16 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId
python -m stage4_mi.mi_preflight --mode metadata --config FW512_FS8 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId
python -m stage4_mi.mi_preflight --mode metadata --config FW512_FS16 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId
```

Confirm that each result reports `"status": "PASS"`. A metadata failure
blocks MI for that configuration.

The preflight accepts either a sidecar schema JSON or the ordered schema in
the `train_windows.parquet` footer. The fallback is metadata-only and does
not read Train feature values. The implementation accepts the known Train
filenames `train_windows.parquet`, `train.parquet`, or
`train_features.parquet`; it does not search or open Validation/Test files.

## 4. Run the resource gate immediately before MI

```powershell
python -m stage4_mi.mi_preflight --mode resource --config FW256_FS8 `
  --dataset-root $datasetRoot
python -m stage4_mi.mi_preflight --mode resource --config FW256_FS16 `
  --dataset-root $datasetRoot
python -m stage4_mi.mi_preflight --mode resource --config FW512_FS8 `
  --dataset-root $datasetRoot
python -m stage4_mi.mi_preflight --mode resource --config FW512_FS16 `
  --dataset-root $datasetRoot
```

If a resource gate is `BLOCKED`, stop for that configuration. Do not sample,
reduce MI parameters, or automatically retry.

## 5. Run one full MI configuration at a time

The explicit confirmation flag records that these are provisional research
outputs, not official reproduction results.

```powershell
python -m stage4_mi.mi_selection --mode full --config FW256_FS8 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId `
  --confirm-provisional-research
```

After a successful run, repeat with a new run ID for the next configuration:

```powershell
python -m stage4_mi.mi_selection --mode full --config FW256_FS16 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id 'RUN_MI_20261027_02' `
  --confirm-provisional-research
python -m stage4_mi.mi_selection --mode full --config FW512_FS8 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id 'RUN_MI_20261027_03' `
  --confirm-provisional-research
python -m stage4_mi.mi_selection --mode full --config FW512_FS16 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id 'RUN_MI_20261027_04' `
  --confirm-provisional-research
```

The full path reads only `train_windows.parquet` values. It must not be
started until the corresponding metadata and resource gates pass.

## 6. Resume an interrupted run

Resume the exact run directory after checking that its source identity and
scientific parameters are unchanged:

```powershell
python -m stage4_mi.mi_selection --mode resume --config FW256_FS8 `
  --dataset-root $datasetRoot --runs-root $runsRoot --run-id $runId `
  --confirm-provisional-research
```

Do not delete `RUN_INCOMPLETE.json` or `mi_progress.json` to force progress.

## 7. Verify without fitting

```powershell
python -m stage4_mi.mi_selection --mode verify `
  --run-dir 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs\FW256_FS8\RUN_MI_20261027_01'
```

Expected final artifacts are `mi_scores.csv`, `mi_ranking.json`,
`top100_features.json`, `mi_manifest.json`, `resource_usage.json`,
`mi_report.md`, and the last-written `MI_COMPLETE.json`.

## 8. Review and report

Record the exact command output, runtime, resource snapshot, run identity,
and any blocked/failed reason. Keep the provisional label until the CTU-13
reference reconciliation is complete. Do not promote an MI output to an
official result solely because the technical run passed.

