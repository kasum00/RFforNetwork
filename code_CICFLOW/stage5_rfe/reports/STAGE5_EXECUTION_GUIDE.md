# Stage 5A — Execution Guide

All executable commands below use the shared wrapper. The wrapper keeps output
visible and appends the record to:

`D:\COMPACT-MI-RFE\code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log`

No command in this guide was executed while preparing the implementation.
Leave `-ScientificStatus NOT_ASSESSED` until the program output has been
reviewed scientifically.

## 1. Define paths

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'

$wrapper = 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1'
$datasetRoot = 'D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV'
$miRunsRoot = 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs'
$stage5RunsRoot = 'D:\COMPACT-MI-RFE\code_CICFLOW\stage5_rfe\runs'
$gitCommit = 'NOT_CAPTURED'

$configs = @('FW256_FS8', 'FW256_FS16', 'FW512_FS8', 'FW512_FS16')
$runIds = @{
  FW256_FS8  = 'RUN_MI_20261027_01'
  FW256_FS16 = 'RUN_MI_20261027_02'
  FW512_FS8  = 'RUN_MI_20261027_03'
  FW512_FS16 = 'RUN_MI_20261027_04'
}
```

If a verified commit identifier is available, replace `NOT_CAPTURED` with it;
do not let the code execute Git automatically.

## 2. Synthetic Stage 5A unit tests

```powershell
& $wrapper `
  -Stage '5A' `
  -Task 'Stage 5A synthetic unit tests' `
  -Experiment 'CICFLOW_COMPACT_MI_RFE' `
  -Configuration 'SYNTHETIC' `
  -RunId 'TEST_STAGE5_RFE' `
  -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
  -Command "python -m unittest discover -s 'D:\COMPACT-MI-RFE\code_CICFLOW\stage5_rfe\tests' -p 'test_*.py' -v" `
  -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\stage5_rfe\tests' `
  -ScientificStatus NOT_ASSESSED
```

## 3. Metadata preflight for both branches and four configurations

```powershell
foreach ($config in $configs) {
  $runId = $runIds[$config]
  foreach ($branch in @('rf', 'svm')) {
    & $wrapper `
      -Stage '5A' `
      -Task 'metadata preflight' `
      -Experiment 'CICFLOW_COMPACT_MI_RFE' `
      -Configuration $config `
      -RunId $runId `
      -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
      -Command "python -m stage5_rfe.rfe_preflight --mode metadata --branch $branch --config $config --dataset-root '$datasetRoot' --mi-runs-root '$miRunsRoot' --runs-root '$stage5RunsRoot' --run-id '$runId'" `
      -OutputDirectory "$stage5RunsRoot\$config\$runId\$branch" `
      -ScientificStatus NOT_ASSESSED
  }
}
```

`metadata` must pass before resource preflight or full RFE. It is expected to
refuse an already existing Stage 5 branch run directory rather than overwrite
it.

## 4. Resource preflight

```powershell
foreach ($config in $configs) {
  $runId = $runIds[$config]
  foreach ($branch in @('rf', 'svm')) {
    & $wrapper `
      -Stage '5A' `
      -Task 'resource preflight' `
      -Experiment 'CICFLOW_COMPACT_MI_RFE' `
      -Configuration $config `
      -RunId $runId `
      -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
      -Command "python -m stage5_rfe.rfe_preflight --mode resource --branch $branch --config $config --dataset-root '$datasetRoot' --mi-runs-root '$miRunsRoot' --runs-root '$stage5RunsRoot' --run-id '$runId'" `
      -OutputDirectory "$stage5RunsRoot\$config\$runId\$branch" `
      -ScientificStatus NOT_ASSESSED
  }
}
```

If a resource gate is `BLOCKED`, stop. Do not reduce rows, trees, features,
or RFE step, and do not retry automatically.

## 5. Full Stage 5A RF-RFE

Run only after the corresponding metadata and resource gates pass.

```powershell
foreach ($config in $configs) {
  $runId = $runIds[$config]
  & $wrapper `
    -Stage '5A' `
    -Task 'RF-RFE full run' `
    -Experiment 'CICFLOW_COMPACT_MI_RFE' `
    -Configuration $config `
    -RunId $runId `
    -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
    -Command "python -m stage5_rfe.rfe_selection --mode full --branch rf --config $config --dataset-root '$datasetRoot' --mi-runs-root '$miRunsRoot' --runs-root '$stage5RunsRoot' --run-id '$runId' --git-commit '$gitCommit' --confirm-provisional-research" `
    -OutputDirectory "$stage5RunsRoot\$config\$runId\rf" `
    -ScientificStatus NOT_ASSESSED
}
```

## 6. Full Stage 5A SVM-RFE branch

The branch label is `svm`; its RFE estimator is the specified RandomForest
with `max_depth=15`. Its unresolved parameters remain explicitly provisional.

```powershell
foreach ($config in $configs) {
  $runId = $runIds[$config]
  & $wrapper `
    -Stage '5A' `
    -Task 'SVM-RFE full run' `
    -Experiment 'CICFLOW_COMPACT_MI_RFE' `
    -Configuration $config `
    -RunId $runId `
    -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
    -Command "python -m stage5_rfe.rfe_selection --mode full --branch svm --config $config --dataset-root '$datasetRoot' --mi-runs-root '$miRunsRoot' --runs-root '$stage5RunsRoot' --run-id '$runId' --git-commit '$gitCommit' --confirm-provisional-research" `
    -OutputDirectory "$stage5RunsRoot\$config\$runId\svm" `
    -ScientificStatus NOT_ASSESSED
}
```

## 7. Resume an interrupted branch

Resume only an incomplete branch with matching identity. Do not resume a
branch containing `COMPLETE.json`.

```powershell
$config = 'FW256_FS8'
$runId = $runIds[$config]
& $wrapper `
  -Stage '5A' `
  -Task 'RF-RFE resume' `
  -Experiment 'CICFLOW_COMPACT_MI_RFE' `
  -Configuration $config `
  -RunId $runId `
  -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
  -Command "python -m stage5_rfe.rfe_selection --mode resume --branch rf --config $config --dataset-root '$datasetRoot' --mi-runs-root '$miRunsRoot' --runs-root '$stage5RunsRoot' --run-id '$runId' --git-commit '$gitCommit' --confirm-provisional-research" `
  -OutputDirectory "$stage5RunsRoot\$config\$runId\rf" `
  -ScientificStatus NOT_ASSESSED
```

Change `--branch rf` and the wrapper task when resuming the SVM branch.

## 8. Verify branch artifacts

```powershell
foreach ($config in $configs) {
  $runId = $runIds[$config]
  foreach ($branch in @('rf', 'svm')) {
    & $wrapper `
      -Stage '5A' `
      -Task 'RFE artifact verification' `
      -Experiment 'CICFLOW_COMPACT_MI_RFE' `
      -Configuration $config `
      -RunId $runId `
      -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
      -Command "python -m stage5_rfe.rfe_selection --mode verify --run-dir '$stage5RunsRoot\$config\$runId\$branch'" `
      -OutputDirectory "$stage5RunsRoot\$config\$runId\$branch" `
      -ScientificStatus NOT_ASSESSED
  }
}
```

## 9. XGBoost

Do not run an XGBoost full branch. Its Stage 5A ranking contract is
`BLOCKED` until the estimator and source parameters are reconciled.

## 10. Stage 5B

Do not run k-search, final Train+Validation refit, or Test evaluation in this
task. Follow [STAGE5B_CONTRACT.md](STAGE5B_CONTRACT.md) only after Stage 5A
ranking is scientifically locked.

