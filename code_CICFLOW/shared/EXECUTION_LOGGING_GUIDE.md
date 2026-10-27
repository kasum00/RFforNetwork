# CICFLOW execution logging

All user-run CICFLOW commands should be launched through:

`D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1`

The default aggregate log is:

`D:\COMPACT-MI-RFE\code_CICFLOW\logs\ALL_EXECUTION_RESULTS.log`

The wrapper appends a structured record, displays child stdout/stderr live,
and returns the child exit code. It never infers a scientific PASS from an
exit code. Supply `-ScientificStatus` only after inspecting the scientific
output; otherwise it remains `NOT_ASSESSED`.

The wrapper performs best-effort redaction of common password/token/secret
patterns. Do not put secrets in commands or output. If the log cannot be
created or appended, the wrapper emits a warning and preserves the child exit
code; it does not claim that the result was saved.

## Wrapper smoke test

The smoke-test harness is provided but is not run by Codex:

```powershell
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\tests\Test-Invoke-CicflowLogged.ps1'
```

## Manual invocation pattern

```powershell
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' `
  -Stage '4 — Mutual Information' `
  -Task 'metadata preflight' `
  -Experiment 'CICFLOW_COMPACT_MI_RFE' `
  -Configuration 'FW256_FS8' `
  -RunId 'RUN_MI_20261027_01' `
  -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' `
  -Command "python -m stage4_mi.mi_preflight --mode metadata --config FW256_FS8 --dataset-root 'D:\COMPACT-MI-RFE\CICFLOWMETER_FEATURE_CSV'" `
  -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs\FW256_FS8\RUN_MI_20261027_01' `
  -ScientificStatus NOT_ASSESSED
```

