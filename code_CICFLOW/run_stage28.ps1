# Stage 2.8 launcher: estimate by default, explicit full metadata dry-run.
param(
    [string]$Repository = 'D:\COMPACT-MI-RFE',
    [string]$OutputRoot = 'D:\COMPACT-MI-RFE\STAGE28_CPLUS_PLAN',
    [string]$PythonExe = '',
    [switch]$Run,
    [switch]$AcceptDiskEstimate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'stage28_cplus_dryrun.py'
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Missing Stage 2.8 Python code: $scriptPath"
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidate = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $candidate) {
        throw 'Python not found; supply -PythonExe with an absolute python.exe path.'
    }
    $PythonExe = $candidate.Source
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if ($Run -and -not $AcceptDiskEstimate) {
    throw 'Full dry-run requires -Run -AcceptDiskEstimate.'
}
$arguments = @($scriptPath, '--repository', $Repository, '--output-root', $OutputRoot)
if ($Run) { $arguments += @('--execute', '--accept-disk-estimate') }
& $PythonExe @arguments
exit $LASTEXITCODE
