[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sharedRoot = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $sharedRoot "Invoke-CicflowLogged.ps1"
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cicflow_logging_test_" + [guid]::NewGuid().ToString("N"))
$logPath = Join-Path $tempRoot "ALL_EXECUTION_RESULTS.log"
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "ASSERTION FAILED: $Message" }
}

try {
    & $wrapper `
        -Stage "TEST" `
        -Task "success" `
        -Configuration "SYNTHETIC" `
        -RunId "TEST_SUCCESS" `
        -Command "Write-Output 'stdout-success'" `
        -OutputDirectory $tempRoot `
        -LogPath $logPath
    $successCode = $LASTEXITCODE
    Assert-True ($successCode -eq 0) "successful child exit code was not preserved"

    & $wrapper `
        -Stage "TEST" `
        -Task "stderr-and-failure" `
        -Configuration "SYNTHETIC" `
        -RunId "TEST_FAILURE" `
        -Command "[Console]::Error.WriteLine('stderr-failure'); Write-Output 'stdout-failure'; exit 7" `
        -OutputDirectory $tempRoot `
        -LogPath $logPath
    $failureCode = $LASTEXITCODE
    Assert-True ($failureCode -eq 7) "failing child exit code was not preserved"

    $content = Get-Content -LiteralPath $logPath -Raw
    Assert-True ($content.Contains("RUN_ID: TEST_SUCCESS")) "first record is missing"
    Assert-True ($content.Contains("RUN_ID: TEST_FAILURE")) "append record is missing"
    Assert-True ($content.Contains("stdout-failure")) "stdout was not captured"
    Assert-True ($content.Contains("stderr-failure")) "stderr was not captured"
    Assert-True (($content -split "=================================================").Count -ge 3) "history was overwritten"

    $lockStream = [System.IO.File]::Open(
        $logPath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::None
    )
    try {
        & $wrapper `
            -Stage "TEST" `
            -Task "locked-log" `
            -Configuration "SYNTHETIC" `
            -RunId "TEST_LOCKED_LOG" `
            -Command "Write-Output 'log-lock-check'" `
            -OutputDirectory $tempRoot `
            -LogPath $logPath
        $lockedCode = $LASTEXITCODE
        Assert-True ($lockedCode -eq 0) "log failure changed the child exit code"
    }
    finally {
        $lockStream.Dispose()
    }

    Write-Output "PASS: wrapper success, stderr, append, and exit-code preservation checks"
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

