[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Task,

    [string]$Experiment = "",
    [string]$Configuration = "",
    [string]$RunId = "",

    [Parameter(Mandatory = $true)]
    [string]$Command,

    [string]$OutputDirectory = "",
    [string]$ResourceInfo = "",
    [ValidateSet("NOT_ASSESSED", "PASS", "FAIL", "BLOCKED", "PROVISIONAL_RESEARCH", "UNKNOWN")]
    [string]$ScientificStatus = "NOT_ASSESSED",
    [string]$WorkingDirectory = (Get-Location).Path,
    [string]$LogPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Protect-LogText {
    param([AllowNull()][string]$Value)

    if ($null -eq $Value) {
        return ""
    }

    $result = $Value
    $patterns = @(
        @{ Pattern = '(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+'; Replacement = '$1[REDACTED]' },
        @{ Pattern = '(?i)((?:password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key)\s*[=:]\s*)(?:"[^"]*"|''[^'']*''|\S+)'; Replacement = '$1[REDACTED]' },
        @{ Pattern = '(?i)((?:--password|--token|--secret|--api-key)(?:=|\s+))\S+'; Replacement = '$1[REDACTED]' }
    )

    foreach ($item in $patterns) {
        $result = [regex]::Replace($result, $item.Pattern, $item.Replacement)
    }
    return $result
}

function Resolve-LogPath {
    param([string]$RequestedPath)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        return [System.IO.Path]::GetFullPath($RequestedPath)
    }

    $codeRoot = Split-Path -Parent $PSScriptRoot
    return Join-Path (Join-Path $codeRoot "logs") "ALL_EXECUTION_RESULTS.log"
}

function Resolve-ExistingWorkingDirectory {
    param([string]$RequestedPath)

    $resolved = [System.IO.Path]::GetFullPath($RequestedPath)
    if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
        throw "Working directory does not exist: $resolved"
    }
    return $resolved
}

function Get-ChildShellPath {
    if ($PSVersionTable.PSEdition -eq "Core") {
        return Join-Path $PSHOME "pwsh.exe"
    }
    return Join-Path $PSHOME "powershell.exe"
}

function Append-ExecutionRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Record
    )

    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = $null
    $writer = $null
    try {
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Append,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::ReadWrite
        )
        $writer = New-Object System.IO.StreamWriter($stream, $encoding)
        $writer.Write($Record)
        $writer.Flush()
    }
    finally {
        if ($null -ne $writer) { $writer.Dispose() }
        elseif ($null -ne $stream) { $stream.Dispose() }
    }
}

$logPath = Resolve-LogPath $LogPath
$workingDirectory = Resolve-ExistingWorkingDirectory $WorkingDirectory
$start = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$stdoutBuilder = New-Object System.Text.StringBuilder
$stderrBuilder = New-Object System.Text.StringBuilder
$exitCode = 1
$launchError = $null
$process = $null

$safeCommand = Protect-LogText $Command

try {
    $shellPath = Get-ChildShellPath
    if (-not (Test-Path -LiteralPath $shellPath -PathType Leaf)) {
        throw "PowerShell child executable was not found: $shellPath"
    }

    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($Command)
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $shellPath
    $startInfo.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encodedCommand"
    $startInfo.WorkingDirectory = $workingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo

    # Start child process.
    # Read stdout and stderr concurrently without PowerShell scriptblock
    # callbacks on background threads.
    if (-not $process.Start()) {
        throw "The child process did not start"
    }

    $stdoutTask = $process.StandardOutput.ReadLineAsync()
    $stderrTask = $process.StandardError.ReadLineAsync()

    $stdoutDone = $false
    $stderrDone = $false

    while (-not ($stdoutDone -and $stderrDone)) {
        $received = $false

        if (-not $stdoutDone -and $stdoutTask.IsCompleted) {
            $received = $true
            $line = $stdoutTask.GetAwaiter().GetResult()

            if ($null -eq $line) {
                $stdoutDone = $true
            }
            else {
                $line = Protect-LogText $line
                [void]$stdoutBuilder.AppendLine($line)
                Write-Host $line
                $stdoutTask = $process.StandardOutput.ReadLineAsync()
            }
        }

        if (-not $stderrDone -and $stderrTask.IsCompleted) {
            $received = $true
            $line = $stderrTask.GetAwaiter().GetResult()

            if ($null -eq $line) {
                $stderrDone = $true
            }
            else {
                $line = Protect-LogText $line
                [void]$stderrBuilder.AppendLine($line)
                Write-Host "[stderr] $line" -ForegroundColor Yellow
                $stderrTask = $process.StandardError.ReadLineAsync()
            }
        }

        if (-not $received) {
            Start-Sleep -Milliseconds 50
        }
    }

    $process.WaitForExit()
    $exitCode = $process.ExitCode
}
catch {
    $launchError = $_.Exception.Message
    $safeError = Protect-LogText $launchError
    [void]$stderrBuilder.AppendLine($safeError)
    Write-Warning "Child launch/execution error: $safeError"
    $exitCode = 1
}
finally {
    $stopwatch.Stop()
    if ($null -ne $process) {
        $process.Dispose()
    }
}

$end = Get-Date
$runtimeSeconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
$programStatus = if ($exitCode -eq 0) { "PROGRAM_SUCCEEDED" } else { "PROGRAM_FAILED" }
$outputValue = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { "NOT_PROVIDED" } else { [System.IO.Path]::GetFullPath($OutputDirectory) }
$resourceValue = if ([string]::IsNullOrWhiteSpace($ResourceInfo)) { "NOT_PROVIDED" } else { Protect-LogText $ResourceInfo }
$stdoutValue = Protect-LogText $stdoutBuilder.ToString().TrimEnd()
$stderrValue = Protect-LogText $stderrBuilder.ToString().TrimEnd()
if ([string]::IsNullOrWhiteSpace($stdoutValue)) { $stdoutValue = "(empty)" }
if ([string]::IsNullOrWhiteSpace($stderrValue)) { $stderrValue = "(empty)" }

$record = @"
=================================================
STAGE: $(Protect-LogText $Stage)
TASK: $(Protect-LogText $Task)
EXPERIMENT: $(Protect-LogText $Experiment)
CONFIG: $(Protect-LogText $Configuration)
RUN_ID: $(Protect-LogText $RunId)
START: $($start.ToUniversalTime().ToString("o"))
COMMAND: $safeCommand
WORKING_DIRECTORY: $(Protect-LogText $workingDirectory)
-------------------------------------------------
STDOUT:
$stdoutValue
-------------------------------------------------
STDERR:
$stderrValue
-------------------------------------------------
EXIT_CODE: $exitCode
PROGRAM_STATUS: $programStatus
SCIENTIFIC_STATUS: $(Protect-LogText $ScientificStatus)
RUNTIME_SECONDS: $runtimeSeconds
RESOURCE_INFO: $resourceValue
OUTPUT_DIRECTORY: $(Protect-LogText $outputValue)
WARNING: $(if ($null -eq $launchError) { "" } else { "Child process launch/execution error was recorded above." })
END: $($end.ToUniversalTime().ToString("o"))
=================================================
"@

$loggingSucceeded = $false
try {
    Append-ExecutionRecord -Path $logPath -Record $record
    $loggingSucceeded = $true
    Write-Host "Execution record appended: $logPath" -ForegroundColor Green
}
catch {
    Write-Warning "EXECUTION LOGGING FAILED. The original exit code is preserved. Log path: $logPath. Reason: $($_.Exception.Message)"
}

if (-not $loggingSucceeded) {
    Write-Warning "No claim is made that this execution was saved to the aggregate log."
}

exit $exitCode

