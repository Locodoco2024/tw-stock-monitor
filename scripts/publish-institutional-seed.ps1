param(
    [string]$StateDir = "runtime/institutional",
    [string]$Remote = "origin",
    [string]$Branch = "state"
)

$ErrorActionPreference = "Stop"

function Invoke-GitCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$Quiet,
        [switch]$AllowFailure
    )

    # Windows PowerShell 5.1 converts normal native stderr output from Git
    # into ErrorRecord objects. With ErrorActionPreference=Stop, messages such
    # as "From https://..." can incorrectly terminate the script. Temporarily
    # use Continue and determine success only from Git's process exit code.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $commandOutput = @(& git @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if (-not $Quiet) {
        foreach ($line in $commandOutput) {
            Write-Host ($line.ToString())
        }
    }

    if (-not $AllowFailure -and $exitCode -ne 0) {
        $displayCommand = "git " + ($Arguments -join " ")
        throw "$displayCommand failed with exit code $exitCode."
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $commandOutput
    }
}

$resolvedState = (Resolve-Path -LiteralPath $StateDir).Path
$required = @(
    "rolling_market_data.csv.gz",
    "universe.csv",
    "score_history.csv.gz",
    "latest_scores.csv.gz",
    "notification_plan.csv",
    "update_manifest.json"
)

foreach ($file in $required) {
    $requiredPath = Join-Path $resolvedState $file
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Missing seed file: $file. Run seed-institutional-deployment.ps1 first."
    }
}

$repoResult = Invoke-GitCommand -Arguments @("rev-parse", "--show-toplevel") -Quiet
$repoRoot = (($repoResult.Output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
if (-not $repoRoot) {
    throw "The current directory is not a Git repository."
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tw-stock-state-" + [guid]::NewGuid())

try {
    $fetchResult = Invoke-GitCommand `
        -Arguments @("fetch", $Remote, $Branch) `
        -Quiet `
        -AllowFailure
    $remoteExists = ($fetchResult.ExitCode -eq 0)

    if ($remoteExists) {
        Invoke-GitCommand -Arguments @(
            "worktree", "add", "--force", "-B", $Branch, $tempRoot, "$Remote/$Branch"
        ) | Out-Null
    }
    else {
        Invoke-GitCommand -Arguments @("worktree", "add", "--detach", $tempRoot, "HEAD") |
            Out-Null
        Invoke-GitCommand -Arguments @("-C", $tempRoot, "switch", "--orphan", $Branch) |
            Out-Null
        Invoke-GitCommand `
            -Arguments @("-C", $tempRoot, "rm", "-rf", ".") `
            -Quiet `
            -AllowFailure | Out-Null
    }

    $institutionalTarget = Join-Path $tempRoot "institutional"
    if (Test-Path -LiteralPath $institutionalTarget) {
        Remove-Item -LiteralPath $institutionalTarget -Recurse -Force
    }
    Copy-Item -LiteralPath $resolvedState -Destination $institutionalTarget -Recurse -Force

    $localState = Join-Path $repoRoot "runtime/state.json"
    $targetState = Join-Path $tempRoot "state.json"
    if (Test-Path -LiteralPath $localState) {
        Copy-Item -LiteralPath $localState -Destination $targetState -Force
    }
    elseif (-not (Test-Path -LiteralPath $targetState)) {
        '{"records":{},"institutional_notification_keys":{}}' |
            Set-Content -LiteralPath $targetState -Encoding UTF8
    }

    Invoke-GitCommand -Arguments @("-C", $tempRoot, "add", "state.json", "institutional") |
        Out-Null
    Invoke-GitCommand -Arguments @(
        "-C", $tempRoot, "config", "user.name", "phase5h-seed"
    ) -Quiet | Out-Null
    Invoke-GitCommand -Arguments @(
        "-C", $tempRoot, "config", "user.email", "phase5h-seed@users.noreply.github.com"
    ) -Quiet | Out-Null

    $diffResult = Invoke-GitCommand `
        -Arguments @("-C", $tempRoot, "diff", "--cached", "--quiet") `
        -Quiet `
        -AllowFailure

    if ($diffResult.ExitCode -eq 1) {
        Invoke-GitCommand -Arguments @(
            "-C", $tempRoot, "commit", "-m", "Publish Phase 5H institutional seed [skip ci]"
        ) | Out-Null
    }
    elseif ($diffResult.ExitCode -eq 0) {
        Write-Host "Seed files are unchanged. Commit skipped."
    }
    else {
        throw "Failed to inspect staged state changes."
    }

    Invoke-GitCommand -Arguments @(
        "-C", $tempRoot, "push", $Remote, "HEAD:$Branch"
    ) | Out-Null

    Write-Host "Institutional seed published to $Remote/$Branch."
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Invoke-GitCommand `
            -Arguments @("worktree", "remove", "--force", $tempRoot) `
            -Quiet `
            -AllowFailure | Out-Null
        if (Test-Path -LiteralPath $tempRoot) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force
        }
    }
}
