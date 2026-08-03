param(
    [string]$StateDir = "runtime/institutional_twse",
    [string]$Remote = "origin",
    [string]$Branch = "state"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

function Invoke-GitCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$Quiet,
        [switch]$AllowFailure
    )
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
        throw (("git " + ($Arguments -join " ")) + " failed with exit code $exitCode.")
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $commandOutput }
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
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedState $file))) {
        throw "Missing TWSE seed file: $file. Run seed-twse-institutional-deployment.ps1 first."
    }
}

$repoResult = Invoke-GitCommand -Arguments @("rev-parse", "--show-toplevel") -Quiet
$repoRoot = (($repoResult.Output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
if (-not $repoRoot) { throw "The current directory is not a Git repository." }
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tw-stock-twse-state-" + [guid]::NewGuid())

try {
    $fetch = Invoke-GitCommand -Arguments @("fetch", $Remote, $Branch) -Quiet -AllowFailure
    if ($fetch.ExitCode -eq 0) {
        Invoke-GitCommand -Arguments @("worktree", "add", "--force", "-B", $Branch, $tempRoot, "$Remote/$Branch") | Out-Null
    }
    else {
        Invoke-GitCommand -Arguments @("worktree", "add", "--detach", $tempRoot, "HEAD") | Out-Null
        Invoke-GitCommand -Arguments @("-C", $tempRoot, "switch", "--orphan", $Branch) | Out-Null
        Invoke-GitCommand -Arguments @("-C", $tempRoot, "rm", "-rf", ".") -Quiet -AllowFailure | Out-Null
    }

    $target = Join-Path $tempRoot "institutional_twse"
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    Copy-Item -LiteralPath $resolvedState -Destination $target -Recurse -Force

    $localState = Join-Path $repoRoot "runtime/state.json"
    $targetState = Join-Path $tempRoot "state.json"
    if (Test-Path -LiteralPath $localState) {
        Copy-Item -LiteralPath $localState -Destination $targetState -Force
    }
    elseif (-not (Test-Path -LiteralPath $targetState)) {
        '{"records":{},"institutional_notification_keys":{}}' | Set-Content -LiteralPath $targetState -Encoding UTF8
    }

    Invoke-GitCommand -Arguments @("-C", $tempRoot, "add", "state.json", "institutional_twse") | Out-Null
    Invoke-GitCommand -Arguments @("-C", $tempRoot, "config", "user.name", "phase6d-seed") -Quiet | Out-Null
    Invoke-GitCommand -Arguments @("-C", $tempRoot, "config", "user.email", "phase6d-seed@users.noreply.github.com") -Quiet | Out-Null
    $diff = Invoke-GitCommand -Arguments @("-C", $tempRoot, "diff", "--cached", "--quiet") -Quiet -AllowFailure
    if ($diff.ExitCode -eq 1) {
        Invoke-GitCommand -Arguments @("-C", $tempRoot, "commit", "-m", "Publish Phase 6D TWSE seed [skip ci]") | Out-Null
    }
    elseif ($diff.ExitCode -ne 0) { throw "Failed to inspect staged TWSE state changes." }
    Invoke-GitCommand -Arguments @("-C", $tempRoot, "push", $Remote, "HEAD:$Branch") | Out-Null
    Write-Host "TWSE institutional seed published to $Remote/$Branch."
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Invoke-GitCommand -Arguments @("worktree", "remove", "--force", $tempRoot) -Quiet -AllowFailure | Out-Null
        if (Test-Path -LiteralPath $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
    }
}
