param(
    [string]$StateDir = "runtime/institutional",
    [string]$Remote = "origin",
    [string]$Branch = "state"
)

$ErrorActionPreference = "Stop"
$resolvedState = Resolve-Path $StateDir
$required = @(
    "rolling_market_data.csv.gz",
    "universe.csv",
    "score_history.csv.gz",
    "latest_scores.csv.gz",
    "notification_plan.csv",
    "update_manifest.json"
)
foreach ($file in $required) {
    if (-not (Test-Path (Join-Path $resolvedState $file))) {
        throw "缺少 seed 檔案：$file。請先執行 seed-institutional-deployment.ps1"
    }
}

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "目前目錄不是 Git Repository"
}
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tw-stock-state-" + [guid]::NewGuid())
try {
    git fetch $Remote $Branch 2>$null
    $remoteExists = $LASTEXITCODE -eq 0
    if ($remoteExists) {
        git worktree add --force -B $Branch $tempRoot "$Remote/$Branch"
    }
    else {
        git worktree add --detach $tempRoot HEAD
        git -C $tempRoot switch --orphan $Branch
        git -C $tempRoot rm -rf . 2>$null
    }

    $institutionalTarget = Join-Path $tempRoot "institutional"
    if (Test-Path $institutionalTarget) {
        Remove-Item $institutionalTarget -Recurse -Force
    }
    Copy-Item $resolvedState $institutionalTarget -Recurse -Force

    $localState = Join-Path $repoRoot "runtime/state.json"
    $targetState = Join-Path $tempRoot "state.json"
    if (Test-Path $localState) {
        Copy-Item $localState $targetState -Force
    }
    elseif (-not (Test-Path $targetState)) {
        '{"records":{},"institutional_notification_keys":{}}' | Set-Content $targetState -Encoding UTF8
    }

    git -C $tempRoot add state.json institutional
    git -C $tempRoot config user.name "phase5h-seed"
    git -C $tempRoot config user.email "phase5h-seed@users.noreply.github.com"
    git -C $tempRoot commit -m "Publish Phase 5H institutional seed [skip ci]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "seed 內容沒有變更，略過 commit。"
    }
    git -C $tempRoot push $Remote $Branch
}
finally {
    if (Test-Path $tempRoot) {
        git worktree remove --force $tempRoot 2>$null
        if (Test-Path $tempRoot) {
            Remove-Item $tempRoot -Recurse -Force
        }
    }
}
