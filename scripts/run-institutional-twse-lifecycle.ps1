$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$sourceOutput = Join-Path $root "research/output"
$twseOutput = Join-Path $sourceOutput "twse"
New-Item -ItemType Directory -Force -Path $twseOutput | Out-Null

$requiredMetadata = @(
    "phase3_summary.csv",
    "phase3_dataset_manifest.csv",
    "phase3_stock_summary.csv",
    "phase3b_summary.csv"
)

foreach ($name in $requiredMetadata) {
    $source = Join-Path $sourceOutput $name
    if (-not (Test-Path $source)) {
        throw "Missing required file: $source. Complete Phase 3B first."
    }
    Copy-Item -Force $source (Join-Path $twseOutput $name)
}

$phase6bSummary = Join-Path $twseOutput "phase6b_summary.csv"
if (-not (Test-Path $phase6bSummary)) {
    throw "Missing Phase 6B result: $phase6bSummary"
}

Write-Host "[1/1] TWSE lifecycle and liquidity validation"
python -m research.institutional_model.cli phase6c `
    --output-dir $twseOutput @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "TWSE lifecycle validation passed. Report: research/output/twse/phase6c_twse_lifecycle_validation_reports.zip"
