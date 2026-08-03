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

Write-Host "[1/4] TWSE 10/20/40-day horizon research"
python -m research.institutional_model.cli phase4d `
    --phase4-market twse `
    --output-dir $twseOutput @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] TWSE 20-day return-rank validation"
python -m research.institutional_model.cli phase4e `
    --phase4-market twse `
    --output-dir $twseOutput @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] TWSE signal lifecycle validation"
python -m research.institutional_model.cli phase4f `
    --phase4-market twse `
    --output-dir $twseOutput @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] TWSE final model training and historical replay"
python -m research.institutional_model.cli phase5d `
    --phase4-market twse `
    --output-dir $twseOutput @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "TWSE model training completed. Report: research/output/twse/phase5d_final_model_reports.zip"
