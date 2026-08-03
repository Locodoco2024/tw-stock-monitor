param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$argsList = @(
    "-m", "research.institutional_model.cli",
    "phase6d",
    "--output-dir", "research/output/twse",
    "--phase6d-model-dir", "models/twse"
)
if ($Force) {
    $argsList += "--force"
}

Write-Host "Training the final TWSE 40-day institutional model..."
& python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "TWSE final model completed."
Write-Host "Model: models/twse/phase5d_final_rank_model.json"
Write-Host "Report: research/output/twse/phase6d_twse_final_model_reports.zip"
