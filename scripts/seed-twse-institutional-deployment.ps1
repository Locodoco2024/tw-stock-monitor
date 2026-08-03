param(
    [string]$AsOfDate = ""
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$argsList = @(
    "-m", "src.institutional.twse_daily_pipeline",
    "seed",
    "--state-dir", "runtime/institutional_twse",
    "--model-dir", "models/twse"
)
if ($AsOfDate) {
    $argsList += @("--as-of-date", $AsOfDate)
}
& python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
