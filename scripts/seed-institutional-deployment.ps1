param(
    [string]$Database = "research/data/institutional_phase1.sqlite",
    [string]$StateDir = "runtime/institutional",
    [string]$UsersConfig = "configs/users",
    [int]$LookbackMarketDays = 100,
    [string]$AsOfDate = ""
)

$ErrorActionPreference = "Stop"
$arguments = @(
    "-m", "src.institutional.daily_pipeline", "seed",
    "--db", $Database,
    "--state-dir", $StateDir,
    "--users-config", $UsersConfig,
    "--lookback-market-days", $LookbackMarketDays
)
if ($AsOfDate) {
    $arguments += @("--as-of-date", $AsOfDate)
}
python @arguments
