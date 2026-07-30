$ErrorActionPreference = "Stop"

if (-not $env:FINMIND_TOKEN) {
    Write-Warning "FINMIND_TOKEN is not set. Full-market backfill requires a token."
}

python -m research.institutional_model.cli self-check
python -m research.institutional_model.cli phase2 --max-stocks 100 --continuous --quota-wait-minutes 65 @args
