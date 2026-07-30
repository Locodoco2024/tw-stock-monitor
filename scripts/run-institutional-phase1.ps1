$ErrorActionPreference = "Stop"

if (-not $env:FINMIND_TOKEN) {
    Write-Warning "FINMIND_TOKEN is not set. Anonymous API quota will be used."
}

python -m research.institutional_model.cli self-check
python -m research.institutional_model.cli phase1 @args
