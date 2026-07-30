#!/usr/bin/env bash
set -euo pipefail
DB="${1:-research/data/institutional_phase1.sqlite}"
python -m src.institutional.daily_pipeline seed --db "$DB"
