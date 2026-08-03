#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRA_ARGS=()
if [[ -n "${1:-}" ]]; then
  EXTRA_ARGS+=(--as-of-date "$1")
fi
python -m src.institutional.twse_daily_pipeline seed \
  --state-dir runtime/institutional_twse \
  --model-dir models/twse \
  "${EXTRA_ARGS[@]}"
