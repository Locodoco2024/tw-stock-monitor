#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FINMIND_TOKEN:-}" ]]; then
  echo "Warning: FINMIND_TOKEN is not set. Full-market backfill requires a token." >&2
fi

python -m research.institutional_model.cli self-check
python -m research.institutional_model.cli phase2 --max-stocks 100 --continuous --quota-wait-minutes 65 "$@"
