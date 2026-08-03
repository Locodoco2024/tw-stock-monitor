#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_ARGS=()
if [[ "${1:-}" == "--force" ]]; then
  EXTRA_ARGS+=(--force)
fi

python -m research.institutional_model.cli phase6d \
  --output-dir research/output/twse \
  --phase6d-model-dir models/twse \
  "${EXTRA_ARGS[@]}"
