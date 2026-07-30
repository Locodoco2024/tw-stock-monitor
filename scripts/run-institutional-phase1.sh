#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FINMIND_TOKEN:-}" ]]; then
  echo "提醒：尚未設定 FINMIND_TOKEN，將使用匿名 API 額度。" >&2
fi

python -m research.institutional_model.cli self-check
python -m research.institutional_model.cli phase1 "$@"
