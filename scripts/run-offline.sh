#!/usr/bin/env bash
set -euo pipefail
python -m src.main \
  --offline-fixture tests/fixtures/holding_quotes.json \
  --no-discord \
  --force-notify \
  --state-file runtime/test-state.json
