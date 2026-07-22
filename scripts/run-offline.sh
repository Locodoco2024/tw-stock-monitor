#!/usr/bin/env bash
set -euo pipefail
python -m src.main \
  --offline-fixture tests/fixtures/sample_bundle.json \
  --no-discord \
  --state-file runtime/test-state.json \
  --output-dir site-test
echo "Report: site-test/index.html"
