$ErrorActionPreference = "Stop"
python -m src.main `
  --offline-fixture tests/fixtures/sample_bundle.json `
  --no-discord `
  --state-file runtime/test-state.json `
  --output-dir site-test
Write-Host "Report: site-test/index.html"
