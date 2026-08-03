#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source_output="research/output"
twse_output="$source_output/twse"
mkdir -p "$twse_output"

for name in phase3_summary.csv phase3_dataset_manifest.csv phase3_stock_summary.csv phase3b_summary.csv; do
  test -f "$source_output/$name" || {
    echo "Missing $source_output/$name. Complete Phase 3B first." >&2
    exit 1
  }
  cp -f "$source_output/$name" "$twse_output/$name"
done

python -m research.institutional_model.cli phase6b --output-dir "$twse_output" "$@"

echo "TWSE 40-day validation passed: $twse_output/phase6b_twse_40d_validation_reports.zip"
