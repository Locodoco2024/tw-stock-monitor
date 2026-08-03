#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source_output="research/output"
twse_output="$source_output/twse"
mkdir -p "$twse_output"

for name in phase3_summary.csv phase3_dataset_manifest.csv phase3_stock_summary.csv phase3b_summary.csv; do
  test -f "$source_output/$name" || {
    echo "缺少 $source_output/$name，請先完成 Phase 3B。" >&2
    exit 1
  }
  cp -f "$source_output/$name" "$twse_output/$name"
done

python -m research.institutional_model.cli phase4d --phase4-market twse --output-dir "$twse_output" "$@"
python -m research.institutional_model.cli phase4e --phase4-market twse --output-dir "$twse_output" "$@"
python -m research.institutional_model.cli phase4f --phase4-market twse --output-dir "$twse_output" "$@"
python -m research.institutional_model.cli phase5d --phase4-market twse --output-dir "$twse_output" "$@"

echo "TWSE 模型訓練完成：$twse_output/phase5d_final_model_reports.zip"
