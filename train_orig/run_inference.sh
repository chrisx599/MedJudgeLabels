#!/bin/bash
# Quick inference on test data

echo "========================================"
echo "Running Inference on Test Data"
echo "========================================"

python inference.py \
  --test_data_path /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test_orig.jsonl \
  --lora_adapter_path ./outputs/lora_adapter \
  --output ./inference_results.json \
  --max_samples 100

echo "✓ Done! Results saved to ./inference_results.json"
