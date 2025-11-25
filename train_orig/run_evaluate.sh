#!/bin/bash
# Quick evaluation script

echo "========================================"
echo "Running Evaluation"
echo "========================================"

python evaluate.py \
  --train_data_path /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_train_orig.jsonl \
  --test_data_path /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test_orig.jsonl \
  --lora_adapter_path ./outputs/lora_adapter \
  --output ./evaluation_results.json \
  --predictions_file ./predictions.json

echo "✓ Done! Results saved to ./evaluation_results.json"
