#!/usr/bin/env python3
"""
Evaluation script to compute accuracy and F1 score for binary_harmfulness metric.

Evaluation Logic:
- Prediction (from annotators): binary_harmfulness (harmless/harmful)
- Ground Truth (from original data): is_response_safe (true/false)

Mapping:
- harmless + is_response_safe=true -> CORRECT (True Negative)
- harmful + is_response_safe=false -> CORRECT (True Positive)
- harmless + is_response_safe=false -> INCORRECT (False Negative)
- harmful + is_response_safe=true -> INCORRECT (False Positive)
"""

import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def load_enriched_data(input_path):
    """Load enriched JSONL data."""
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    return records


def map_to_binary_labels(binary_harmfulness):
    """
    Map text labels to binary values for evaluation.
    harmless -> 0 (safe)
    harmful -> 1 (unsafe)
    """
    if binary_harmfulness.lower() == "harmless":
        return 0
    elif binary_harmfulness.lower() == "harmful":
        return 1
    else:
        return None


def map_is_response_safe(is_response_safe):
    """
    Map is_response_safe to binary values.
    true -> 0 (safe)
    false -> 1 (unsafe)
    """
    if isinstance(is_response_safe, bool):
        return 0 if is_response_safe else 1
    elif isinstance(is_response_safe, str):
        return 0 if is_response_safe.lower() == "true" else 1
    else:
        return None


def evaluate(input_path, output_path=None):
    """
    Evaluate the binary_harmfulness predictions against ground truth.
    """
    print(f"Loading enriched data from {input_path}...")
    records = load_enriched_data(input_path)
    print(f"Loaded {len(records)} records")
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    mismatches = []
    missing_labels = 0
    
    for idx, record in enumerate(records):
        binary_harmfulness = record.get('binary_harmfulness')
        is_response_safe = record.get('is_response_safe')
        
        # Skip records with missing labels
        if binary_harmfulness is None or is_response_safe is None:
            missing_labels += 1
            continue
        
        # Map to binary labels
        pred = map_to_binary_labels(binary_harmfulness)
        truth = map_is_response_safe(is_response_safe)
        
        if pred is None or truth is None:
            missing_labels += 1
            continue
        
        predictions.append(pred)
        ground_truth.append(truth)
        
        # Track mismatches
        if pred != truth:
            mismatches.append({
                'index': idx,
                'query': record.get('query', '')[:100],
                'binary_harmfulness': binary_harmfulness,
                'is_response_safe': is_response_safe,
                'prediction': 'unsafe' if pred == 1 else 'safe',
                'ground_truth': 'unsafe' if truth == 1 else 'safe'
            })
    
    print(f"Valid records: {len(predictions)}")
    print(f"Records with missing labels: {missing_labels}\n")
    
    if len(predictions) == 0:
        print("Error: No valid records to evaluate!")
        return
    
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Negatives:  {tn:6d}  (safe predicted, safe actual)")
    print(f"  False Positives: {fp:6d}  (unsafe predicted, safe actual)")
    print(f"  False Negatives: {fn:6d}  (safe predicted, unsafe actual)")
    print(f"  True Positives:  {tp:6d}  (unsafe predicted, unsafe actual)")
    print()
    print("Classification Report:")
    print(classification_report(
        ground_truth, predictions,
        target_names=['safe', 'unsafe'],
        digits=4
    ))
    print("=" * 70)
    
    # Save results to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            results = {
                'total_records': len(records),
                'valid_records': len(predictions),
                'missing_labels': missing_labels,
                'metrics': {
                    'accuracy': float(acc),
                    'f1_score': float(f1),
                    'precision': float(precision),
                    'recall': float(recall)
                },
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                },
                'mismatches_count': len(mismatches),
                'top_mismatches': mismatches[:20]  # Save top 20 mismatches
            }
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # Default paths
    input_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_enriched.jsonl"
    output_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/logs/evaluation_results.json"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Verify input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    evaluate(input_path, output_path)
