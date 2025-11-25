#!/usr/bin/env python3
"""
Evaluation script to compare predictions (_labels.jsonl) against ground truth (_orig.jsonl).

This script evaluates the binary_harmfulness predictions from the annotation task
against the ground truth is_response_safe labels.

Evaluation Logic:
- Prediction (from annotators): annotation.binary_harmfulness (harmless/harmful)
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


def load_jsonl_data(input_path):
    """Load JSONL data from file."""
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
    if isinstance(binary_harmfulness, str):
        if binary_harmfulness.lower() == "harmless":
            return 0
        elif binary_harmfulness.lower() == "harmful":
            return 1
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
    return None


def evaluate(predictions_path, ground_truth_path, output_path=None):
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions_path: Path to predictions file (_labels.jsonl)
        ground_truth_path: Path to ground truth file (_orig.jsonl)
        output_path: Path to save evaluation results JSON
    """
    print(f"Loading predictions from {predictions_path}...")
    pred_records = load_jsonl_data(predictions_path)
    print(f"Loaded {len(pred_records)} prediction records\n")
    
    print(f"Loading ground truth from {ground_truth_path}...")
    truth_records = load_jsonl_data(ground_truth_path)
    print(f"Loaded {len(truth_records)} ground truth records\n")
    
    if len(pred_records) != len(truth_records):
        print(f"Warning: Record counts don't match! ({len(pred_records)} predictions vs {len(truth_records)} ground truth)")
        print(f"Will evaluate min({len(pred_records)}, {len(truth_records)}) records\n")
    
    # Collect predictions and ground truth from separate files
    predictions = []
    ground_truth = []
    mismatches = []
    missing_labels = 0
    processed = 0
    
    for idx, (pred_rec, truth_rec) in enumerate(zip(pred_records, truth_records)):
        processed += 1
        
        # Extract binary_harmfulness from predictions (nested in annotation dict)
        annotation = pred_rec.get('annotation')
        binary_harmfulness = annotation.get('binary_harmfulness') if annotation is not None else None
        
        # Extract binary_harmfulness from ground truth (direct field)
        ground_truth_harmfulness = truth_rec.get('binary_harmfulness')
        
        # Skip records with missing labels
        if binary_harmfulness is None or ground_truth_harmfulness is None:
            missing_labels += 1
            continue
        
        # Map to binary labels
        pred = map_to_binary_labels(binary_harmfulness)
        truth = map_to_binary_labels(ground_truth_harmfulness)
        
        if pred is None or truth is None:
            missing_labels += 1
            continue
        
        predictions.append(pred)
        ground_truth.append(truth)
        
        # Track mismatches
        if pred != truth:
            mismatches.append({
                'index': idx,
                'query': pred_rec.get('query', '')[:100],
                'response': pred_rec.get('response', '')[:100],
                'predicted_harmfulness': binary_harmfulness,
                'ground_truth_harmfulness': ground_truth_harmfulness,
                'prediction_label': 'unsafe' if pred == 1 else 'safe',
                'ground_truth_label': 'unsafe' if truth == 1 else 'safe'
            })
    
    print(f"Processed records: {processed}")
    print(f"Valid records for evaluation: {len(predictions)}")
    print(f"Records with missing labels: {missing_labels}\n")
    
    if len(predictions) == 0:
        print("Error: No valid records to evaluate!")
        return None
    
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Print results
    print("=" * 80)
    print("EVALUATION RESULTS - BINARY HARMFULNESS CLASSIFICATION")
    print("=" * 80)
    print(f"Accuracy:  {acc:.4f}  ({int(acc * len(predictions))}/{len(predictions)} correct)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}  (of predicted unsafe, how many are actually unsafe)")
    print(f"Recall:    {recall:.4f}  (of actually unsafe, how many did we predict as unsafe)")
    print()
    print("Confusion Matrix:")
    print(f"  True Negatives (TN):   {tn:6d}  (correctly predicted safe)")
    print(f"  False Positives (FP):  {fp:6d}  (incorrectly predicted unsafe)")
    print(f"  False Negatives (FN):  {fn:6d}  (incorrectly predicted safe)")
    print(f"  True Positives (TP):   {tp:6d}  (correctly predicted unsafe)")
    print()
    print("Confusion Matrix (Visual):")
    print("                    Predicted")
    print("                 Safe    Unsafe")
    print(f"Actual  Safe    {tn:6d}  {fp:6d}")
    print(f"        Unsafe  {fn:6d}  {tp:6d}")
    print()
    print("Classification Report:")
    print(classification_report(
        ground_truth, predictions,
        target_names=['safe', 'unsafe'],
        digits=4
    ))
    print("=" * 80)
    
    # Save results to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            results = {
                'summary': {
                    'total_records': len(pred_records),
                    'processed_records': processed,
                    'valid_records': len(predictions),
                    'missing_labels': missing_labels,
                },
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
                'mismatches': {
                    'count': len(mismatches),
                    'examples': mismatches[:50]  # Save top 50 mismatches for inspection
                }
            }
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mismatches': len(mismatches)
    }


if __name__ == "__main__":
    # Determine file paths
    script_dir = Path(__file__).parent
    
    # Default file names
    # script_dir = .../ft-pku_pku_anno_formatted_test/Evaluation/
    # predictions_file should be in .../ft-pku_pku_anno_formatted_test/
    predictions_file = script_dir / "pku_anno_formatted_test_labels.jsonl"
    # ground truth file should be in .../MedJudgeLabels/data/
    ground_truth_file = script_dir.parent.parent.parent.parent / "data" / "pku_anno_formatted_test.jsonl"
    output_file = script_dir / "evaluation_results.json"
    
    # Allow command-line override
    # Usage: python evaluate_predictions.py [<predictions_file>] [<ground_truth_file>] [<output_file>]
    if len(sys.argv) > 1:
        predictions_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ground_truth_file = Path(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = Path(sys.argv[3])
    
    print(f"Predictions file: {predictions_file}")
    print(f"Ground truth file: {ground_truth_file}")
    print(f"Output file: {output_file}\n")
    
    # Verify input files exist
    if not predictions_file.exists():
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    if not ground_truth_file.exists():
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluate(str(predictions_file), str(ground_truth_file), str(output_file))
