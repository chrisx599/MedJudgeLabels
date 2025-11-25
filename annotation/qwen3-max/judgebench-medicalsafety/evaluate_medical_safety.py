#!/usr/bin/env python3
"""
Evaluation script to compare GPT-predicted risk levels against human-annotated ground truth.

This script evaluates the query_risk_level-gpt predictions against the query_risk_level-human
labels from the medsafety_labels.jsonl file.

Risk Level Classification:
- Non-serious: Low risk query (0)
- Serious: Medium risk query (1)
- Critical: High risk query (2)

Evaluation Logic:
- Prediction: query_risk_level-gpt (Non-serious/Serious/Critical)
- Ground Truth: query_risk_level-human (Non-serious/Serious/Critical)
- Compare predictions against ground truth labels
"""

import json
import sys
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from collections import Counter


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


def map_risk_level_to_numeric(risk_level):
    """
    Map risk level text to numeric values for evaluation.
    Non-serious -> 0 (low risk)
    Serious -> 1 (medium risk)
    Critical -> 2 (high risk)
    """
    if isinstance(risk_level, str):
        risk_lower = risk_level.lower().strip()
        if risk_lower == "non-serious":
            return 0
        elif risk_lower == "serious":
            return 1
        elif risk_lower == "critical":
            return 2
    return None


def map_numeric_to_risk_level(numeric):
    """Convert numeric back to text for display."""
    mapping = {0: "Non-serious", 1: "Serious", 2: "Critical"}
    return mapping.get(numeric, "Unknown")


def evaluate(data_path, output_path=None):
    """
    Evaluate GPT predictions against human ground truth.
    
    Args:
        data_path: Path to medsafety_labels.jsonl file
        output_path: Path to save evaluation results JSON
    """
    print(f"Loading data from {data_path}...")
    records = load_jsonl_data(data_path)
    print(f"Loaded {len(records)} records\n")
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    mismatches = []
    missing_labels = 0
    processed = 0
    
    # Track distribution of labels
    pred_dist = Counter()
    truth_dist = Counter()
    
    for idx, record in enumerate(records):
        processed += 1
        
        # Extract risk levels
        gpt_risk = record.get('query_risk_level-gpt')
        human_risk = record.get('query_risk_level-human')
        
        # Skip records with missing labels
        if gpt_risk is None or human_risk is None:
            missing_labels += 1
            continue
        
        # Map to numeric labels
        pred = map_risk_level_to_numeric(gpt_risk)
        truth = map_risk_level_to_numeric(human_risk)
        
        if pred is None or truth is None:
            missing_labels += 1
            continue
        
        predictions.append(pred)
        ground_truth.append(truth)
        
        # Track distribution
        pred_dist[gpt_risk] += 1
        truth_dist[human_risk] += 1
        
        # Track mismatches
        if pred != truth:
            mismatches.append({
                'index': idx,
                'id': record.get('id', ''),
                'query': record.get('query', '')[:100],
                'response': record.get('response', '')[:100],
                'predicted_risk': gpt_risk,
                'ground_truth_risk': human_risk,
                'predicted_numeric': pred,
                'ground_truth_numeric': truth
            })
    
    print(f"Processed records: {processed}")
    print(f"Valid records for evaluation: {len(predictions)}")
    print(f"Records with missing labels: {missing_labels}\n")
    
    if len(predictions) == 0:
        print("Error: No valid records to evaluate!")
        return None
    
    # Print label distribution
    print("=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    print("\nGPT Predictions (query_risk_level-gpt):")
    for level in ["Non-serious", "Serious", "Critical"]:
        count = pred_dist[level]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {level:15s}: {count:6d} ({pct:5.1f}%)")
    
    print("\nHuman Ground Truth (query_risk_level-human):")
    for level in ["Non-serious", "Serious", "Critical"]:
        count = truth_dist[level]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {level:15s}: {count:6d} ({pct:5.1f}%)")
    print()
    
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    f1_weighted = f1_score(ground_truth, predictions, average='weighted')
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    precision_weighted = precision_score(ground_truth, predictions, average='weighted')
    recall_weighted = recall_score(ground_truth, predictions, average='weighted')
    cm = confusion_matrix(ground_truth, predictions)
    
    # Print results
    print("=" * 80)
    print("EVALUATION RESULTS - QUERY RISK LEVEL CLASSIFICATION")
    print("=" * 80)
    print(f"Accuracy:           {acc:.4f}  ({int(acc * len(predictions))}/{len(predictions)} correct)")
    print(f"F1 Score (Weighted):{f1_weighted:.4f}")
    print(f"F1 Score (Macro):  {f1_macro:.4f}")
    print(f"Precision (Weighted):{precision_weighted:.4f}")
    print(f"Recall (Weighted):  {recall_weighted:.4f}")
    print()
    print("Confusion Matrix (rows=ground_truth, cols=predictions):")
    print("                   Non-serious  Serious  Critical")
    risk_levels = ["Non-serious", "Serious", "Critical"]
    for i, true_label in enumerate(risk_levels):
        row_str = f"  {true_label:15s}"
        for j in range(3):
            row_str += f"  {cm[i, j]:8d}"
        print(row_str)
    print()
    print("Classification Report (per class):")
    print(classification_report(
        ground_truth, predictions,
        target_names=['Non-serious', 'Serious', 'Critical'],
        digits=4
    ))
    print("=" * 80)
    
    # Print sample mismatches
    if mismatches:
        print(f"\nSample Mismatches ({min(10, len(mismatches))} of {len(mismatches)}):")
        print("-" * 80)
        for i, mismatch in enumerate(mismatches[:10]):
            print(f"\n[{i+1}] ID: {mismatch['id']}")
            print(f"    Query: {mismatch['query']}")
            print(f"    Response: {mismatch['response']}")
            print(f"    Predicted: {mismatch['predicted_risk']} | Ground Truth: {mismatch['ground_truth_risk']}")
    
    # Save results to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            results = {
                'summary': {
                    'total_records': len(records),
                    'processed_records': processed,
                    'valid_records': len(predictions),
                    'missing_labels': missing_labels,
                },
                'label_distribution': {
                    'gpt_predictions': dict(pred_dist),
                    'human_ground_truth': dict(truth_dist)
                },
                'metrics': {
                    'accuracy': float(acc),
                    'f1_score_weighted': float(f1_weighted),
                    'f1_score_macro': float(f1_macro),
                    'precision_weighted': float(precision_weighted),
                    'recall_weighted': float(recall_weighted)
                },
                'confusion_matrix': {
                    'labels': ['Non-serious', 'Serious', 'Critical'],
                    'matrix': cm.tolist()
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
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision_weighted,
        'recall': recall_weighted,
        'mismatches': len(mismatches)
    }


if __name__ == "__main__":
    # Determine file paths
    script_dir = Path(__file__).parent
    
    # Default file paths
    data_file = script_dir / "medsafety_labels.jsonl"
    output_file = script_dir / "evaluation_results.json"
    
    # Allow command-line override
    # Usage: python evaluate_medical_safety.py [<data_file>] [<output_file>]
    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    
    print(f"Data file: {data_file}")
    print(f"Output file: {output_file}\n")
    
    # Verify input file exists
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluate(str(data_file), str(output_file))
