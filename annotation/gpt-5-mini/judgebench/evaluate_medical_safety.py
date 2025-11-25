#!/usr/bin/env python3
"""
Evaluation script to compare GPT-5-Mini predictions against human-annotated ground truth.

This script evaluates two classification tasks:
1. Query Risk Level: query_risk_level-gpt vs query_risk_level-human
2. Response Type: respond_type-gpt vs respond_type-human

Risk Level Classification:
- Non-serious: Low risk query (0)
- Serious: Medium risk query (1)
- Critical: High risk query (2)

Response Type Classification:
- Non-medical: Not medical related (0)
- No answer: No answer provided (1)
- Irrelevant or nonsensical: Irrelevant response (2)
- General information: General info provided (3)

Evaluation Logic:
- Prediction: query_risk_level-gpt, respond_type-gpt
- Ground Truth: query_risk_level-human, respond_type-human
- Compare predictions against ground truth labels for both tasks
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


def map_response_type_to_numeric(response_type):
    """
    Map response type text to numeric values for evaluation.
    Non-medical -> 0
    No answer -> 1
    Irrelevant or nonsensical -> 2
    General information -> 3
    """
    if isinstance(response_type, str):
        type_lower = response_type.lower().strip()
        if type_lower == "non-medical":
            return 0
        elif type_lower == "no answer":
            return 1
        elif type_lower == "irrelevant or nonsensical":
            return 2
        elif type_lower == "general information":
            return 3
    return None


def map_numeric_to_risk_level(numeric):
    """Convert numeric back to text for display."""
    mapping = {0: "Non-serious", 1: "Serious", 2: "Critical"}
    return mapping.get(numeric, "Unknown")


def map_numeric_to_response_type(numeric):
    """Convert numeric back to text for display."""
    mapping = {0: "Non-medical", 1: "No answer", 2: "Irrelevant or nonsensical", 3: "General information"}
    return mapping.get(numeric, "Unknown")


def evaluate_task(records, task_name, pred_field, truth_field, map_func, class_names):
    """
    Evaluate a single classification task.
    
    Args:
        records: List of data records
        task_name: Name of the task for display
        pred_field: Field name for predictions
        truth_field: Field name for ground truth
        map_func: Function to map text values to numeric
        class_names: List of class names for reporting
    
    Returns:
        Dictionary with evaluation metrics and data
    """
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"{'='*80}")
    
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
        
        # Extract predictions and ground truth
        pred_text = record.get(pred_field)
        truth_text = record.get(truth_field)
        
        # Skip records with missing labels
        if pred_text is None or truth_text is None:
            missing_labels += 1
            continue
        
        # Map to numeric labels
        pred = map_func(pred_text)
        truth = map_func(truth_text)
        
        if pred is None or truth is None:
            missing_labels += 1
            continue
        
        predictions.append(pred)
        ground_truth.append(truth)
        
        # Track distribution
        pred_dist[pred_text] += 1
        truth_dist[truth_text] += 1
        
        # Track mismatches
        if pred != truth:
            mismatches.append({
                'index': idx,
                'id': record.get('id', ''),
                'query': record.get('query', '')[:100],
                'response': record.get('response', '')[:100],
                'predicted': pred_text,
                'ground_truth': truth_text,
                'predicted_numeric': pred,
                'ground_truth_numeric': truth
            })
    
    print(f"Processed records: {processed}")
    print(f"Valid records for evaluation: {len(predictions)}")
    print(f"Records with missing labels: {missing_labels}\n")
    
    if len(predictions) == 0:
        print(f"Error: No valid records to evaluate for {task_name}!")
        return None
    
    # Print label distribution
    print("Label Distribution (Predictions):")
    for cls in class_names:
        count = pred_dist[cls]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {cls:30s}: {count:6d} ({pct:5.1f}%)")
    
    print("\nLabel Distribution (Ground Truth):")
    for cls in class_names:
        count = truth_dist[cls]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {cls:30s}: {count:6d} ({pct:5.1f}%)")
    print()
    
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(ground_truth, predictions, labels=list(range(len(class_names))))
    
    # Print results
    print(f"Accuracy:            {acc:.4f}  ({int(acc * len(predictions))}/{len(predictions)} correct)")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro):    {f1_macro:.4f}")
    print(f"Precision (Weighted):{precision_weighted:.4f}")
    print(f"Recall (Weighted):   {recall_weighted:.4f}")
    print()
    print(f"Confusion Matrix (rows=ground_truth, cols=predictions):")
    print("                                " + "  ".join(f"{cls:20s}" for cls in class_names))
    for i, true_label in enumerate(class_names):
        row_str = f"  {true_label:30s}"
        for j in range(len(class_names)):
            row_str += f"  {cm[i, j]:20d}"
        print(row_str)
    print()
    print("Classification Report (per class):")
    print(classification_report(
        ground_truth, predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))
    
    # Print sample mismatches
    if mismatches:
        print(f"\nSample Mismatches ({min(10, len(mismatches))} of {len(mismatches)}):")
        print("-" * 80)
        for i, mismatch in enumerate(mismatches[:10]):
            print(f"\n[{i+1}] ID: {mismatch['id']}")
            print(f"    Query: {mismatch['query']}")
            print(f"    Response: {mismatch['response']}")
            print(f"    Predicted: {mismatch['predicted']} | Ground Truth: {mismatch['ground_truth']}")
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision_weighted,
        'recall': recall_weighted,
        'confusion_matrix': cm.tolist(),
        'label_distribution': {
            'predictions': dict(pred_dist),
            'ground_truth': dict(truth_dist)
        },
        'mismatches': {
            'count': len(mismatches),
            'examples': mismatches[:50]  # Save top 50 mismatches
        },
        'summary': {
            'total_records': len(records),
            'processed_records': processed,
            'valid_records': len(predictions),
            'missing_labels': missing_labels
        }
    }


def evaluate(data_path, output_path=None):
    """
    Evaluate both risk level and response type classifications.
    
    Args:
        data_path: Path to medsafety_gpt5mini_labels.jsonl file
        output_path: Path to save evaluation results JSON
    """
    print(f"Loading data from {data_path}...")
    records = load_jsonl_data(data_path)
    print(f"Loaded {len(records)} records")
    
    # Evaluate Risk Level Task
    risk_results = evaluate_task(
        records,
        "Query Risk Level Classification (query_risk_level-gpt vs query_risk_level-human)",
        'query_risk_level-gpt',
        'query_risk_level-human',
        map_risk_level_to_numeric,
        ["Non-serious", "Serious", "Critical"]
    )
    
    # Evaluate Response Type Task
    response_results = evaluate_task(
        records,
        "Response Type Classification (respond_type-gpt vs respond_type-human)",
        'respond_type-gpt',
        'respond_type-human',
        map_response_type_to_numeric,
        ["Non-medical", "No answer", "Irrelevant or nonsensical", "General information"]
    )
    
    # Save results to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            results = {
                'query_risk_level': risk_results,
                'response_type': response_results
            }
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    # Determine file paths
    script_dir = Path(__file__).parent
    
    # Default file paths
    data_file = script_dir / "medsafety_gpt5mini_labels.jsonl"
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
