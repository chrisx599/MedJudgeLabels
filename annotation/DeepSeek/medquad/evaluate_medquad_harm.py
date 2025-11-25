#!/usr/bin/env python3
"""
Evaluation script for MedQuad harm assessment classifications.

This script compares two harm classification tasks:
1. Extent of Harm: MedQuad_Extent_of_Harm vs llmAns_Extend_of_Harm
2. Likelihood of Harm: MedQuad_Likelihood_of_Harm vs llmAns_Likelihood_of_Harm

Harm Extent Classification:
- No harm (0)
- Moderate or mild harm (1)
- Death or severe harm (2)

Harm Likelihood Classification:
- Low (0)
- Medium (1)
- High (2)

This evaluation compares how well the MedQuad dataset labels (reference)
align with the harm predictions based on LLM answers.
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


def map_harm_extent_to_numeric(harm_extent):
    """
    Map harm extent text to numeric values.
    No harm -> 0
    Moderate or mild harm -> 1
    Death or severe harm -> 2
    """
    if isinstance(harm_extent, str):
        harm_lower = harm_extent.lower().strip()
        if harm_lower == "no harm":
            return 0
        elif harm_lower == "moderate or mild harm":
            return 1
        elif harm_lower == "death or severe harm":
            return 2
    return None


def map_harm_likelihood_to_numeric(harm_likelihood):
    """
    Map harm likelihood text to numeric values.
    Low -> 0
    Medium -> 1
    High -> 2
    """
    if isinstance(harm_likelihood, str):
        likelihood_lower = harm_likelihood.lower().strip()
        if likelihood_lower == "low":
            return 0
        elif likelihood_lower == "medium":
            return 1
        elif likelihood_lower == "high":
            return 2
    return None


def map_numeric_to_harm_extent(numeric):
    """Convert numeric back to text for display."""
    mapping = {0: "No harm", 1: "Moderate or mild harm", 2: "Death or severe harm"}
    return mapping.get(numeric, "Unknown")


def map_numeric_to_harm_likelihood(numeric):
    """Convert numeric back to text for display."""
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    return mapping.get(numeric, "Unknown")


def evaluate_task(records, task_name, truth_field, pred_field, map_func, class_names):
    """
    Evaluate a single harm classification task.
    
    Args:
        records: List of data records
        task_name: Name of the task for display
        truth_field: Field name for ground truth (MedQuad label)
        pred_field: Field name for predictions (LLM answer label)
        map_func: Function to map text values to numeric
        class_names: List of class names for reporting
    
    Returns:
        Dictionary with evaluation metrics and data
    """
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"{'='*80}")
    print(f"Ground Truth Field: {truth_field}")
    print(f"Prediction Field: {pred_field}\n")
    
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
        
        # Extract ground truth and predictions
        truth_text = record.get(truth_field)
        pred_text = record.get(pred_field)
        
        # Skip records with missing labels
        if pred_text is None or truth_text is None:
            missing_labels += 1
            continue
        
        # Map to numeric labels
        truth = map_func(truth_text)
        pred = map_func(pred_text)
        
        if pred is None or truth is None:
            missing_labels += 1
            continue
        
        predictions.append(pred)
        ground_truth.append(truth)
        
        # Track distribution
        truth_dist[truth_text] += 1
        pred_dist[pred_text] += 1
        
        # Track mismatches
        if pred != truth:
            mismatches.append({
                'index': idx,
                'qid': record.get('QID', ''),
                'query': record.get('query', '')[:100],
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
    print("Label Distribution (Ground Truth - MedQuad):")
    for cls in class_names:
        count = truth_dist[cls]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {cls:30s}: {count:6d} ({pct:5.1f}%)")
    
    print("\nLabel Distribution (Predictions - LLM Answer):")
    for cls in class_names:
        count = pred_dist[cls]
        pct = (count / len(predictions)) * 100 if len(predictions) > 0 else 0
        print(f"  {cls:30s}: {count:6d} ({pct:5.1f}%)")
    print()
    
    # Calculate metrics
    acc = accuracy_score(ground_truth, predictions)
    f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    # Ensure confusion matrix includes all label classes, even if not present in ground truth or predictions
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(ground_truth, predictions, labels=all_labels)
    
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
            print(f"\n[{i+1}] QID: {mismatch['qid']}")
            print(f"    Query: {mismatch['query']}")
            print(f"    Ground Truth (MedQuad): {mismatch['ground_truth']}")
            print(f"    Predicted (LLM): {mismatch['predicted']}")
    
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
    Evaluate both harm extent and harm likelihood classifications.
    
    Args:
        data_path: Path to medquad_annotated_xx.jsonl file
        output_path: Path to save evaluation results JSON
    """
    print(f"Loading data from {data_path}...")
    records = load_jsonl_data(data_path)
    print(f"Loaded {len(records)} records")
    
    # Evaluate Harm Extent Task
    extent_results = evaluate_task(
        records,
        "Harm Extent Classification (MedQuad_Extent_of_Harm vs llmAns_Extend_of_Harm)",
        'MedQuad_Extent_of_Harm',
        'llmAns_Extend_of_Harm',
        map_harm_extent_to_numeric,
        ["No harm", "Moderate or mild harm", "Death or severe harm"]
    )
    
    # Evaluate Harm Likelihood Task
    likelihood_results = evaluate_task(
        records,
        "Harm Likelihood Classification (MedQuad_Likelihood_of_Harm vs llmAns_Likelihood_of_Harm)",
        'MedQuad_Likelihood_of_Harm',
        'llmAns_Likelihood_of_Harm',
        map_harm_likelihood_to_numeric,
        ["Low", "Medium", "High"]
    )
    
    # Save results to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            results = {
                'harm_extent': extent_results,
                'harm_likelihood': likelihood_results
            }
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    # Determine file paths
    script_dir = Path(__file__).parent
    
    # Find the most recent medquad annotated file
    medquad_files = sorted(script_dir.glob("medquad_annotated_*.jsonl"))
    
    if not medquad_files:
        print("Error: No medquad_annotated_*.jsonl files found in the script directory!")
        sys.exit(1)
    
    data_file = medquad_files[-1]  # Use the most recent file
    output_file = script_dir / "medquad_evaluation_results.json"
    
    # Allow command-line override
    # Usage: python evaluate_medquad_harm.py [<data_file>] [<output_file>]
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
