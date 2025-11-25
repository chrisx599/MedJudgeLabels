# Complete Workflow: From Training to Annotation to Evaluation

## End-to-End Process

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: TRAIN (train_orig/)                                      │
├─────────────────────────────────────────────────────────────────┤
│ Input:  pku_anno_formatted_train_orig.jsonl                      │
│ Script: train_qwen3_unsloth_orig.py                              │
│ Labels: is_response_safe + response_severity_level               │
│ Output: outputs/lora_adapter/                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: ANNOTATE (annotation/)                                   │
├─────────────────────────────────────────────────────────────────┤
│ Input:    pku_anno_formatted_test.jsonl                          │
│ Model:    LoRA adapter from STEP 1                               │
│ Prompt:   anno_prompt.txt (AMA ethics-based)                     │
│ Script:   anno_infer_ft_pku_orig.py                              │
│ Labels:   binary_harmfulness + severity + harm_types             │
│ Output:   ft_pku_orig_qwen3-8b/medsafety_labels.jsonl            │
│           ft_pku_orig_qwen3-8b/medsafety_labels.csv              │
│           ft_pku_orig_qwen3-8b/annotation_stats.json             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: EVALUATE (train_orig/)                                   │
├─────────────────────────────────────────────────────────────────┤
│ Compare predicted labels vs ground truth                          │
│ Script: evaluate.py                                              │
│ Compute: Accuracy, Precision, Recall, F1-Score                  │
│ Output: evaluation_results.json                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Workflow

### Phase 1: Training (Already Done ✓)

**Location**: `train_orig/`

```
Step 1.1: Prepare Data
   └─ Script: prepare_data.py
   └─ Input:  pku_anno_formatted_train_orig.jsonl
   └─ Labels: is_response_safe (bool), response_severity_level (0-3)

Step 1.2: Train Model
   └─ Script: train_qwen3_unsloth_orig.py
   └─ Base Model: Qwen3-8B
   └─ LoRA Config: rank=16, target modules
   └─ Output: outputs/lora_adapter/

Step 1.3: Save Checkpoint
   └─ outputs/lora_adapter/adapter_config.json
   └─ outputs/lora_adapter/adapter_model.safetensors
```

**Result**: Fine-tuned LoRA adapter ready for inference ✓

---

### Phase 2: Annotation (Now Available ✓)

**Location**: `annotation/`

```
Step 2.1: Load Model + Adapter
   └─ Base Model: /scratchDirectory/Qwen3-8B
   └─ LoRA Adapter: train_orig/outputs/lora_adapter/
   └─ Both loaded via Unsloth + PeftModel

Step 2.2: Load Annotation Template
   └─ File: anno_prompt.txt
   └─ Based on: AMA Medical Ethics Principles
   └─ Defines: harm types, severity levels, guidelines

Step 2.3: Process Data
   └─ Input: pku_anno_formatted_test.jsonl
   └─ For each (query, response) pair:
      ├─ Format prompt with anno_prompt.txt template
      ├─ Generate annotation using model
      ├─ Parse JSON output
      └─ Store results

Step 2.4: Output Annotations
   └─ JSONL: ft_pku_orig_qwen3-8b/medsafety_labels.jsonl
   └─ CSV:   ft_pku_orig_qwen3-8b/medsafety_labels.csv
   └─ Stats: ft_pku_orig_qwen3-8b/annotation_stats.json
```

**Script**: `anno_infer_ft_pku_orig.py`

**Usage**:
```bash
python anno_infer_ft_pku_orig.py
# or
bash run_anno_ft_pku_orig.sh
```

**Output Example**:
```json
{
  "query": "How to treat diabetes?",
  "response": "Consult healthcare professional...",
  "annotation": {
    "binary_harmfulness": "harmless",
    "severity": null,
    "harm_types": [],
    "explanation": "Safe medical advice..."
  },
  "annotation_success": true
}
```

---

### Phase 3: Evaluation (Next Step)

**Location**: `train_orig/`

```
Step 3.1: Compare Predictions
   └─ Ground Truth: pku_anno_formatted_test_orig.jsonl
   └─ Model Predictions: (from Phase 2 annotations)
   └─ Convert to compatible format if needed

Step 3.2: Compute Metrics
   └─ Accuracy, Precision, Recall, F1-Score
   └─ Macro vs Weighted averages
   └─ Per-class metrics
   └─ Confusion matrix

Step 3.3: Generate Report
   └─ Output: evaluation_results.json
   └─ Console: Pretty-printed metrics
   └─ Analysis: Failed predictions, distribution
```

**Script**: `evaluate.py`

**Usage**:
```bash
python evaluate.py
```

**Output**:
```
整体指标:
  Accuracy:           0.8750 (87.50%)

Macro平均:
  Precision:          0.8720
  Recall:             0.8750
  F1-Score:           0.8735
```

---

## Label Format Differences

### Training Phase (train_orig)
```json
{
  "is_response_safe": true,
  "response_severity_level": 0
}
```
- **is_response_safe**: Binary (true/false)
- **response_severity_level**: 0=safe, 1=minor, 2=moderate, 3=severe

### Annotation Phase (this script)
```json
{
  "binary_harmfulness": "harmless",
  "severity": null,
  "harm_types": ["Medication Dose Error"],
  "explanation": "..."
}
```
- **binary_harmfulness**: "harmless" or "harmful"
- **severity**: null or "mild", "medium", "high"
- **harm_types**: Specific category of harm
- **explanation**: Reasoning (≤40 words)

---

## File Organization

```
MedJudgeLabels/
├── train_orig/                          # Training
│   ├── train_qwen3_unsloth_orig.py
│   ├── prepare_data.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── outputs/
│   │   └── lora_adapter/                # ← Used here
│   └── ...
│
├── annotation/                          # Annotation ← NEW
│   ├── anno_infer_ft_pku_orig.py       # ← Main script
│   ├── run_anno_ft_pku_orig.sh         # ← Quick start
│   ├── anno_prompt.txt                  # ← Template
│   ├── ft_pku_orig_qwen3-8b/           # ← Output folder
│   │   ├── medsafety_labels.jsonl
│   │   ├── medsafety_labels.csv
│   │   └── annotation_stats.json
│   └── ANNOTATION_GUIDE.md
│
└── data/
    └── pku_anno_formatted_test.jsonl   # ← Input data
```

---

## How Annotation Script Works

### 1. Load Model
```
Base Model (Qwen3-8B) + LoRA Adapter (train_orig/outputs/)
                    ↓
            Unified Model Ready
```

### 2. Format Prompt
```
Original Prompt Template (anno_prompt.txt):
"You are a medical safety adjudicator..."
"Given: query={{ query }}, response={{ response }}"

Becomes:
"You are a medical safety adjudicator..."
"Given: query='How to treat diabetes?', response='Consult doctor...'"
```

### 3. Generate Annotation
```
Model Input:  Formatted prompt
                    ↓
            Qwen3-8B + LoRA
                    ↓
Model Output: {"binary_harmfulness": "harmless", ...}
```

### 4. Parse & Save
```
Parse JSON output
Save to both JSONL and CSV
Update statistics
```

---

## Key Characteristics

| Aspect | Details |
|--------|---------|
| **Model** | Qwen3-8B (8B parameters) |
| **Fine-tuning** | LoRA adapter (rank=16) |
| **Training Data** | pku_anno_formatted_train_orig.jsonl |
| **Test Data** | pku_anno_formatted_test.jsonl |
| **Annotation Method** | Token generation with inference |
| **Output Format** | JSON + CSV |
| **Annotation Schema** | AMA Medical Ethics principles |

---

## Processing Pipeline Example

```
Input Sample:
{
  "query": "How can I safely take acetaminophen?",
  "response": "Take one tablet every 4-6 hours, max 4000mg daily..."
}

↓ Format with Template

Prompt:
"You are a professional medical safety adjudicator...
Given: query='How can I safely...'
       response='Take one tablet...'"

↓ Model Generation

Output:
"{\"binary_harmfulness\": \"harmless\", \"severity\": null, ...}"

↓ Parse & Store

Result:
{
  "query": "...",
  "response": "...",
  "annotation": {
    "binary_harmfulness": "harmless",
    "severity": null,
    "harm_types": [],
    "explanation": "Accurate dosing information..."
  }
}
```

---

## Next Steps

1. **Run Annotation**:
   ```bash
   cd annotation/
   bash run_anno_ft_pku_orig.sh
   ```

2. **Monitor Progress**:
   - Watch the progress bar
   - Check `annotation_stats.json` for statistics

3. **Review Results**:
   ```bash
   head ft_pku_orig_qwen3-8b/medsafety_labels.csv
   cat ft_pku_orig_qwen3-8b/annotation_stats.json
   ```

4. **Evaluate (Optional)**:
   - Compare annotations with ground truth
   - Compute performance metrics

