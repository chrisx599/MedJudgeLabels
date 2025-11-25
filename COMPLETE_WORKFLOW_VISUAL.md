# Visual Workflow: From Inference to Annotation to Evaluation

## The Big Picture

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                    COMPLETE ANALYSIS PIPELINE                         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

                          PHASE 1: TRAINING ✓
    ┌─────────────────────────────────────────────────┐
    │  train_orig/train_qwen3_unsloth_orig.py         │
    │                                                  │
    │  Input:  pku_anno_formatted_train_orig.jsonl    │
    │  Output: outputs/lora_adapter/                  │
    │                                                  │
    │  Labels: is_response_safe,                      │
    │          response_severity_level (0-3)          │
    └─────────────────────────────────────────────────┘
                           ↓↓↓
                  (LoRA Adapter Created)
                           ↓↓↓
                  PHASE 2: ANNOTATION (NEW)
    ┌─────────────────────────────────────────────────┐
    │  annotation/anno_infer_ft_pku_orig.py           │
    │                                                  │
    │  Input:  pku_anno_formatted_test.jsonl          │
    │  Template: anno_prompt.txt (AMA ethics)         │
    │  Model: Qwen3-8B + LoRA adapter                 │
    │                                                  │
    │  Output: ft_pku_orig_qwen3-8b/                  │
    │    - medsafety_labels.jsonl                     │
    │    - medsafety_labels.csv                       │
    │    - annotation_stats.json                      │
    │                                                  │
    │  Labels: binary_harmfulness,                    │
    │          severity,                              │
    │          harm_types,                            │
    │          explanation                            │
    └─────────────────────────────────────────────────┘
                           ↓↓↓
                  (Data Annotated with Labels)
                           ↓↓↓
                  PHASE 3: EVALUATION
    ┌─────────────────────────────────────────────────┐
    │  train_orig/evaluate.py                         │
    │                                                  │
    │  Compare:                                        │
    │    - Ground truth labels                        │
    │    - Model predictions (from annotation)        │
    │                                                  │
    │  Output: evaluation_results.json                │
    │    - Accuracy                                   │
    │    - Precision, Recall, F1-Score               │
    │    - Confusion Matrix                          │
    │    - Per-class Metrics                         │
    └─────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  RAW QUERY-RESPONSE PAIRS                                        │
│  (pku_anno_formatted_test.jsonl)                                 │
│  ├─ query: "How to treat diabetes?"                             │
│  └─ response: "Consult a healthcare professional..."            │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   ANNOTATION SCRIPT           │
         │  (anno_infer_ft_pku_orig.py)  │
         │                               │
         │  1. Load Fine-tuned Model     │
         │  2. Format with Template      │
         │  3. Generate Annotation       │
         │  4. Parse JSON Output         │
         │  5. Save Results              │
         └────────────┬──────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ANNOTATED DATA (JSONL)                                          │
│  (ft_pku_orig_qwen3-8b/medsafety_labels.jsonl)                  │
│                                                                  │
│  {                                                               │
│    "query": "How to treat diabetes?",                           │
│    "response": "Consult a healthcare professional...",          │
│    "annotation": {                                              │
│      "binary_harmfulness": "harmless",                          │
│      "severity": null,                                          │
│      "harm_types": [],                                          │
│      "explanation": "Safe recommendation..."                    │
│    },                                                            │
│    "annotation_success": true                                   │
│  }                                                               │
│                                                                  │
└────────┬──────────────────────────────────────────────────┬─────┘
         │                                                  │
         ↓                                                  ↓
    ┌─────────────┐                              ┌─────────────────┐
    │ JSONL       │                              │ CSV             │
    │ (Machine    │                              │ (Human          │
    │  Readable)  │                              │  Readable)      │
    └─────────────┘                              └─────────────────┘
         
         Also generates: annotation_stats.json
         ├─ total: 1000
         ├─ successful: 980
         ├─ harmless: 750
         ├─ harmful: 230
         └─ severity distribution
```

---

## Script Integration

```
train_orig/
│
├─ inference.py  ←─────┐
│   (Test inference)   │
│                      │
├─ evaluate.py  ←──────┼───────────────────┐
│   (Evaluate)         │                   │
│                      │                   │
└─ outputs/            │                   │
   └─ lora_adapter/ ───┼───────────┐       │
                       │           │       │
                       ↓           ↓       ↓
annotation/
│
├─ anno_infer_ft_pku_orig.py ←── Uses both:
│   (Annotate with FT model)      1. LoRA adapter
│                                  2. anno_prompt.txt
├─ anno_prompt.txt
│   (Medical ethics template)
│
└─ ft_pku_orig_qwen3-8b/
   (Output folder)
   ├─ medsafety_labels.jsonl  ─┐
   ├─ medsafety_labels.csv    ─┼─→ Can be used as input
   └─ annotation_stats.json     │   to evaluate.py
```

---

## Three-Phase Process

### Phase 1️⃣: Training (DONE ✓)

```
Raw Training Data
       ↓
   Fine-tune with LoRA
       ↓
   LoRA Adapter Saved
       ✓ Ready to Use
```

**Your Model**: `train_orig/outputs/lora_adapter/`

---

### Phase 2️⃣: Annotation (NOW AVAILABLE ✓)

```
Test Data
       ↓
  + Fine-tuned Model
       ↓
  + Medical Ethics Prompt
       ↓
   Generate Annotations
       ↓
   Parse Output
       ↓
  Save as JSONL + CSV
       ✓ Results Ready
```

**Your Script**: `annotation/anno_infer_ft_pku_orig.py`
**Output**: `annotation/ft_pku_orig_qwen3-8b/`

---

### Phase 3️⃣: Evaluation (NEXT STEP)

```
Annotations
       ↓
  + Ground Truth
       ↓
  Compare Labels
       ↓
 Calculate Metrics
       ↓
  Report Results
       ✓ Analysis Ready
```

**Your Script**: `train_orig/evaluate.py`

---

## Why This Workflow?

```
Train → Annotate → Evaluate
  ↓         ↓         ↓
Create   Use Model  Measure
Model    to Label   Quality
         Data
```

1. **Train**: Fine-tune model on labeled data
2. **Annotate**: Use model to predict labels on new data
3. **Evaluate**: Compare predictions vs ground truth

---

## Label Evolution

```
TRAINING DATA (Original Format)
{
  "is_response_safe": true,
  "response_severity_level": 0
}
        ↓
        │ (Your model learns this)
        ↓
FINE-TUNED MODEL
        ↓
        │ (Use to annotate)
        ↓
ANNOTATED DATA (New Format)
{
  "binary_harmfulness": "harmless",
  "severity": null,
  "harm_types": [],
  "explanation": "..."
}
        ↓
        │ (Compare with ground truth)
        ↓
EVALUATION METRICS
{
  "accuracy": 0.87,
  "precision": 0.84,
  ...
}
```

---

## Quick Execution

### Single Command (Everything)
```bash
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation
bash run_anno_ft_pku_orig.sh
```

### Step by Step
```bash
# 1. Go to annotation folder
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation

# 2. Run main script
python anno_infer_ft_pku_orig.py

# 3. Check results
ls -lh ft_pku_orig_qwen3-8b/
cat ft_pku_orig_qwen3-8b/annotation_stats.json
```

---

## File Organization

```
MedJudgeLabels/
│
├── train_orig/                          ← Training folder
│   ├── train_qwen3_unsloth_orig.py
│   ├── prepare_data.py
│   ├── inference.py                     ← Test inference
│   ├── evaluate.py                      ← Evaluation
│   └── outputs/
│       └── lora_adapter/                ← USED BY ANNOTATION
│
├── annotation/                          ← Annotation folder (NEW)
│   ├── anno_infer_ft_pku_orig.py       ← MAIN SCRIPT
│   ├── run_anno_ft_pku_orig.sh         ← Quick start
│   ├── anno_prompt.txt                 ← Medical ethics template
│   ├── ft_pku_orig_qwen3-8b/           ← OUTPUT FOLDER
│   │   ├── medsafety_labels.jsonl
│   │   ├── medsafety_labels.csv
│   │   └── annotation_stats.json
│   ├── ANNOTATION_GUIDE.md
│   ├── QUICK_REFERENCE.md
│   └── README_ANNOTATION_SETUP.md
│
└── data/
    └── pku_anno_formatted_test.jsonl   ← INPUT DATA
```

---

## Information Flow

```
                    ╔════════════════════════╗
                    ║   Your Fine-tuned      ║
                    ║   LoRA Model           ║
                    ║   (train_orig/)        ║
                    ╚════════════╤═══════════╝
                                 │
         ┌───────────────────────┼───────────────────────┐
         ↓                       ↓                       ↓
    ┌─────────────┐         ┌──────────┐          ┌──────────────┐
    │   Inference │         │Annotate  │          │  Evaluate    │
    │   (Quick    │         │(Full     │          │  (Compare    │
    │    Test)    │         │Pipeline) │          │   Results)   │
    └─────────────┘         └──────────┘          └──────────────┘
                                 ↓
                        ┌───────────────┐
                        │ Output Data   │
                        │ (JSONL+CSV)   │
                        └───────────────┘
```

---

## Summary

```
BEFORE:
✗ Raw data (query + response only)
✗ No labels

AFTER ANNOTATION:
✓ Labeled data (safety + harm type)
✓ Ready for analysis
✓ Can evaluate model quality

USES:
✓ Understand model predictions
✓ Compare with ground truth
✓ Identify model weaknesses
✓ Improve training data
```

