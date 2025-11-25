# 📚 Complete Documentation Index

## Quick Navigation

### 🚀 Getting Started (Pick One)
1. **New to this project?** → Start with [`ANNOTATION_SETUP_COMPLETE.md`](#annotation_setup_complete)
2. **Want a quick start?** → Go to [`annotation/QUICK_REFERENCE.md`](#quick_reference)
3. **Need details?** → Read [`annotation/ANNOTATION_GUIDE.md`](#annotation_guide)
4. **Visual learner?** → Check [`COMPLETE_WORKFLOW_VISUAL.md`](#complete_workflow_visual)

---

## 📄 Documentation Files

### Root Folder: MedJudgeLabels/

#### 1. **ANNOTATION_SETUP_COMPLETE.md** ⭐ START HERE
- **Purpose**: High-level setup summary
- **Contents**: What was created, quick start, key features
- **Length**: 5 min read
- **Best for**: Getting oriented

#### 2. **ANNOTATION_WORKFLOW.md**
- **Purpose**: End-to-end workflow explanation
- **Contents**: Three phases (Train → Annotate → Evaluate)
- **Length**: 10 min read
- **Best for**: Understanding the complete process

#### 3. **COMPLETE_WORKFLOW_VISUAL.md**
- **Purpose**: Visual diagrams and flowcharts
- **Contents**: ASCII diagrams, information flow, file organization
- **Length**: 7 min read
- **Best for**: Visual understanding

---

### annotation/ Folder

#### 4. **README_ANNOTATION_SETUP.md** ⭐ COMPLETE REFERENCE
- **Purpose**: Comprehensive annotation setup guide
- **Contents**: Overview, workflow, scripts, files, examples
- **Length**: 15 min read
- **Best for**: Full reference

#### 5. **ANNOTATION_GUIDE.md** ⭐ DETAILED GUIDE
- **Purpose**: Detailed usage guide with examples
- **Contents**: Input/output, schemas, commands, troubleshooting
- **Length**: 20 min read
- **Best for**: Learning all details

#### 6. **QUICK_REFERENCE.md** ⭐ QUICK LOOKUP
- **Purpose**: Quick command and reference guide
- **Contents**: TL;DR, commands, output examples
- **Length**: 5 min read
- **Best for**: Quick lookup while working

---

### train_orig/ Folder (Existing)

#### Related Documents
- `QUICK_START_INFERENCE_EVAL.md` - Inference & evaluation
- `INFERENCE_EVALUATION_GUIDE.md` - Detailed guide
- `DIFFERENCES_TRAIN_VS_TRAIN_ORIG.md` - Comparison with train/

---

## 🗂️ Main Scripts Created

### In annotation/ Folder

```
anno_infer_ft_pku_orig.py          # Main annotation script
run_anno_ft_pku_orig.sh            # Quick start bash script
```

### Related Existing Scripts

```
train_orig/inference.py            # Basic inference testing
train_orig/evaluate.py             # Evaluation metrics
```

---

## 📊 Complete File Structure

```
MedJudgeLabels/
│
├─ 📖 ANNOTATION_SETUP_COMPLETE.md     ← START HERE
├─ ANNOTATION_WORKFLOW.md
├─ COMPLETE_WORKFLOW_VISUAL.md
│
├─ train_orig/
│  ├─ train_qwen3_unsloth_orig.py     (Training - done)
│  ├─ inference.py                    (Basic inference)
│  ├─ evaluate.py                     (Evaluation)
│  └─ outputs/lora_adapter/           (Used by annotation)
│
├─ annotation/                        (NEW)
│  ├─ 📖 README_ANNOTATION_SETUP.md   ← COMPLETE REFERENCE
│  ├─ 📖 ANNOTATION_GUIDE.md          ← DETAILED GUIDE
│  ├─ 📖 QUICK_REFERENCE.md           ← QUICK LOOKUP
│  ├─ anno_infer_ft_pku_orig.py       ← MAIN SCRIPT
│  ├─ run_anno_ft_pku_orig.sh         ← QUICK START
│  ├─ anno_prompt.txt                 (Medical ethics template)
│  └─ ft_pku_orig_qwen3-8b/           (Output folder)
│     ├─ medsafety_labels.jsonl
│     ├─ medsafety_labels.csv
│     └─ annotation_stats.json
│
└─ data/
   └─ pku_anno_formatted_test.jsonl   (Input data)
```

---

## 🚀 Quick Start Paths

### Path 1: Just Run It (2 minutes)
```
1. Read: ANNOTATION_SETUP_COMPLETE.md (top section)
2. Run:  bash run_anno_ft_pku_orig.sh
3. Done: Check ft_pku_orig_qwen3-8b/
```

### Path 2: Understand First (30 minutes)
```
1. Read: ANNOTATION_SETUP_COMPLETE.md
2. Read: COMPLETE_WORKFLOW_VISUAL.md
3. Read: annotation/QUICK_REFERENCE.md
4. Run:  python anno_infer_ft_pku_orig.py --max_samples 10
5. Read: annotation/ANNOTATION_GUIDE.md
6. Run:  bash run_anno_ft_pku_orig.sh
```

### Path 3: Deep Dive (1 hour)
```
1. Read: ANNOTATION_WORKFLOW.md
2. Read: ANNOTATION_SETUP_COMPLETE.md
3. Read: annotation/ANNOTATION_GUIDE.md
4. Read: annotation/QUICK_REFERENCE.md
5. Study: Complete file structure
6. Run: python anno_infer_ft_pku_orig.py
7. Analyze: Output files and statistics
```

---

## 📋 What Each Document Covers

### ANNOTATION_SETUP_COMPLETE.md
- ✅ What was created
- ✅ Quick start (30 seconds)
- ✅ Output example
- ✅ Key differences from other scripts
- ✅ Complete workflow
- ✅ Next steps

### ANNOTATION_WORKFLOW.md
- ✅ End-to-end process
- ✅ Phase-by-phase breakdown
- ✅ Label format differences
- ✅ File organization
- ✅ Processing pipeline
- ✅ Key characteristics

### COMPLETE_WORKFLOW_VISUAL.md
- ✅ Visual diagrams
- ✅ Big picture overview
- ✅ Data flow diagram
- ✅ Script integration
- ✅ Three-phase process
- ✅ Label evolution

### annotation/README_ANNOTATION_SETUP.md
- ✅ Comprehensive overview
- ✅ Workflow explanation
- ✅ Script descriptions
- ✅ Input/output details
- ✅ Command-line options
- ✅ Troubleshooting

### annotation/ANNOTATION_GUIDE.md
- ✅ Overview and workflow
- ✅ Script documentation
- ✅ Input/output formats
- ✅ Annotation schema
- ✅ Harm types and severity
- ✅ Output file examples
- ✅ Statistics explanation
- ✅ Troubleshooting

### annotation/QUICK_REFERENCE.md
- ✅ TL;DR summary
- ✅ How it works
- ✅ Quick commands
- ✅ Output examples
- ✅ Annotation schema
- ✅ Expected statistics
- ✅ Troubleshooting

---

## 🎯 By Use Case

### "I want to run annotation now"
1. Read: `ANNOTATION_SETUP_COMPLETE.md` (Quick Start section)
2. Run: `bash run_anno_ft_pku_orig.sh`

### "I want to understand what's happening"
1. Read: `COMPLETE_WORKFLOW_VISUAL.md`
2. Read: `ANNOTATION_WORKFLOW.md`
3. Read: `annotation/QUICK_REFERENCE.md`

### "I need detailed instructions"
1. Read: `annotation/ANNOTATION_GUIDE.md`
2. Read: `annotation/README_ANNOTATION_SETUP.md`
3. Run: `python anno_infer_ft_pku_orig.py`

### "I want to customize the script"
1. Read: `annotation/ANNOTATION_GUIDE.md` (Command-line Options)
2. Read: `annotation/anno_infer_ft_pku_orig.py` (Code comments)
3. Modify and run

### "I want to compare with other methods"
1. Read: `annotation/QUICK_REFERENCE.md` (Differences section)
2. Compare with: `train_orig/DIFFERENCES_TRAIN_VS_TRAIN_ORIG.md`

---

## 📚 Learning Order (Recommended)

For new users, read in this order:

1. **Start Here** (5 min)
   - `ANNOTATION_SETUP_COMPLETE.md`

2. **Understand Process** (10 min)
   - `COMPLETE_WORKFLOW_VISUAL.md`

3. **Learn Commands** (5 min)
   - `annotation/QUICK_REFERENCE.md`

4. **Details & Examples** (15 min)
   - `annotation/ANNOTATION_GUIDE.md`

5. **Run It** (varies)
   - `bash run_anno_ft_pku_orig.sh`

6. **Reference** (as needed)
   - `annotation/README_ANNOTATION_SETUP.md`

---

## 🔍 Quick Lookup by Topic

### Running the Script
- `annotation/QUICK_REFERENCE.md` → Commands section
- `annotation/ANNOTATION_GUIDE.md` → Quick Start section

### Understanding Output
- `annotation/QUICK_REFERENCE.md` → Output Examples
- `annotation/ANNOTATION_GUIDE.md` → Output Format Example

### Command Options
- `annotation/QUICK_REFERENCE.md` → Commands section
- `annotation/ANNOTATION_GUIDE.md` → Command-line Options

### Annotation Schema
- `annotation/QUICK_REFERENCE.md` → Annotation Schema
- `ANNOTATION_WORKFLOW.md` → Label Format Differences

### Troubleshooting
- `annotation/QUICK_REFERENCE.md` → Troubleshooting
- `annotation/ANNOTATION_GUIDE.md` → Troubleshooting

### Comparing Scripts
- `annotation/README_ANNOTATION_SETUP.md` → Key Differences
- `annotation/QUICK_REFERENCE.md` → Relationship to train_orig
- `train_orig/DIFFERENCES_TRAIN_VS_TRAIN_ORIG.md` → Format comparison

---

## 💡 Key Concepts Quick Reference

### The Three Phases
1. **Train** (done): `train_orig/` creates LoRA adapter
2. **Annotate** (new): `annotation/` uses adapter to label data
3. **Evaluate** (next): `train_orig/evaluate.py` measures quality

### Annotation Levels
- Binary: harmless/harmful
- Severity: mild/moderate/severe (if harmful)
- Harm Types: 6 specific categories
- Explanation: Concise reasoning

### Output Formats
- JSONL: Machine-readable, full details
- CSV: Human-readable, tabular
- JSON Stats: Summary statistics

---

## 📞 Need Help?

### Script Issues
→ See `annotation/ANNOTATION_GUIDE.md` → Troubleshooting

### Command Questions
→ See `annotation/QUICK_REFERENCE.md` → Commands

### Understanding Process
→ See `COMPLETE_WORKFLOW_VISUAL.md` → Diagrams

### Setup Questions
→ See `ANNOTATION_SETUP_COMPLETE.md` → Key Features

### Detailed Guide
→ See `annotation/ANNOTATION_GUIDE.md` → Complete documentation

---

## ✅ Summary

You have access to comprehensive documentation organized by:
- **Skill level**: Quick reference to detailed guide
- **Use case**: Running script vs understanding details
- **Topic**: Commands, output, troubleshooting
- **Format**: Text guides, visual diagrams, code examples

**Start with**: `ANNOTATION_SETUP_COMPLETE.md` (5 minutes)
**Then run**: `bash run_anno_ft_pku_orig.sh`
**Reference as needed**: Other documents for details

