# Model Evaluation & Error Analysis Guide

This guide shows you how to:
1. ✅ See how your model performs on **seen data** (training set)
2. ✅ See how your model performs on **unseen data** (test set)
3. ✅ Identify **exactly what the model gets wrong**
4. ✅ Make predictions on **new unlabeled data**
5. ✅ Iteratively improve based on errors

---

## Quick Start

After training a model, run this one-liner:

```bash
./scripts/eval/evaluate_model_workflow.sh outputs/artifacts/all_weighted/2025-11-13T22-18-43Z random_forest
```

This will create:
- **Error CSVs**: Lists of false positives and false negatives
- **Visualizations**: Confusion matrices and probability distributions
- **Statistics**: Detailed JSON with all metrics

---

## Step-by-Step Workflow

### Step 1: Train Your Model

```bash
flybehavior-response train \
  --data-csv data/all_envelope_rows_wide.csv \
  --labels-csv data/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --n-pcs 6 \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 150 \
  --artifacts-dir outputs/artifacts/my_experiment
```

This creates a directory like: `outputs/artifacts/my_experiment/2025-11-13T22-18-43Z/`

---

### Step 2: Analyze Errors on SEEN Data (Training Set)

The training set predictions are already saved! Check:

```bash
# View all training predictions
cat outputs/artifacts/my_experiment/*/predictions_random_forest_train.csv

# Count errors in training set
grep "False" outputs/artifacts/my_experiment/*/predictions_random_forest_train.csv | wc -l
```

**What to look for:**
- **Few errors = good** (model learned the patterns)
- **Zero errors = suspicious** (might be overfitting, but RF at 100% train accuracy is normal)

---

### Step 3: Analyze Errors on UNSEEN Data (Test Set)

This is the **most important** metric. Run:

```bash
python scripts/analysis/analyze_model_errors.py \
  --run-dir outputs/artifacts/my_experiment/2025-11-13T22-18-43Z \
  --model random_forest \
  --output-dir outputs/artifacts/my_experiment/2025-11-13T22-18-43Z/error_analysis
```

**Output:**
```
TEST SET:
  Accuracy:  95.7%
  Precision: 96.4%
  Recall:    93.1%
  F1 Score:  94.7%

  Confusion Matrix:
    True Negatives:  40
    False Positives:  1 ⚠️
    False Negatives:  2 ⚠️
    True Positives:  27
```

**Files created:**
- `errors_random_forest_test_false_positives.csv` - What did model wrongly call "responder"?
- `errors_random_forest_test_false_negatives.csv` - What responders did model miss?
- `confusion_matrices_random_forest.png` - Visual comparison across splits
- `probability_distributions_random_forest.png` - Shows confidence for each error type

---

### Step 4: Review What the Model Got Wrong

Open the error CSVs:

```bash
# See the false positives (model said "Yes" but true answer was "No")
cat outputs/artifacts/my_experiment/*/error_analysis/errors_random_forest_test_false_positives.csv

# See the false negatives (model said "No" but true answer was "Yes")
cat outputs/artifacts/my_experiment/*/error_analysis/errors_random_forest_test_false_negatives.csv
```

**Each CSV contains:**
- `dataset`, `fly`, `fly_number`, `trial_type`, `trial_label` - Identifies the sample
- `user_score_odor` - True label (0 or 1-5)
- `user_score_odor_intensity` - Strength of response (0-5)
- `predicted_label` - What the model predicted (0 or 1)
- `prob_reaction` - Model's confidence (0.0 to 1.0)
- `correct` - False (it's an error)
- `sample_weight` - How much this sample was weighted during training
- **All your features** - `global_max`, `AUC-During`, etc.

**Example false negative:**
```csv
dataset,fly,fly_number,trial_type,trial_label,user_score_odor,user_score_odor_intensity,predicted_label,prob_reaction,correct,sample_weight,global_max,local_max,...
opto,september_09_fly_3,3,testing,t2,1,2,0,0.38,False,2.0,1.5,1.8,...
```

This tells you:
- Trial `september_09_fly_3 / t2` was labeled as a weak responder (intensity=2)
- Model predicted "No" with only 38% confidence
- It was wrong! This WAS a responder
- Looking at features, you can see why the model struggled

---

### Step 5: Visualize Errors

Open the PNG files:

```bash
# On Linux with display
xdg-open outputs/artifacts/my_experiment/*/error_analysis/confusion_matrices_random_forest.png
xdg-open outputs/artifacts/my_experiment/*/error_analysis/probability_distributions_random_forest.png

# Or copy to your desktop/view in file manager
```

**Confusion Matrices** show:
- How many in each category (TP, TN, FP, FN)
- Comparison across train/validation/test

**Probability Distributions** show:
- Green: True Positives - should be high confidence (>0.7)
- Blue: True Negatives - should be low confidence (<0.3)
- Orange: False Positives - shows how confident model was when wrong
- Red: False Negatives - shows if model was uncertain (near 0.5) or confidently wrong

---

### Step 6: Predict on NEW Unlabeled Data

You have **520 unlabeled traces**. Make predictions:

```bash
# Predict on ALL your data (including unlabeled)
flybehavior-response predict \
  --data-csv data/all_envelope_rows_wide.csv \
  --model-path outputs/artifacts/my_experiment/*/model_random_forest.joblib \
  --output-csv predictions_all_900_traces.csv
```

**Output CSV contains:**
- All original columns from your data
- `predicted_label` - 0 or 1
- `prob_reaction` - Confidence (0.0 to 1.0)

**Filter for high-confidence predictions:**
```bash
# See traces model is VERY confident are responders (>90% confidence)
awk -F',' '$NF > 0.9' predictions_all_900_traces.csv | head

# See traces model is uncertain about (40-60% confidence)
awk -F',' '$NF > 0.4 && $NF < 0.6' predictions_all_900_traces.csv
```

**Strategy for labeling remaining 520 traces:**
1. **Prioritize uncertain predictions** (prob near 0.5) - these are hardest cases
2. **Sample from high-confidence correct** - verify model isn't systematically wrong
3. **Focus on edge cases** - weak responses (intensity 1-2) that model missed

---

## Understanding the Results

### Your Current Random Forest Results:

**Test Set (Unseen Data):**
- Accuracy: 95.7% - **Excellent!**
- Recall: 93.1% - Catches 27/29 responders
- Precision: 96.4% - Only 1 false positive
- **2 false negatives, 1 false positive**

**What this means:**
- ✅ Model generalizes very well to unseen data
- ✅ Almost never calls non-responders "responders" (1/41)
- ⚠️ Misses 2 out of 29 actual responders (6.9% miss rate)

**Training Set (Seen Data):**
- Accuracy: 100% - Perfect
- This is normal for Random Forest with enough trees

**Gap between train (100%) and test (95.7%):**
- 4.3% gap is **acceptable** for RF
- Shows slight overfitting but not severe
- Can reduce with `--rf-max-depth 8` if needed

---

## Common Questions

### Q: How do I know if my model is good?

**Look at TEST set metrics:**
- Accuracy > 90% = Good
- Accuracy > 95% = Excellent (you have 95.7%!)
- Recall > 90% = Catching most positives (you have 93.1%)
- Precision > 90% = Few false positives (you have 96.4%)

**Your Random Forest is EXCELLENT.** ✅

---

### Q: What if I see a pattern in the errors?

**Example patterns to look for:**

1. **All false negatives are weak responses (intensity 1-2)**
   - Model struggles with weak signals
   - Solution: Add more weak response examples, or use `--rf-class-weights "0:1.0,1:3.0"`

2. **False positives have high `local_max` but low `AUC-During`**
   - Model confused by brief spikes
   - Solution: Add features that capture sustained response

3. **Errors are from specific flies or datasets**
   - Individual variation
   - Solution: Make sure training includes diverse flies

**Tell me the pattern and I'll suggest fixes!**

---

### Q: How do I improve the model?

**Three main approaches:**

1. **Add more labeled data** (you have 520 unlabeled)
   - Focus on cases similar to current errors
   - Label uncertain predictions first

2. **Tune hyperparameters**
   ```bash
   # Try different RF settings
   --rf-n-estimators 200
   --rf-max-depth 10
   --rf-class-weights "0:1.0,1:2.0"
   ```

3. **Engineer better features**
   - If errors show specific patterns, add features to capture them
   - Example: If missing weak sustained responses, add "sustained_extension_duration"

---

## Iterative Improvement Workflow

```
1. Train model → 2. Analyze errors → 3. Identify pattern → 4. Fix → 5. Retrain

Examples:
  Error pattern: "Missing weak responses (intensity 1-2)"
  → Fix: Use --rf-class-weights "0:1.0,1:3.0"

  Error pattern: "FP on trials with brief spikes"
  → Fix: Add feature "peak_duration" to distinguish spikes from sustained

  Error pattern: "Model uncertain on 10% of test set"
  → Fix: Label more traces similar to uncertain cases, retrain
```

---

## Example Session

```bash
# 1. Train
flybehavior-response train --model random_forest --n-pcs 6 \
  --rf-class-weights balanced --artifacts-dir exp1

# 2. Analyze
python scripts/analysis/analyze_model_errors.py --run-dir exp1/*/ --model random_forest

# 3. Review errors
cat exp1/*/error_analysis/errors_random_forest_test_false_negatives.csv

# Output shows: Both FN are weak responses (intensity 1-2) with low prob_reaction (0.38, 0.42)
# Pattern: Model not sensitive enough to weak responses

# 4. Fix: Increase weight on positive class
flybehavior-response train --model random_forest --n-pcs 6 \
  --rf-class-weights "0:1.0,1:3.0" --artifacts-dir exp2

# 5. Compare
python scripts/analysis/analyze_model_errors.py --run-dir exp2/*/ --model random_forest
# Check if FN decreased!
```

---

## Predicting on Unlabeled Data

```bash
# Make predictions on all 900 traces
flybehavior-response predict \
  --data-csv /path/to/all_900_traces.csv \
  --model-path exp1/*/model_random_forest.joblib \
  --output-csv predictions_900.csv

# Sort by confidence to find uncertain cases
sort -t',' -k last -n predictions_900.csv > predictions_sorted.csv

# Filter for unlabeled traces only (those without user_score_odor in original data)
# You'll need to merge with your labels file to identify which are unlabeled
```

---

## Summary

**To evaluate your model:**
1. Run `python scripts/analysis/analyze_model_errors.py --run-dir <run-dir> --model random_forest`
2. Check test set metrics (accuracy, recall, precision)
3. Review error CSVs to see what went wrong
4. Look at visualizations for patterns
5. Tell me what patterns you see, and I'll help you fix them!

**Your current Random Forest (95.7% test accuracy) is excellent!** But if you want to push to 97-98%, label more of those 520 traces and retrain.
