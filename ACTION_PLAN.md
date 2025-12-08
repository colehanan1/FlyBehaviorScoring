# ACTION PLAN: Fix Label Iteration Workflow

## Current Branch
✅ `claude/label-iteration-fix-01Y6UtrPonpiMPTVA53R6sAr`

---

## ANSWER THESE QUESTIONS FIRST

Before we proceed, I need this info from you:

### Q1: Label Changes
**How many labels did you change?** (Approximately)
- [ ] 1-5 labels
- [ ] 5-10 labels
- [ ] 10-20 labels
- [ ] More than 20

### Q2: Which Trials?
**Do you remember which specific trials you changed?**
(Or can you compare your current labels CSV to an old backup?)

### Q3: Old Labels Backup
**Do you have the OLD labels file saved somewhere?**
- Path to OLD labels: `_____________________________________`
- Path to NEW (corrected) labels: `/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv`

If you don't have old labels saved, that's OK - we'll proceed without comparison.

### Q4: Why Did You Change Labels?
**What made you decide to change these labels?**
- [ ] Looked at model errors and disagreed with my original labels
- [ ] Noticed systematic pattern (e.g., threshold was wrong)
- [ ] Reviewed raw data/videos independently
- [ ] Other: _______________

### Q5: Did Train/Test Split Change?
**When you said "it changed the layout of the data", do you mean:**
- [ ] A) Just the label VALUES changed (0→1 or 1→0), but same trials in train/test
- [ ] B) Different trials ended up in train vs test sets (this is BAD)
- [ ] C) I don't know / not sure

---

## COMMANDS TO RUN (In Order)

Once you answer the questions above, run these commands in your LOCAL environment (not here):

### Step 1: Document Your Label Changes

```bash
# Open the label correction log and fill it out
nano label_correction_log.md

# Fill in:
# - Date
# - How many labels changed
# - Which trials (if you know)
# - Why you changed them
# - Old vs new label for each (if you have old backup)
```

### Step 2: Compare Old vs New Labels (If you have old backup)

If you have the old labels file:

```bash
# This will show you which labels changed
diff /path/to/OLD_labels.csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv

# Or for better comparison:
python compare_labels.py \
  /path/to/OLD_labels.csv \
  /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv
```

(I can create `compare_labels.py` script if you have old backup)

### Step 3: Retrain Model with Corrected Labels

**CRITICAL: Use seed=42 to maintain same train/test split!**

```bash
cd /home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring

# Activate your environment
conda activate flybehavior

# Retrain with SAME configuration but NEW labels
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 100 \
  --n-pcs 12 \
  --seed 42 \
  --artifacts-dir artifacts/rf_corrected_labels_v1
```

**This will take ~30 seconds to run.**

### Step 4: Analyze the New Errors

```bash
# Find the latest run directory
ls -ltr artifacts/rf_corrected_labels_v1/

# Look at the predictions (replace TIMESTAMP with actual timestamp)
python analyze_errors.py artifacts/rf_corrected_labels_v1/TIMESTAMP/predictions_random_forest_test.csv
```

**Questions to ask yourself:**
- Are the NEW errors more ambiguous than before?
- Or are they obvious mistakes (suggesting bad label corrections)?
- How many errors now vs before?

### Step 5: Compare Before vs After (If you have old metrics)

```bash
# Compare to your original run
python compare_label_versions.py \
  artifacts/rf_tuned/2025-11-14T01-47-01Z/metrics.json \
  artifacts/rf_corrected_labels_v1/TIMESTAMP/metrics.json
```

This will show you:
- Did test accuracy improve or get worse?
- Did overfitting gap change?
- Which metrics changed?

### Step 6: Decide Next Steps

Based on results:

#### If Test Accuracy IMPROVED (or stayed within 1%)
✅ **Your label corrections were good!**

Next steps:
```bash
# Try hyperparameter tuning
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 200 \
  --rf-max-depth 20 \
  --n-pcs 15 \
  --seed 42 \
  --artifacts-dir artifacts/rf_tuned_v2
```

#### If Test Accuracy DROPPED >2%
⚠️ **Possible issues:**

Option A: Use Cross-Validation instead
```bash
# Switch to CV to avoid test set contamination
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 200 \
  --n-pcs 12 \
  --cv 5 \
  --seed 42 \
  --artifacts-dir artifacts/rf_cv_final
```

**Report CV accuracy ± std dev instead of test accuracy**

Option B: Review the new errors carefully
- Are they genuinely ambiguous? → Good, you improved labels
- Are they obvious mistakes? → Bad, label corrections were inconsistent

---

## SUMMARY: What You'll Run

**Minimum commands (if you don't have old labels backup):**

```bash
# 1. Retrain
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 100 \
  --n-pcs 12 \
  --seed 42 \
  --artifacts-dir artifacts/rf_corrected_labels_v1

# 2. Check new errors
python analyze_errors.py artifacts/rf_corrected_labels_v1/LATEST_TIMESTAMP/predictions_random_forest_test.csv

# 3. Decide: tune hyperparameters OR switch to CV
```

**Full commands (if you have old labels):**

```bash
# 1. Compare old vs new labels
python compare_labels.py OLD_labels.csv NEW_labels.csv

# 2. Document changes in label_correction_log.md

# 3. Retrain
flybehavior-response train ... (same as above)

# 4. Compare metrics
python compare_label_versions.py OLD_metrics.json NEW_metrics.json

# 5. Analyze new errors
python analyze_errors.py ...

# 6. Decide next steps
```

---

## Files Available to Help You

In this repository, you now have:

1. ✅ `analyze_errors.py` - Analyze misclassified test samples
2. ✅ `compare_label_versions.py` - Compare model performance before/after label changes
3. ✅ `label_correction_log.md` - Template to document your changes
4. ✅ `tune_random_forest.py` - Hyperparameter tuning experiments
5. ✅ `ITERATIVE_LABELING_GUIDE.md` - Full guide on label quality
6. ✅ `OVERFITTING_ANALYSIS.md` - Guide to RF overfitting and max_features
7. ✅ `RECOMMENDATIONS.md` - Original recommendations for improving from 93.6%

---

## Quick Reference Card

Copy this and pin it:

```bash
# RETRAIN (after label changes)
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 100 \
  --n-pcs 12 \
  --seed 42 \
  --artifacts-dir artifacts/rf_corrected_labels_v1

# ANALYZE ERRORS
python analyze_errors.py artifacts/rf_corrected_labels_v1/*/predictions_random_forest_test.csv

# TUNE HYPERPARAMETERS (after verifying labels are good)
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 200 \
  --rf-max-depth 20 \
  --n-pcs 15 \
  --seed 42 \
  --artifacts-dir artifacts/rf_tuned_v2

# USE CV (to avoid test set contamination)
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 200 \
  --n-pcs 12 \
  --cv 5 \
  --seed 42 \
  --artifacts-dir artifacts/rf_cv_final
```

---

## ANSWER THE QUESTIONS ABOVE, THEN START WITH STEP 3!
