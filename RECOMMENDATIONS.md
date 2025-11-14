# Analysis of Your Random Forest Results and Next Steps

## Your Current Results: 93.6% Test Accuracy ✅

**This is actually EXCELLENT!** Here's why:

- ✅ **Proper hybrid splitting**: 370 train (32 flies) / 110 test (8 flies)
- ✅ **No fly leakage**: Test flies completely separate from training
- ✅ **Better than previous models**: Your RF beats the 90.9% XGBoost from the earlier analysis
- ✅ **Good generalization**: Only 7 errors out of 110 test samples

### Confusion Matrix Breakdown
```
Test Set (110 samples):
  True Negatives:  64  ← Correctly identified non-responders
  False Positives:  3  ← Said "reaction" but none happened
  False Negatives:  4  ← Missed actual reactions
  True Positives:  39  ← Correctly identified responders
```

## Your Questions Answered

### 1. "Is using LogReg stupid compared to Random Forest?"

**You're RIGHT!** For your problem, Random Forest is likely superior because:

- **Non-linear patterns**: Proboscis extension traces have complex temporal dynamics
- **Feature interactions**: RF automatically learns which combinations matter (e.g., "high local_max_during AND low global_max")
- **Robustness**: Less sensitive to feature scaling and outliers
- **Your results prove it**: 93.6% RF vs 90.9% LogReg/XGBoost

Logistic regression assumes linear decision boundaries, which is too simplistic for biological time series data.

### 2. "Should I check labels to see if they are the problem?"

**YES - THIS IS YOUR #1 PRIORITY!** Here's why:

With 110 test samples and 7 errors:
- To reach 95% accuracy, you can only afford **5-6 errors**
- You need to eliminate just **1-2 errors** to get there
- If even 2 of your 7 errors are labeling mistakes, fixing them gets you to 95.5%!

**Action:** Run the error analysis script I created:
```bash
python analyze_errors.py artifacts/rf_tuned/2025-11-14T01-47-01Z/predictions_random_forest_test.csv
```

This will show you:
- Which specific trials were misclassified
- Which flies are hardest to predict
- Whether errors cluster around uncertain cases (prob ≈ 0.5)
- Feature patterns in misclassified samples

**What to look for:**
- Samples with `prob_reaction` near 0.5 → genuinely ambiguous
- Samples with very high/low confidence but wrong → potential label errors
- Multiple errors from the same fly → that fly might have unusual behavior patterns

### 3. "Any other advice?"

## Recommended Next Steps (Priority Order)

### Priority 1: Error Analysis & Label Quality Review (2-3 days)
**Expected gain: +1-3% accuracy**

1. **Run error analysis:**
   ```bash
   python analyze_errors.py artifacts/rf_tuned/2025-11-14T01-47-01Z/predictions_random_forest_test.csv
   ```

2. **Manually review the 7 misclassified cases:**
   - Look at the actual trace data for these trials
   - Ask: "Would I have labeled this the same way?"
   - Get a second opinion on ambiguous cases
   - Check for data quality issues (noisy traces, artifacts)

3. **Create label confidence scores:**
   - Mark labels as "certain", "likely", or "uncertain"
   - Consider removing highly ambiguous cases from training
   - Or use sample weighting to down-weight uncertain labels

### Priority 2: Hyperparameter Tuning (1 day)
**Expected gain: +0.5-2% accuracy**

Your current config:
- 100 trees
- No max depth (trees grow fully)
- 12 PCA components

**Run systematic tuning:**
```bash
python tune_random_forest.py
```

This will test:
- More trees (200, 300) → better ensemble averaging
- Controlled depth (15, 20) → reduce overfitting
- More PCA components (15, 20) → capture more trace variance

**Quick manual tests you can try NOW:**
```bash
# More trees
flybehavior-response train \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model random_forest \
  --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
  --rf-class-weights balanced \
  --rf-n-estimators 300 \
  --rf-max-depth 20 \
  --artifacts-dir artifacts/rf_tuned_v2 \
  --n-pcs 15
```

### Priority 3: Feature Importance Analysis (1 day)
**Expected gain: Better understanding, potential +0.5-1% accuracy**

Understand which features matter most:

```python
import joblib
import numpy as np
import pandas as pd

# Load your trained model
pipeline = joblib.load("artifacts/rf_tuned/2025-11-14T01-47-01Z/model_random_forest.joblib")
rf_model = pipeline.named_steps['model']

# Get feature importances
importances = rf_model.feature_importances_

# After PCA, features are PC1, PC2, ..., PC12, plus your 8 engineered features
feature_names = [f"PC{i+1}" for i in range(12)] + [
    "global_max", "local_min", "local_max", "local_max_before",
    "local_max_during", "local_max_during_over_global_min",
    "AUC-During", "Peak-Value"
]

# Sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df)
```

This tells you:
- Which engineered features are most predictive
- Whether you need more/fewer PCA components
- If you can drop low-importance features (simplify model)

### Priority 4: Ensemble Methods (if needed)
**Expected gain: +0.5-1.5% accuracy**

If you're still below 95% after the above:

**Stacking ensemble:**
Train multiple models and combine predictions:
```bash
# Train all three models on same data
flybehavior-response train \
  --model random_forest,logreg,xgboost \
  --rf-n-estimators 300 \
  --n-pcs 15 \
  ...
```

Then combine predictions (weighted average or meta-learner).

### Priority 5: Collect More Data (if needed)
**Expected gain: +2-4% accuracy**

Only if above steps don't get you to 95%:

**Current bottleneck:**
- Only 8 flies in your test set
- High variance in test accuracy
- Some flies might just be inherently harder to predict

**Target:**
- 800-1000 total trials (you have 480 now)
- 50-60 flies total (you have ~40 now)
- Focus on flies with intermediate response rates (harder cases)

## Why NOT Deep Learning (Yet)?

The previous analysis suggested 1D CNNs or LSTMs. **I disagree** for now because:

1. **You have limited data**: 480 trials is borderline too small for deep learning
2. **RF is already working well**: 93.6% is strong performance
3. **Interpretability matters**: RF lets you understand feature importance
4. **Faster iteration**: RF trains in seconds, DL takes hours to tune
5. **Overfitting risk**: DL would likely overfit with only 370 train samples

**When to try DL:**
- If you collect 800+ trials
- If RF plateaus below 95% despite tuning
- If you need to squeeze out the last 1-2% of accuracy

## Summary Action Plan

**Week 1:**
1. ✅ Run error analysis script
2. ✅ Manually review 7 misclassified cases
3. ✅ Fix any obvious label errors
4. ✅ Re-train and measure improvement

**Week 2:**
1. ✅ Run hyperparameter tuning experiments
2. ✅ Analyze feature importances
3. ✅ Select best configuration
4. ✅ Document final model performance

**Week 3 (if needed):**
1. ✅ Implement ensemble if still below 95%
2. ✅ Plan data collection if necessary
3. ✅ Write up final validation results

## Expected Outcome

**Conservative estimate:**
- Error analysis + label fixes: 93.6% → 94.5%
- Hyperparameter tuning: 94.5% → 95.0%
- Feature optimization: 95.0% → 95.5%

**Optimistic (if 2-3 label errors found):**
- 93.6% → 96%+ directly

## You're On the Right Track! 🎯

Your instinct to use Random Forest was correct. Focus on understanding and fixing those 7 errors before anything else. You're only 1.4% away from 95% - that's just **1-2 additional correct predictions** out of 110!
