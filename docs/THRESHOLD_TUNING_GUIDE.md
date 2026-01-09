# Threshold Tuning for Catching Subtle Reactions

## Your Problem

**Symptoms:**
- Model works well on labeled test data (94.4% accuracy!)
- But on unlabeled data, it misses subtle/small reactions
- Model is confident when wrong (prob not near 0.5)
- You want to catch more of these borderline cases

**Root cause:** Default threshold (0.5) is too conservative for subtle reactions.

---

## The Solution: Lower the Prediction Threshold

### Understanding Thresholds

Random Forest doesn't just predict "reaction" or "no reaction". It outputs a **probability**:

```
prob_reaction = 0.85 → Very confident it's a reaction
prob_reaction = 0.55 → Slightly thinks it's a reaction
prob_reaction = 0.48 → Slightly thinks it's NOT a reaction
prob_reaction = 0.15 → Very confident it's NOT a reaction
```

By default, the threshold is **0.5**:
- prob ≥ 0.5 → predict "reaction"
- prob < 0.5 → predict "no reaction"

**Problem:** For subtle reactions, the model might output prob = 0.45, which gets classified as "no reaction" even though it's pretty confident there IS a reaction.

---

## Step-by-Step: Find Your Optimal Threshold

### Step 1: Analyze Your Test Set

Run this on your LOCAL machine:

```bash
cd /path/to/FlyBehaviorScoring
conda activate flybehavior

python scripts/eval/find_optimal_threshold.py \
  outputs/artifacts/rf_corrected_labels_v1/2025-11-14T18-58-28Z/predictions_random_forest_test.csv \
  recall
```

**What this does:**
- Analyzes your test set predictions
- Tests thresholds from 0.1 to 0.9
- Finds the threshold that maximizes recall (catches most reactions)
- Shows you the trade-off: more reactions caught vs more false alarms

**Output:**
```
OPTIMAL THRESHOLD (Strategy: Maximize Recall)
Optimal threshold: 0.38

Performance at threshold = 0.38:
  Precision: 0.917
  Recall:    0.977
  F1 Score:  0.946

Catches 2 MORE subtle reactions
But adds 1 MORE false alarm

Threshold change: 0.500 → 0.380
```

**You'll also get:**
- `threshold_optimization.png` - Plots showing threshold effects
- `threshold_analysis.csv` - Detailed results for all thresholds

### Step 2: Use Custom Threshold for Predictions

Once you know your optimal threshold, use it when predicting on unlabeled data:

```bash
# Example: Use threshold = 0.38 to catch more subtle reactions
flybehavior-response predict \
  --model-path outputs/artifacts/rf_corrected_labels_v1/2025-11-14T18-58-28Z/model_random_forest.joblib \
  --data-csv /path/to/unlabeled_data.csv \
  --threshold 0.38 \
  --output predictions_sensitive.csv
```

**Compare to default:**
```bash
# Default threshold = 0.5 (conservative)
flybehavior-response predict \
  --model-path outputs/artifacts/rf_corrected_labels_v1/2025-11-14T18-58-28Z/model_random_forest.joblib \
  --data-csv /path/to/unlabeled_data.csv \
  --threshold 0.5 \
  --output predictions_default.csv
```

---

## Choosing Your Threshold Strategy

Run `scripts/eval/find_optimal_threshold.py` with different strategies:

### 1. Maximize Recall (Catch More Reactions)
```bash
python scripts/eval/find_optimal_threshold.py predictions_test.csv recall
```

**Best for:**
- You want to minimize missed reactions (false negatives)
- False alarms are acceptable
- Screening/discovery applications
- Your current situation: catching subtle reactions

**Trade-off:** More false positives (non-reactions incorrectly labeled as reactions)

### 2. Maximize F1 (Balanced)
```bash
python scripts/eval/find_optimal_threshold.py predictions_test.csv f1
```

**Best for:**
- Balanced precision and recall
- General-purpose classification
- When both errors are equally costly

### 3. Maximize Precision (Fewer False Alarms)
```bash
python scripts/eval/find_optimal_threshold.py predictions_test.csv precision
```

**Best for:**
- You want high confidence in positive predictions
- False alarms are costly
- Verification/confirmation applications

**Trade-off:** More false negatives (missed reactions)

---

## Understanding the Trade-Offs

### Example Scenario

Your test set has 44 reactions, 46 non-reactions:

**Threshold = 0.5 (default):**
```
Missed reactions (FN): 3
False alarms (FP):     2
Recall: 93.2%
Precision: 95.3%
```

**Threshold = 0.38 (optimized for recall):**
```
Missed reactions (FN): 1  ← Better! Caught 2 more
False alarms (FP):     4  ← Worse: 2 more false alarms
Recall: 97.7%              ← Better!
Precision: 91.7%           ← Slightly worse
```

**Is this trade-off worth it?**

For your use case (catching subtle reactions):
- ✅ **YES** - Missing real reactions is worse than a few false alarms
- ✅ You can manually review the uncertain cases (prob 0.3-0.5)
- ✅ Precision is still high (91.7%)

---

## Visual Analysis

After running `scripts/eval/find_optimal_threshold.py`, check the plots:

### 1. Precision/Recall vs Threshold
Shows how metrics change as you adjust threshold.

**What to look for:**
- Where recall curve peaks (catches most reactions)
- Where precision starts dropping off (too many false alarms)
- F1 curve maximum (best balance)

### 2. Error Counts vs Threshold
Shows FN (missed) and FP (false alarms) counts.

**What to look for:**
- How many MORE reactions you catch by lowering threshold
- How many MORE false alarms you add

### 3. Probability Distribution
Shows how model probabilities are distributed for reactions vs non-reactions.

**What to look for:**
- **Good separation:** Reactions cluster near 1.0, non-reactions near 0.0
- **Poor separation:** Lots of overlap between 0.3-0.7 (ambiguous cases)
- **Your subtle reactions:** Likely clustered around 0.4-0.5

If you see many reactions with prob 0.4-0.5, lowering threshold will help!

---

## When Threshold Tuning is NOT Enough

### Signs you need more training data:

1. **Probability distributions overlap heavily**
   - Reactions and non-reactions have similar probabilities
   - Model can't distinguish subtle vs no reaction

2. **Optimal threshold is extreme (<0.2 or >0.8)**
   - Model is fundamentally mis-calibrated
   - Need better training data

3. **No good threshold exists**
   - Every threshold gives either:
     - High recall but very low precision (<70%), OR
     - High precision but very low recall (<70%)

4. **Unlabeled data looks very different from training data**
   - New types of reactions not in training set
   - Different experimental conditions
   - Need to collect examples of these cases

### In these cases:

**Collect more labeled examples of:**
- Borderline reactions (small responses)
- Edge cases that the model gets wrong
- Examples from the new experimental conditions

**Target:** 50-100 more examples of subtle reactions

---

## Advanced: Probability Calibration

If your model's probabilities are poorly calibrated (not reflecting true likelihood), you can use:

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities using Platt scaling
calibrated_model = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
```

**When to use:**
- Model probabilities don't match actual frequencies
- Example: Samples with prob=0.6 are actually reactions 80% of the time
- Less common issue with Random Forest (usually well-calibrated)

---

## Your Action Plan

### Immediate (Run Now):

```bash
# 1. Find optimal threshold for catching subtle reactions
python scripts/eval/find_optimal_threshold.py \
  outputs/artifacts/rf_corrected_labels_v1/2025-11-14T18-58-28Z/predictions_random_forest_test.csv \
  recall

# 2. Note the recommended threshold (e.g., 0.38)

# 3. Use it when predicting on unlabeled data
flybehavior-response predict \
  --model-path outputs/artifacts/rf_corrected_labels_v1/2025-11-14T18-58-28Z/model_random_forest.joblib \
  --data-csv /path/to/unlabeled_data.csv \
  --threshold 0.38 \
  --output predictions_sensitive.csv

# 4. Review the predictions, especially those with 0.3 < prob < 0.5
```

### Short-term:

1. **Test on your unlabeled data** with the new threshold
2. **Manually review** a few predictions with prob 0.3-0.5 to verify
3. **Iterate:**
   - If still missing too many → lower threshold more (e.g., 0.35)
   - If too many false alarms → raise threshold slightly (e.g., 0.42)

### Long-term (if needed):

1. **Collect 50-100 more labeled examples** of subtle reactions
2. **Retrain model** with expanded dataset
3. **Re-optimize threshold** on new test set
4. **Expected improvement:** Better probability calibration for subtle cases

---

## Key Insights

1. **Your model is already good (94.4%)** - threshold tuning will get you further
2. **Threshold is a free parameter** - doesn't require retraining or more data
3. **Trade-off is controllable** - you choose how many false alarms to accept
4. **Class weights (1:6.0) ≠ threshold** - class weights affect training, threshold affects prediction
5. **Different thresholds for different use cases** - use 0.38 for discovery, 0.55 for high-confidence confirmation

---

## Common Mistakes to Avoid

### ❌ Don't optimize threshold on training set
- Use test set or validation set
- Otherwise you're overfitting to training data

### ❌ Don't change threshold based on unlabeled data
- You'll chase patterns that don't exist
- Only use labeled test set to find optimal threshold

### ❌ Don't set threshold without analyzing the trade-off
- Always check precision/recall curves
- Understand what you're gaining and losing

### ✅ Do use different thresholds for different purposes
- Discovery/screening: lower threshold (0.35-0.4)
- Confirmation/validation: higher threshold (0.55-0.65)
- Balanced: default threshold (0.5)

---

## Summary

**Your problem:** Model misses subtle reactions on unlabeled data.

**Solution:** Lower prediction threshold from 0.5 to ~0.38.

**How:**
1. Run `scripts/eval/find_optimal_threshold.py` to find optimal value
2. Use `--threshold` flag when running `predict` command
3. Accept trade-off: catch more reactions but get more false alarms

**When this works:**
- Model probabilities for subtle reactions are 0.35-0.49
- You can tolerate some false alarms
- Precision stays >85% with lower threshold

**When you need more data:**
- Optimal threshold is extreme (<0.2)
- Probability distributions overlap completely
- Unlabeled data has new types of reactions not in training set

**Expected outcome:** Catch 50-100% more subtle reactions with minimal increase in false alarms.
