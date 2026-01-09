# Optimizing Models to Reduce False Negatives

This guide explains how to use the new features added to reduce false negatives in your fly behavior classification models.

## Problem

Your logistic regression at 6 PCs was producing **all false negatives** (missing actual responders). This happens when the model is too conservative and needs to be more sensitive to the positive class.

## Solutions Implemented

### 1. Class Weighting for Logistic Regression

**What it does**: Increases the importance of responder samples during training, making the model more sensitive to positive class.

**Usage**:
```bash
# Use balanced weights (sklearn automatically computes optimal weights)
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model logreg \
  --n-pcs 6 \
  --logreg-class-weights balanced

# Or use custom weights (format: class:weight)
# Example: Give responders 3x weight to reduce false negatives
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model logreg \
  --n-pcs 6 \
  --logreg-class-weights "0:1.0,1:3.0"
```

**Recommendation for your case**: Start with `--logreg-class-weights "0:1.0,1:2.0"` or `--logreg-class-weights "0:1.0,1:3.0"` to make responders 2-3x more important.

### 2. Random Forest Classifier

**What it does**: Random Forest can capture non-linear patterns that logistic regression misses. It's an ensemble of decision trees that can handle complex relationships.

**Usage**:
```bash
# Basic Random Forest
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6

# Random Forest with class weights to reduce false negatives
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6 \
  --rf-class-weights "0:1.0,1:3.0"

# Tune Random Forest hyperparameters
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6 \
  --rf-n-estimators 200 \
  --rf-max-depth 10 \
  --rf-class-weights balanced
```

**Parameters**:
- `--rf-n-estimators`: Number of trees (default: 100). More trees = better but slower. Try 100-500.
- `--rf-max-depth`: Maximum tree depth (default: None = unlimited). Limit to 5-20 to prevent overfitting.
- `--rf-class-weights`: Class weights, same format as logreg. Use "balanced" or "0:1.0,1:3.0".

### 3. Compare Multiple Models

**Train all models** to see which performs best:
```bash
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model all \
  --n-pcs 6 \
  --logreg-class-weights "0:1.0,1:2.0" \
  --rf-class-weights "0:1.0,1:2.0" \
  --artifacts-dir outputs/artifacts/experiment_1
```

This trains:
- LDA (Linear Discriminant Analysis)
- Logistic Regression (with class weights)
- Random Forest (with class weights)
- MLP (Multi-Layer Perceptron)
- FP-Optimized MLP (False-Positive optimized)

## Recommended Workflow

### Step 1: Baseline with Class Weights
```bash
# Test logistic regression with different class weights
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model logreg \
  --n-pcs 6 \
  --logreg-class-weights "0:1.0,1:2.0" \
  --artifacts-dir outputs/artifacts/logreg_cw_2

flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model logreg \
  --n-pcs 6 \
  --logreg-class-weights "0:1.0,1:3.0" \
  --artifacts-dir outputs/artifacts/logreg_cw_3
```

### Step 2: Try Random Forest
```bash
# Random Forest with balanced weights
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6 \
  --rf-class-weights balanced \
  --rf-n-estimators 150 \
  --artifacts-dir outputs/artifacts/rf_balanced

# Random Forest with custom weights
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6 \
  --rf-class-weights "0:1.0,1:3.0" \
  --rf-n-estimators 150 \
  --artifacts-dir outputs/artifacts/rf_cw_3
```

### Step 3: Compare Results

After training, check the metrics in `outputs/artifacts/*/metrics.json`:
- **Recall**: Percentage of actual responders correctly identified. Higher = fewer false negatives.
- **Precision**: Percentage of predicted responders that are actually responders.
- **F1 Score**: Balance between precision and recall.
- **Confusion Matrix**: See exactly what your model got wrong.

```bash
# View metrics for each run
cat outputs/artifacts/logreg_cw_2/*/metrics.json
cat outputs/artifacts/rf_balanced/*/metrics.json
```

### Step 4: Adjust Based on Results

- If **recall is still too low** (many false negatives):
  - Increase class weights further (e.g., "0:1.0,1:4.0" or "0:1.0,1:5.0")
  - Try more PCs (8 or 10) for Random Forest
  - Use Random Forest with more trees (--rf-n-estimators 300)

- If **precision drops too much** (too many false positives):
  - Reduce class weights
  - Limit Random Forest depth (--rf-max-depth 10)
  - Use cross-validation to find optimal threshold

## Understanding Your MLP Overfitting

You mentioned: "mlp at 8pcs and 10000 neurons generalized great to train set but was very bad during testing"

**This is classic overfitting**. Solutions:
1. **Reduce network size**: 10,000 neurons is likely too many. Try 256, 512, or 1024.
2. **Use regularization**: The MLP already has L2 regularization (alpha parameter).
3. **Early stopping**: Already enabled, but you can tune `n_iter_no_change`.
4. **Use fewer PCs**: You said 6 PCs works best for logreg - try that for MLP too.
5. **Try the fp_optimized_mlp**: It uses early stopping and smaller architecture.

```bash
# Better MLP configuration
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model mlp \
  --n-pcs 6 \
  --best-params-json outputs/optuna_results/best_params.json  # If you have tuned params
```

## Class Weight Guidelines

**For reducing false negatives**:
- `"0:1.0,1:2.0"` - Moderate increase in sensitivity (good starting point)
- `"0:1.0,1:3.0"` - Significant increase in sensitivity
- `"0:1.0,1:5.0"` - Aggressive increase (use if still getting false negatives)
- `"balanced"` - Let sklearn compute optimal weights based on class distribution

**Trade-off**: Higher weights on class 1 (responders) will:
- ✅ Reduce false negatives (catch more responders)
- ⚠️ May increase false positives (classify non-responders as responders)

## Next Steps with Your Data

Since you mentioned "give me the metrics from running the code", I recommend:

1. **Run these experiments**:
```bash
# Experiment 1: Logreg with different weights
for weight in "0:1.0,1:2.0" "0:1.0,1:3.0" "balanced"; do
  flybehavior-response train \
    --data-csv /path/to/data.csv \
    --labels-csv /path/to/labels.csv \
    --model logreg \
    --n-pcs 6 \
    --logreg-class-weights "$weight" \
    --artifacts-dir "outputs/artifacts/logreg_weight_${weight//[^0-9]/}"
done

# Experiment 2: Random Forest comparison
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model random_forest \
  --n-pcs 6 \
  --rf-class-weights balanced \
  --artifacts-dir outputs/artifacts/rf_balanced

# Experiment 3: Compare all models
flybehavior-response train \
  --data-csv /path/to/data.csv \
  --labels-csv /path/to/labels.csv \
  --model all \
  --n-pcs 6 \
  --logreg-class-weights "0:1.0,1:2.0" \
  --rf-class-weights "0:1.0,1:2.0" \
  --artifacts-dir outputs/artifacts/all_models_cw2
```

2. **Share the results** and I can help you:
   - Interpret which model works best
   - Fine-tune hyperparameters
   - Develop an ensemble approach if needed
   - Optimize for your specific false negative tolerance

## About Ensemble/Bagging

You asked about "random forest for each variable and or bagging the model". Good news:

1. **Random Forest IS an ensemble/bagging method** - it already bags multiple decision trees
2. **You can combine models** using voting/stacking (I can add this if you want)
3. **Feature-specific forests** - Not typically better than single RF with all features, but we can try

Would you like me to implement a **voting classifier** that combines predictions from logreg, RF, and MLP?

## Summary

**Best approach to reduce false negatives**:
1. ✅ Use class weights: `--logreg-class-weights "0:1.0,1:2.0"` or higher
2. ✅ Try Random Forest: `--model random_forest --rf-class-weights balanced`
3. ✅ Compare multiple models: `--model all` with appropriate weights
4. ✅ Review metrics (recall, precision, F1) to find optimal balance

**What to avoid**:
- ❌ Don't use 10,000 neurons in MLP (causes overfitting)
- ❌ Don't rely on single model without trying alternatives
- ❌ Don't ignore class imbalance (use class weights!)

Run the experiments above and share your results. I'll help you identify the best model and tune it further!
