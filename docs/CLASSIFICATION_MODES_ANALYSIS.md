# Classification Modes Analysis: Complete Results

## Configuration
- **Data CSV**: `data/all_envelope_rows_wide.csv`
- **Labels CSV**: `data/scoring_results_opto_new_MINIMAL.csv`
- **Features**: AUC-During, TimeToPeak-During, Peak-Value
- **PCA Components**: 10
- **Random Seed**: 67
- **Cross-validation**: 5 folds
- **Models Tested**: LDA, Logistic Regression, Random Forest, MLP, FP-Optimized MLP

## Test 1: Binary Classification (0 vs 1-5) - Default Mode

This is the original classification mode where label 0 = non-reactive and labels 1-5 are combined as "reactive".

### Results

| Model | Train Acc | Test Acc | Test F1 | Overfitting Gap |
|-------|-----------|----------|---------|-----------------|
| **Logistic Regression** | 88.6% | **90.0%** | 0.871 | -1.4% ✓ |
| Random Forest | 100.0% | 87.5% | 0.833 | +12.5% |
| FP-Optimized MLP | 96.1% | 87.5% | 0.839 | +8.6% |
| LDA | 85.8% | 86.3% | 0.814 | -0.4% ✓ |
| MLP | 96.7% | 82.5% | 0.774 | +14.2% |

### Analysis

**Best Model**: **Logistic Regression** (90.0% test accuracy)
- Excellent generalization with slight negative overfitting gap (-1.4%)
- High F1 score (0.871) shows good balance between precision and recall
- Negative gap indicates the model generalizes even better to test data

**Key Insights**:
- Random Forest shows perfect training accuracy (100%) but overfits significantly (+12.5% gap)
- MLP shows highest overfitting (+14.2%) despite good training performance
- LDA and Logistic Regression show the best generalization (negative gaps)

---

## Test 2: Multi-class Classification (0-5) - 6 Classes

This mode preserves all 6 label intensities as separate classes (0, 1, 2, 3, 4, 5).

### Results

| Model | Train Acc | Test Acc | Test F1 (Macro) | Overfitting Gap |
|-------|-----------|----------|-----------------|-----------------|
| **FP-Optimized MLP** | 75.7% | **77.8%** | 0.419 | -2.1% ✓ |
| LDA | 69.7% | 76.7% | 0.429 | -7.0% ✓ |
| Logistic Regression | 75.1% | 76.7% | 0.459 | -1.5% ✓ |
| MLP | 98.6% | 75.6% | 0.471 | +23.0% |
| Random Forest | 100.0% | 73.3% | 0.321 | +26.7% |

### Analysis

**Best Model**: **FP-Optimized MLP** (77.8% test accuracy)
- Excellent generalization (-2.1% negative gap)
- Good F1 macro score (0.419) for 6-class problem

**Key Insights**:
- **Much more challenging task**: Test accuracies drop from ~90% (binary) to ~77% (multiclass)
- Random Forest severely overfits (26.7% gap) - memorizes training data
- All linear models (LDA, Logreg) show negative gaps = excellent generalization
- MLP overfits significantly (+23%) without class weighting
- **F1 scores are lower** because macro-averaging across 6 classes is harder than binary

**Why Multi-class is Harder**:
1. **More classes to distinguish**: 6 classes vs 2 classes
2. **Imbalanced data**: Label 0 has 313 samples, but labels 3-5 have only 32-57 each
3. **Subtle differences**: Distinguishing intensity 2 from 3 is much harder than 0 vs 1+

---

## Test 3: Threshold-2 Classification (0-2 vs 3-5) - Alternative Binary

This mode treats weak responses (0-2) as non-reactive and strong responses (3-5) as reactive.

### Results

| Model | Train Acc | Test Acc | Test F1 | Overfitting Gap |
|-------|-----------|----------|---------|-----------------|
| **Logistic Regression** | 94.9% | **97.5%** | 0.955 | -2.6% ✓ |
| Random Forest | 100.0% | 96.3% | 0.930 | +3.7% |
| LDA | 94.3% | 95.0% | 0.913 | -0.7% ✓ |
| MLP | 100.0% | 95.0% | 0.909 | +5.0% |
| FP-Optimized MLP | 92.4% | 93.8% | 0.878 | -1.3% ✓ |

### Analysis

**Best Model**: **Logistic Regression** (97.5% test accuracy!)
- **Outstanding performance**: 97.5% accuracy with F1 = 0.955
- Excellent generalization (-2.6% negative gap)
- This threshold creates a much cleaner classification boundary

**Key Insights**:
- **Easiest task**: All models perform exceptionally well (>93% accuracy)
- The 0-2 vs 3-5 split creates a more natural separation in the data
- Even Random Forest overfits less here (+3.7% vs +12.5% in binary mode)
- **Why so much better?**:
  - **Cleaner separation**: Strong responses (3-5) are more distinct from weak/no responses (0-2)
  - **Less label noise**: Ambiguous cases (e.g., intensity 1-2) are now grouped with no-response
  - **Better class balance**: 417 samples in class 0 (0-2) vs 123 in class 1 (3-5)

---

## Comparison Across Modes

### Accuracy Comparison

| Model | Binary (0 vs 1-5) | Multiclass (0-5) | Threshold-2 (0-2 vs 3-5) |
|-------|-------------------|------------------|---------------------------|
| Logistic Regression | 90.0% | 76.7% | **97.5%** ⭐ |
| Random Forest | 87.5% | 73.3% | 96.3% |
| FP-Optimized MLP | 87.5% | **77.8%** ⭐ | 93.8% |
| LDA | 86.3% | 76.7% | 95.0% |
| MLP | 82.5% | 75.6% | 95.0% |

### Overfitting Comparison

| Model | Binary Gap | Multiclass Gap | Threshold-2 Gap |
|-------|------------|----------------|------------------|
| Logistic Regression | -1.4% ✓ | -1.5% ✓ | -2.6% ✓ |
| LDA | -0.4% ✓ | -7.0% ✓ | -0.7% ✓ |
| FP-Optimized MLP | +8.6% | -2.1% ✓ | -1.3% ✓ |
| Random Forest | +12.5% | +26.7% ❌ | +3.7% |
| MLP | +14.2% | +23.0% ❌ | +5.0% |

**✓** = Negative gap (generalizes better to test data)
**❌** = Large positive gap (overfitting)

---

## Key Findings

### 1. Best Overall Model: **Logistic Regression**
- **Consistently excellent** across all modes
- Best at generalization (negative gaps in all modes)
- Achieves top performance in binary and threshold-2 modes
- Simple, interpretable, and reliable

### 2. Task Difficulty Ranking
1. **Easiest**: Threshold-2 (0-2 vs 3-5) → 97.5% accuracy
2. **Medium**: Binary (0 vs 1-5) → 90.0% accuracy
3. **Hardest**: Multiclass (0-5) → 77.8% accuracy

### 3. Overfitting Patterns
- **Linear models** (LDA, Logistic Regression) rarely overfit
- **Tree-based** (Random Forest) consistently overfits, especially on multiclass
- **Neural networks** (MLP) overfit unless class-weighted (FP-Optimized MLP)

### 4. When to Use Each Mode

**Binary (0 vs 1-5)**: Use when you want to detect *any* response
- Good for: Initial screening, high sensitivity needed
- Limitation: Treats all response strengths equally

**Multiclass (0-5)**: Use when you need to quantify response strength
- Good for: Detailed analysis, dose-response studies
- Limitation: More complex, requires more data, lower accuracy

**Threshold-2 (0-2 vs 3-5)**: Use when you want to detect *strong* responses only
- Good for: High-confidence predictions, filtering weak/ambiguous responses
- Advantage: Highest accuracy, cleanest separation

---

## Recommendations

### For Maximum Accuracy:
Use **Threshold-2 mode with Logistic Regression** → 97.5% accuracy

### For Response Detection:
Use **Binary mode with Logistic Regression** → 90.0% accuracy, balanced sensitivity

### For Response Quantification:
Use **Multiclass mode with FP-Optimized MLP** → 77.8% accuracy
- Then apply **threshold slider** post-hoc to adjust cutoffs

### For Interpretation:
Use **Logistic Regression** in any mode - provides:
- Coefficient weights showing feature importance
- Probability scores for uncertainty quantification
- Fast training and prediction

---

## Using the Threshold Slider Tool

For multiclass predictions, you can adjust the threshold post-hoc without retraining:

```bash
# 1. Train multiclass model
flybehavior-response train --classification-mode multiclass --model mlp \\
    --data-csv data.csv --labels-csv labels.csv --artifacts-dir multiclass_run

# 2. Analyze different thresholds
python threshold_slider.py \\
    --predictions-csv multiclass_run/latest/predictions_mlp_test.csv \\
    --min-threshold 1 --max-threshold 5
```

This generates:
- **Performance metrics** for each threshold (1, 2, 3, 4, 5)
- **Precision-recall tradeoffs** visualization
- **Optimal threshold** recommendations

**Example output**: "Best F1 at threshold=2" → same as threshold-2 mode, but you didn't need to retrain!

---

## Conclusion

The new classification modes provide flexibility to match your scientific question:
- **Binary**: Sensitive detection (any response)
- **Multiclass**: Quantitative analysis (response strength)
- **Threshold-2**: Specific detection (strong responses only)

All modes work seamlessly with all models, and the threshold slider enables post-hoc optimization without retraining.
