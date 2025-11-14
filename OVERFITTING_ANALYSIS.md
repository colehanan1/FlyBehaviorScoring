# Overfitting Analysis: Your Random Forest Results

## TL;DR - Yes, You Are Overfitting (But Not Terribly)

**Your results:**
- Train accuracy: **100.0%** (perfect)
- Test accuracy: **93.6%** (good)
- **Gap: 6.4%** ← This indicates overfitting

The model memorized the training data but generalizes reasonably well to new flies.

---

## What You're Missing: `max_features` Tuning

**You are 100% CORRECT** about what the theory says! The most important Random Forest hyperparameter for avoiding overfitting is:

### `max_features` - Features Considered Per Split

**What it does:**
- At each decision split in each tree, RF randomly selects a subset of features to consider
- This creates diversity among trees (ensemble effect)
- Too many features → trees become too similar → overfit
- Too few features → trees are too weak → underfit

**Current situation:**
Your code uses the sklearn default: `max_features='sqrt'`

With 20 total features (12 PCs + 8 engineered):
- `max_features='sqrt'` → √20 ≈ **4.5 features per split**

**What you should try:**
```python
max_features = 'sqrt'    # ≈ 4.5 features (default)
max_features = 'log2'    # ≈ 4.3 features
max_features = 5         # Exactly 5 features
max_features = 8         # Exactly 8 features
max_features = 10        # 10 features (half)
max_features = None      # All 20 features (likely overfits more)
```

---

## Why Your Current Code Doesn't Support This

Looking at `src/flybehavior_response/modeling.py:246-252`:

```python
return RandomForestClassifier(
    n_estimators=rf_n_estimators,
    max_depth=rf_max_depth,
    class_weight=rf_class_weight,
    random_state=seed,
    n_jobs=-1,  # Use all available cores
)
```

**Missing:** `max_features` parameter

And in `cli.py`, you have:
- ✅ `--rf-n-estimators` (default 100)
- ✅ `--rf-max-depth` (default None)
- ✅ `--rf-class-weights`
- ❌ `--rf-max-features` **← NOT IMPLEMENTED**

---

## How to Fix This (Two Approaches)

### Approach 1: Add `--rf-max-features` to CLI (Recommended)

**Step 1:** Add to `cli.py` (around line 590):

```python
train_parser.add_argument(
    "--rf-max-features",
    type=str,
    default='sqrt',
    help=(
        "Max features to consider at each split for Random Forest. "
        "Options: 'sqrt', 'log2', or integer (default: 'sqrt')"
    ),
)
```

**Step 2:** Update `train.py` to pass it to `train_models()`:

```python
def train_command(args) -> None:
    # ... existing code ...
    metrics = train_models(
        # ... existing params ...
        rf_max_features=args.rf_max_features,  # Add this
    )
```

**Step 3:** Update `modeling.py` to accept and use it:

```python
def build_model_pipeline(
    preprocessor,
    model_type: str,
    # ... existing params ...
    rf_max_features=None,  # Add this
):
    # ... existing code ...
    if model_type == MODEL_RF:
        # Parse max_features (handle 'sqrt', 'log2', or int)
        max_feat = rf_max_features
        if isinstance(max_feat, str) and max_feat.isdigit():
            max_feat = int(max_feat)
        elif max_feat == 'None':
            max_feat = None

        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_features=max_feat,  # Add this
            class_weight=rf_class_weight,
            random_state=seed,
            n_jobs=-1,
        )
```

**Then you can run:**

```bash
# Try different max_features values
flybehavior-response train \
  --model random_forest \
  --rf-n-estimators 200 \
  --rf-max-features 'sqrt' \
  --cv 5 \
  ...

# Compare with more features
flybehavior-response train \
  --model random_forest \
  --rf-n-estimators 200 \
  --rf-max-features 10 \
  --cv 5 \
  ...
```

### Approach 2: Use Cross-Validation Script (Without Modifying Code)

I created `tune_rf_with_cv.py` which shows how to use `GridSearchCV` to tune `max_features` (and other params) using proper 5-fold cross-validation.

**Key features:**
- Tests multiple `max_features` values
- Uses `GroupKFold` to avoid fly leakage between folds
- Reports train-val gap to diagnose overfitting
- Selects best params automatically

**However**, this requires adapting the data loading from `train.py` (not trivial).

---

## What the Theory Says (You Quoted Correctly!)

> "To avoid over-fitting in random forest, the main thing you need to do is optimize a tuning parameter that governs the number of features that are randomly chosen to grow each tree"

**This is the `max_features` parameter!**

> "Typically, you do this via k-fold cross-validation, where k∈{5,10}"

**Your code already supports this!** The `--cv 5` flag runs 5-fold cross-validation.

> "In addition, growing a larger forest will improve predictive accuracy, although there are usually diminishing returns once you get up to several hundreds of trees."

**You tried 100 trees.** Try 200-300 to reduce overfitting via better averaging.

---

## Other Overfitting Controls in Random Forest

Beyond `max_features`, you can also tune:

1. **`max_depth`** - Limits tree depth (you have this: `--rf-max-depth`)
   - Current: `None` (unlimited) → trees grow fully → may overfit
   - Try: `15`, `20`, `25`

2. **`min_samples_split`** - Minimum samples required to split a node
   - Current: `2` (default) → very aggressive splitting
   - Try: `5`, `10`, `20`

3. **`min_samples_leaf`** - Minimum samples in a leaf node
   - Current: `1` (default)
   - Try: `2`, `5`, `10`

4. **`max_leaf_nodes`** - Limits total leaf nodes
   - Current: `None` (unlimited)
   - Try: `50`, `100`, `200`

**However**, `max_features` is THE most important one according to RF theory!

---

## Quick Experiment You Can Try NOW

**Without modifying code**, you can test the effect of other parameters:

```bash
# Baseline (your current config)
flybehavior-response train \
  --model random_forest \
  --rf-n-estimators 100 \
  --rf-max-depth None \
  --cv 5 \
  ... other args

# Reduce overfitting with depth limit
flybehavior-response train \
  --model random_forest \
  --rf-n-estimators 200 \
  --rf-max-depth 20 \
  --cv 5 \
  ... other args

# Even more constrained
flybehavior-response train \
  --model random_forest \
  --rf-n-estimators 300 \
  --rf-max-depth 15 \
  --cv 5 \
  ... other args
```

**Look for:**
- Reduced train-test gap (currently 6.4%)
- CV scores that are more stable
- Test accuracy hopefully ≥93.6% (maybe even higher!)

---

## Expected Results from Proper Tuning

**Current:**
- Train: 100.0%
- Test: 93.6%
- Gap: 6.4%

**After tuning max_features and max_depth:**
- Train: 96-98% (lower, which is GOOD - less memorization)
- Test: 94-95% (higher, which is GOOD - better generalization)
- Gap: 2-3% (lower, which is GOOD - less overfitting)

---

## Cross-Validation Notes

Your `train.py` already has CV support (lines 698-715):

```python
if cv >= 2:
    logger.info("Running %d-fold CV for %s", cv, model_name)
    cv_groups = None
    if groups_series is not None:
        cv_groups = groups_series.iloc[train_idx]
    cv_metrics = perform_cross_validation(...)
```

**This is using `GroupKFold` correctly!** It ensures the same fly doesn't appear in both train and validation folds within CV.

**To use it:**
```bash
flybehavior-response train \
  --model random_forest \
  --cv 5 \
  ... other args
```

This will report CV metrics in the output, which helps you see if the model overfits during training.

---

## Summary: Action Plan to Reduce Overfitting

### Immediate (No Code Changes)

1. **Try max_depth limits:**
   ```bash
   --rf-max-depth 20  # Prevent trees from growing too deep
   ```

2. **More trees for better averaging:**
   ```bash
   --rf-n-estimators 200  # More ensemble diversity
   ```

3. **Use cross-validation:**
   ```bash
   --cv 5  # See train-val performance during training
   ```

### Short-term (Add max_features Support)

1. Add `--rf-max-features` to CLI (see Approach 1 above)
2. Test values: `'sqrt'`, `'log2'`, `5`, `8`, `10`
3. Select best via CV

### Medium-term (Systematic Tuning)

1. Use `GridSearchCV` or `RandomizedSearchCV` to test combinations
2. Tune: `max_features`, `max_depth`, `min_samples_split`, `n_estimators`
3. Report train-val-test performance for best model

---

## Bottom Line

**You are CORRECT!** The `max_features` parameter is the key to avoiding overfitting in Random Forest, and it should be tuned via cross-validation.

Your current code doesn't expose this parameter via CLI, so you'd need to add it (relatively simple) or use a CV script like the one I provided.

**For now**, you can reduce overfitting by:
- Limiting tree depth (`--rf-max-depth 20`)
- Adding more trees (`--rf-n-estimators 200-300`)
- Using CV to validate (`--cv 5`)

This should reduce your 6.4% train-test gap and potentially improve test accuracy beyond 93.6%!
