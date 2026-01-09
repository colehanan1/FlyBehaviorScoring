# Iterative Label Refinement: The Right Way

## The Problem You Just Discovered

**What you did:**
1. Trained model → 93.6% test accuracy
2. Looked at errors on test set
3. Changed labels you disagreed with
4. Test accuracy went DOWN

**Why accuracy went down:**
- Old model trained on OLD labels (including incorrect ones)
- You're now testing it on NEW labels
- Of course it does worse! It learned to predict the old errors!

**What you SHOULD have seen:**
- You need to RETRAIN with new labels
- Then compare apples-to-apples

---

## The Bigger Problem: Test Set Contamination

**Warning:** If you keep doing this cycle, you risk a serious problem:

```
1. Train model
2. Look at test set errors
3. Change labels in test set
4. Retrain
5. Repeat...
```

**This is DANGEROUS because:**
- You're using the test set to guide labeling decisions
- You're indirectly "training" on the test set
- Final test accuracy becomes unreliable (overfitted to test set)
- You won't know how well your model generalizes to truly new data

This is called **"test set contamination"** or **"data leakage"**.

---

## The Right Approach: Hold-Out Validation Set

### Strategy 1: Three-Way Split

Instead of train/test, use train/validation/test:

```
All data (480 trials, 40 flies)
├── Training set (60%) - 24 flies, ~290 trials
│   └── Train model here
├── Validation set (20%) - 8 flies, ~95 trials
│   └── Review errors, fix labels, tune hyperparameters
└── Test set (20%) - 8 flies, ~95 trials
    └── NEVER LOOK AT THIS until final evaluation
```

**Workflow:**
1. Train on training set
2. Evaluate on validation set
3. Find errors in VALIDATION set (not test!)
4. Review and fix labels in VALIDATION set
5. Retrain
6. Repeat until satisfied
7. ONLY THEN look at test set for final performance

**Benefits:**
- Test set remains "pristine"
- You can iterate on validation set safely
- Final test accuracy is trustworthy

### Strategy 2: Multiple Rounds with Fresh Hold-Outs

If you've already contaminated your test set:

**Round 1:**
- Train/test split: Flies 1-32 train, Flies 33-40 test
- Review errors, fix labels
- Report performance

**Round 2:**
- NEW train/test split: Flies 1-8 + 17-40 train, Flies 9-16 test
- Use corrected labels from Round 1
- Train on new split
- Report performance on FRESH test flies

**This proves:**
- Your label corrections weren't just overfitting to one test set
- Model generalizes to different flies

### Strategy 3: Cross-Validation Only (Safest)

Don't use a fixed test set at all:

```bash
# Train with 5-fold CV, report average performance
flybehavior-response train \
  --model random_forest \
  --cv 5 \
  ... other args
```

**Benefits:**
- Every fly appears in test fold exactly once
- Can't overfit to a specific test set
- Get confidence intervals on performance
- More stable estimate with small datasets

**When to use:**
- When you have limited data (like your 480 trials)
- When you're still iterating on labels
- For hyperparameter tuning

---

## When to Stop Iterating on Labels

### Stop if:

1. **Diminishing returns**
   - Changed 10 labels, accuracy improved 0.1%
   - Not worth the effort

2. **Errors are genuinely ambiguous**
   - Remaining misclassifications look hard even to you
   - Multiple reviewers disagree on labels
   - These are "irreducible errors"

3. **You're second-guessing yourself**
   - Changing labels back and forth
   - No clear criteria for what's right
   - Sign you need better labeling guidelines, not more iterations

4. **Test set contamination risk**
   - You've looked at test errors more than 2-3 times
   - Time to get fresh test data or use CV

### Continue if:

1. **Clear systematic errors found**
   - Example: "I used the wrong threshold initially"
   - Fix all instances of this error at once

2. **Label distribution is clearly wrong**
   - Example: "I have 80% reactions but video shows only 40%"
   - Suggests systematic over-labeling

3. **Multiple independent reviewers agree**
   - Have colleague review errors blind
   - If they agree with your corrections → good signal

4. **Corrections follow clear criteria**
   - You can write down WHY each label was changed
   - Reproducible by others following your criteria

---

## Best Practices for Label Quality

### 1. Define Clear Labeling Criteria FIRST

Write down explicit rules before labeling:

```markdown
## Reaction (Label = 1) Criteria:
- local_max_during > 2.0 * global_min
- Peak occurs within 0-500ms of stimulus
- Clear upward deflection in trace
- Duration > 100ms

## No Reaction (Label = 0) Criteria:
- local_max_during < 1.5 * global_min
- OR peak occurs before stimulus
- OR only noise/artifact visible
```

### 2. Use Multiple Reviewers

- Have 2-3 people label the same 50 trials
- Calculate inter-rater agreement (Cohen's kappa)
- If kappa < 0.7, your labeling task is too ambiguous
- Discuss disagreements and refine criteria

### 3. Blind Review

When relabeling:
- DON'T look at model predictions
- DON'T look at old labels
- Review raw data only
- Then compare to old labels

This prevents confirmation bias.

### 4. Track Label Changes

Use the `label_correction_log.md` file:
- Document WHY each change was made
- Track how many 0→1 vs 1→0 changes
- Note if changes cluster in certain flies (might indicate fly-specific issues)

### 5. Use Confidence Scores

Instead of binary labels, consider:

```csv
trial,label,confidence
trial_001,1,high       # Clear reaction
trial_002,0,high       # Clear no reaction
trial_003,1,medium     # Likely reaction but ambiguous
trial_004,1,low        # Very uncertain, might be noise
```

Then:
- Train on high+medium confidence only
- Evaluate on high confidence only
- Review low confidence cases with expert

---

## Your Specific Situation

### What you should do NOW:

1. **Document your label changes**
   - Fill out `label_correction_log.md`
   - Note: how many changed, which flies, why

2. **Retrain with corrected labels**
   ```bash
   flybehavior-response train \
     --model random_forest \
     --seed 42 \
     ... (same config as before)
     --artifacts-dir outputs/artifacts/rf_corrected_labels
   ```

3. **Compare results**
   ```bash
   python scripts/analysis/compare_label_versions.py \
     outputs/artifacts/rf_tuned/2025-11-14T01-47-01Z/metrics.json \
     outputs/artifacts/rf_corrected_labels/LATEST/metrics.json
   ```

4. **Decide next steps based on results:**

   **If test accuracy improved or stayed similar:**
   - ✅ Your label corrections were good
   - Continue with corrected dataset
   - Try hyperparameter tuning (max_features, max_depth)

   **If test accuracy dropped significantly (>2%):**
   - ⚠️ Possible test set contamination
   - Or: label changes introduced inconsistencies
   - Review the NEW errors carefully
   - Consider using CV instead of fixed test set

### Long-term strategy:

Since you have limited data (480 trials), I recommend:

**Option A: Use CV exclusively**
```bash
flybehavior-response train \
  --model random_forest \
  --cv 5 \
  ... other args
```
- Report 5-fold CV accuracy ± std dev
- No separate test set to contaminate
- More reliable with small datasets

**Option B: Collect more data**
- Get 200-300 more trials from new flies
- Use OLD flies for train/val, NEW flies for test
- Guarantees no label contamination

---

## Key Takeaways

1. ✅ **Fixing label errors is GOOD** - improves data quality
2. ⚠️ **Looking at test errors to decide which labels to fix is RISKY** - causes test set contamination
3. ✅ **Retraining after label changes is REQUIRED** - old model learned old labels
4. ⚠️ **Accuracy going down after relabeling is NORMAL** - you're comparing to new ground truth
5. ✅ **Use validation set or CV for iterative refinement** - keep test set pristine
6. 🛑 **Stop iterating when errors are genuinely ambiguous** - you've hit the ceiling

---

## Am I Overfitting to the Test Set? (Self-Check)

Ask yourself:

- [ ] Have I looked at test set errors more than 3 times?
- [ ] Have I changed labels specifically in the test set based on model errors?
- [ ] Am I changing labels to make the model "look better"?
- [ ] Do I remember which specific test samples were wrong?

If you answered YES to any of these → **test set is contaminated**

**Solution:** Use CV going forward, or collect fresh test data.

---

## Final Recommendation

**DON'T keep repeating the cycle indefinitely!**

Instead:

1. Do ONE round of careful label review (with clear criteria)
2. Retrain with corrected labels
3. Evaluate with 5-fold CV (not fixed test set)
4. Report CV performance ± std dev
5. Move on to hyperparameter tuning (max_features, max_depth)
6. Collect more data if needed

You'll get better improvements from tuning and more data than from endless label iteration.
