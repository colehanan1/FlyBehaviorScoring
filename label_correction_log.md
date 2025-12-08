# Label Correction Log

## Purpose
Track all manual label corrections to maintain data quality and explain model performance changes.

## Correction History

### Round 1: [DATE]

**Total changes:** [X labels changed]

**Rationale:** [Why you changed labels - e.g., "After reviewing misclassified cases from RF model, found several ambiguous traces that I initially labeled incorrectly"]

#### Individual Changes

| Trial ID | Fly | Old Label | New Label | Reason |
|----------|-----|-----------|-----------|--------|
| trial_123 | fly_5 | 1 (reaction) | 0 (no reaction) | Trace shows minimal response, below threshold |
| trial_456 | fly_12 | 0 (no reaction) | 1 (reaction) | Clear PER visible in video, trace confirms |
| ... | ... | ... | ... | ... |

**Impact:**
- Changed [X] labels from 0→1 (added reactions)
- Changed [Y] labels from 1→0 (removed false reactions)
- Net change in reaction rate: [before X%] → [after Y%]

**Next steps:**
- Retrain model with corrected labels
- Compare test accuracy before/after
- If accuracy improves, labels were good corrections
- If accuracy stays low, may need more data or better features

---

## Guidelines for Future Label Changes

### When to change a label:
1. ✅ Clear evidence in trace data that contradicts label
2. ✅ Video evidence (if available) shows different behavior
3. ✅ Label was ambiguous and you now have better criteria
4. ✅ Systematic error discovered (e.g., threshold was wrong)

### When NOT to change a label:
1. ❌ Model got it wrong but trace is genuinely ambiguous
2. ❌ Just to improve model accuracy (overfitting to test set!)
3. ❌ Based on single model's prediction alone
4. ❌ Without reviewing the actual data

### Best practices:
- Have a second person review ambiguous cases
- Define clear labeling criteria and stick to them
- Don't look at model predictions while relabeling (blind review)
- Keep a log like this for reproducibility
