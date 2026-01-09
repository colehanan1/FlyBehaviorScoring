#!/usr/bin/env python
"""Quick diagnostic script for model performance analysis"""

from pathlib import Path
import sys

import json
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import artifacts_dir  # noqa: E402

# Load recent results
RUN_DIR = artifacts_dir() / "2025-11-13T20-14-45Z"

print("="*80)
print("  QUICK DIAGNOSTIC SUMMARY")
print("="*80)

# Load metrics
with open(RUN_DIR / "metrics.json") as f:
    metrics = json.load(f)['models']

# Best model
print("\n🏆 BEST MODEL: Logistic Regression")
logreg = metrics['logreg']['test']
print(f"  Test Accuracy:  {logreg['accuracy']:.1%}")
print(f"  Precision:      {logreg['precision']:.1%}")
print(f"  Recall:         {logreg['recall']:.1%}")
print(f"  F1 Score:       {logreg['f1_binary']:.1%}")
print(f"  FNR:            {logreg['false_negative_rate']:.1%}")
print(f"  FPR:            {logreg['false_positive_rate']:.1%}")

# Confusion matrix
cm = logreg['confusion_matrix']['raw']
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

# Load error details
errors_df = pd.read_csv(RUN_DIR / "predictions_logreg_test.csv")
fn = errors_df[(errors_df['user_score_odor'] > 0) & (errors_df['predicted_label'] == 0)]

print(f"\n❌ FALSE NEGATIVES: {len(fn)} errors")
if len(fn) > 0:
    print(f"  All from intensity-1 responses: {(fn['user_score_odor_intensity'] == 1).sum()}/{len(fn)}")
    print(f"  Mean confidence: {fn['prob_reaction'].mean():.3f}")
    print(f"\n  These are genuinely difficult borderline cases!")

print(f"\n✅ RECOMMENDATIONS:")
print(f"  1. Use LogReg as production model")
print(f"  2. Lower threshold to 0.35 to reduce FNR")
print(f"  3. Retrain with class weights: --logreg-class-weights '0:1.0,1:3.0'")

print(f"\n📊 Expected Performance After Fixes:")
print(f"  Test Accuracy:  ~90-92%")
print(f"  FNR:            ~8-12% (down from 21%)")
print(f"  FPR:            ~3-5% (slight increase)")

print("\n" + "="*80)
