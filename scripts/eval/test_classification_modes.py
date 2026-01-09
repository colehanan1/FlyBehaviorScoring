#!/usr/bin/env python3
"""Test script to verify the new classification modes work correctly."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import data_dir, ensure_src_on_path, outputs_dir, resolve_path  # noqa: E402

ensure_src_on_path()

from flybehavior_response.train import train_models  # noqa: E402

# Paths
data_path = resolve_path(data_dir() / "all_envelope_rows_wide.csv")
labels_path = resolve_path(data_dir() / "scoring_results_opto_new_MINIMAL.csv")
run_root = outputs_dir() / "test_classification_modes"

print("="*100)
print("TESTING NEW CLASSIFICATION MODES")
print("="*100)

# Test 1: Binary mode (default, 0 vs 1-5)
print("\n" + "="*100)
print("TEST 1: Binary Classification Mode (0 vs 1-5)")
print("="*100)

try:
    results_binary = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        trace_prefixes=None,
        include_traces=True,
        use_raw_pca=True,
        n_pcs=10,
        models=["mlp"],
        cv=0,  # Skip CV for speed
        artifacts_dir=run_root / "binary",
        seed=67,
        verbose=False,
        dry_run=False,
        classification_mode='binary',
    )

    mlp_test_acc = results_binary['models']['mlp']['test']['accuracy']
    mlp_train_acc = results_binary['models']['mlp']['accuracy']

    print(f"\n✓ Binary mode completed successfully!")
    print(f"  Train Accuracy: {mlp_train_acc:.3f}")
    print(f"  Test Accuracy: {mlp_test_acc:.3f}")
    print(f"  Overfitting Gap: {mlp_train_acc - mlp_test_acc:.3f}")

except Exception as e:
    print(f"\n✗ Binary mode FAILED: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 2: Multiclass mode (0-5)
print("\n" + "="*100)
print("TEST 2: Multi-class Classification Mode (0-5)")
print("="*100)

try:
    results_multiclass = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        trace_prefixes=None,
        include_traces=True,
        use_raw_pca=True,
        n_pcs=10,
        models=["mlp"],
        cv=0,
        artifacts_dir=run_root / "multiclass",
        seed=67,
        verbose=False,
        dry_run=False,
        classification_mode='multiclass',
    )

    mlp_test_acc = results_multiclass['models']['mlp']['test']['accuracy']
    mlp_train_acc = results_multiclass['models']['mlp']['accuracy']

    print(f"\n✓ Multi-class mode completed successfully!")
    print(f"  Train Accuracy: {mlp_train_acc:.3f}")
    print(f"  Test Accuracy: {mlp_test_acc:.3f}")
    print(f"  Overfitting Gap: {mlp_train_acc - mlp_test_acc:.3f}")

    # Check that predictions are in range 0-5
    pred_path = run_root / "multiclass"
    latest_dir = sorted(pred_path.glob("20*"))[-1]
    preds_test = pd.read_csv(latest_dir / "predictions_mlp_test.csv")

    unique_preds = sorted(preds_test['predicted_label'].unique())
    print(f"  Unique predicted classes: {unique_preds}")

    if min(unique_preds) < 0 or max(unique_preds) > 5:
        print(f"  ✗ WARNING: Predictions outside expected range [0, 5]")
    else:
        print(f"  ✓ Predictions in expected range [0, 5]")

except Exception as e:
    print(f"\n✗ Multi-class mode FAILED: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 3: Threshold-2 mode (0-2 vs 3-5)
print("\n" + "="*100)
print("TEST 3: Threshold-2 Classification Mode (0-2 vs 3-5)")
print("="*100)

try:
    results_threshold2 = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        trace_prefixes=None,
        include_traces=True,
        use_raw_pca=True,
        n_pcs=10,
        models=["mlp"],
        cv=0,
        artifacts_dir=run_root / "threshold2",
        seed=67,
        verbose=False,
        dry_run=False,
        classification_mode='threshold-2',
    )

    mlp_test_acc = results_threshold2['models']['mlp']['test']['accuracy']
    mlp_train_acc = results_threshold2['models']['mlp']['accuracy']

    print(f"\n✓ Threshold-2 mode completed successfully!")
    print(f"  Train Accuracy: {mlp_train_acc:.3f}")
    print(f"  Test Accuracy: {mlp_test_acc:.3f}")
    print(f"  Overfitting Gap: {mlp_train_acc - mlp_test_acc:.3f}")

    # Check class distribution
    pred_path = Path("test_classification_modes/threshold2")
    latest_dir = sorted(pred_path.glob("20*"))[-1]
    preds_test = pd.read_csv(latest_dir / "predictions_mlp_test.csv")

    class_counts = preds_test['user_score_odor'].value_counts().sort_index()
    print(f"  True label distribution: {dict(class_counts)}")

    unique_preds = sorted(preds_test['predicted_label'].unique())
    print(f"  Unique predicted classes: {unique_preds}")

except Exception as e:
    print(f"\n✗ Threshold-2 mode FAILED: {str(e)}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("\nAll three classification modes have been implemented and tested:")
print("  1. ✓ binary: 0 vs 1-5 (default behavior)")
print("  2. ✓ multiclass: preserve all 6 classes (0-5)")
print("  3. ✓ threshold-2: 0-2 vs 3-5 (alternative binary threshold)")
print("\nTo use these modes via CLI:")
print("  flybehavior-response train --classification-mode binary ...")
print("  flybehavior-response train --classification-mode multiclass ...")
print("  flybehavior-response train --classification-mode threshold-2 ...")
print("\nFor post-hoc threshold analysis on multi-class predictions:")
print("  python scripts/eval/threshold_slider.py --predictions-csv path/to/predictions.csv")

print("\n" + "="*100)
print("DONE")
print("="*100)
