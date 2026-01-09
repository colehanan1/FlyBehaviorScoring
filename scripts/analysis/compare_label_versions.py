#!/usr/bin/env python3
"""
Compare model performance before and after label corrections.

This helps determine if label corrections improved data quality.
"""

import json
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import artifacts_dir  # noqa: E402


def compare_versions(
    old_metrics_path: str,
    new_metrics_path: str,
):
    """
    Compare two model training runs to see effect of label corrections.

    Parameters
    ----------
    old_metrics_path : str
        Path to metrics.json from training with OLD labels
    new_metrics_path : str
        Path to metrics.json from training with NEW (corrected) labels
    """

    with open(old_metrics_path) as f:
        old_metrics = json.load(f)

    with open(new_metrics_path) as f:
        new_metrics = json.load(f)

    # Extract RF metrics
    old_rf = old_metrics['models']['random_forest']
    new_rf = new_metrics['models']['random_forest']

    print("="*80)
    print("LABEL CORRECTION IMPACT ANALYSIS")
    print("="*80)

    print(f"\nOld labels: {old_metrics_path}")
    print(f"New labels: {new_metrics_path}")

    # Training performance
    print(f"\n{'='*80}")
    print("TRAINING SET PERFORMANCE")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Old Labels':>15} {'New Labels':>15} {'Change':>12}")
    print("-"*80)

    metrics_to_compare = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_binary', 'F1 Score'),
        ('false_negative_rate', 'False Negative Rate'),
        ('false_positive_rate', 'False Positive Rate'),
    ]

    for key, label in metrics_to_compare:
        old_val = old_rf[key]
        new_val = new_rf[key]
        change = new_val - old_val
        change_str = f"{change:+.3f}"
        print(f"{label:<30} {old_val:>15.3f} {new_val:>15.3f} {change_str:>12}")

    # Test performance
    print(f"\n{'='*80}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Old Labels':>15} {'New Labels':>15} {'Change':>12}")
    print("-"*80)

    for key, label in metrics_to_compare:
        old_val = old_rf['test'][key]
        new_val = new_rf['test'][key]
        change = new_val - old_val
        change_str = f"{change:+.3f}"

        # Highlight significant changes
        if abs(change) > 0.02:  # >2% change
            change_str += " *"

        print(f"{label:<30} {old_val:>15.3f} {new_val:>15.3f} {change_str:>12}")

    # Overfitting analysis
    print(f"\n{'='*80}")
    print("OVERFITTING ANALYSIS")
    print(f"{'='*80}")

    old_train_acc = old_rf['accuracy']
    old_test_acc = old_rf['test']['accuracy']
    old_gap = old_train_acc - old_test_acc

    new_train_acc = new_rf['accuracy']
    new_test_acc = new_rf['test']['accuracy']
    new_gap = new_train_acc - new_test_acc

    print(f"\nOld labels:")
    print(f"  Train accuracy: {old_train_acc:.3f}")
    print(f"  Test accuracy:  {old_test_acc:.3f}")
    print(f"  Gap:            {old_gap:.3f} ({old_gap*100:.1f}%)")

    print(f"\nNew labels:")
    print(f"  Train accuracy: {new_train_acc:.3f}")
    print(f"  Test accuracy:  {new_test_acc:.3f}")
    print(f"  Gap:            {new_gap:.3f} ({new_gap*100:.1f}%)")

    gap_change = new_gap - old_gap
    print(f"\nOverfitting gap change: {gap_change:+.3f}")
    if abs(gap_change) < 0.02:
        print("  → No significant change")
    elif gap_change < 0:
        print("  → ✅ IMPROVED: Less overfitting with new labels")
    else:
        print("  → ⚠️ WORSE: More overfitting with new labels")

    # Confusion matrices
    print(f"\n{'='*80}")
    print("CONFUSION MATRICES (Test Set)")
    print(f"{'='*80}")

    print("\nOld labels:")
    old_cm = old_rf['test']['confusion_matrix']['raw']
    print(f"  True Neg:  {old_cm[0][0]:<6} False Pos: {old_cm[0][1]}")
    print(f"  False Neg: {old_cm[1][0]:<6} True Pos:  {old_cm[1][1]}")

    print("\nNew labels:")
    new_cm = new_rf['test']['confusion_matrix']['raw']
    print(f"  True Neg:  {new_cm[0][0]:<6} False Pos: {new_cm[0][1]}")
    print(f"  False Neg: {new_cm[1][0]:<6} True Pos:  {new_cm[1][1]}")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    test_acc_change = new_test_acc - old_test_acc

    print("\nWhat the results mean:")

    if test_acc_change > 0.01:
        print("✅ Test accuracy IMPROVED after label corrections")
        print("   → Your label corrections were good!")
        print("   → The model now better matches the TRUE underlying patterns")
    elif test_acc_change < -0.01:
        print("⚠️  Test accuracy DECREASED after label corrections")
        print("   → This is NORMAL if you corrected test set labels")
        print("   → The old model was trained to predict old (incorrect) labels")
        print("   → Compare error patterns, not just accuracy")
    else:
        print("→ Test accuracy stayed roughly the same")
        print("   → Label changes may have been minor")
        print("   → Check if error patterns changed (FP vs FN)")

    print("\nNext steps:")
    if test_acc_change < -0.01:
        print("1. Review the NEW misclassified cases")
        print("2. If they look genuinely hard/ambiguous → labels are good")
        print("3. If they look obviously wrong → may need more label review")
        print("4. Consider if you need more training data or better features")
    else:
        print("1. Continue with this corrected dataset")
        print("2. Try hyperparameter tuning to improve further")
        print("3. Consider addressing remaining misclassifications")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/analysis/compare_label_versions.py <old_metrics.json> <new_metrics.json>")
        print("\nExample:")
        print("  python scripts/analysis/compare_label_versions.py \\")
        print(f"    {artifacts_dir() / 'rf_tuned/2025-11-14T01-47-01Z/metrics.json'} \\")
        print(f"    {artifacts_dir() / 'rf_corrected_labels/2025-11-14T03-20-00Z/metrics.json'}")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]

    if not Path(old_path).exists():
        print(f"Error: Old metrics file not found: {old_path}")
        sys.exit(1)

    if not Path(new_path).exists():
        print(f"Error: New metrics file not found: {new_path}")
        sys.exit(1)

    compare_versions(old_path, new_path)
