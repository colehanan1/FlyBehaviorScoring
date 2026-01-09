#!/usr/bin/env python3
"""
Analyze misclassified test samples from Random Forest model.

This script helps identify:
1. Which flies are hardest to predict
2. Whether misclassifications cluster by feature values
3. Potential label quality issues
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import artifacts_dir  # noqa: E402


def analyze_misclassifications(predictions_csv: str):
    """Analyze the misclassified test samples."""

    # Load predictions
    df = pd.read_csv(predictions_csv)

    # Filter to misclassified samples
    errors = df[df['correct'] == False].copy()

    print(f"\n{'='*80}")
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal test samples: {len(df)}")
    print(f"Misclassified: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")
    print(f"Accuracy: {(len(df) - len(errors))/len(df)*100:.1f}%")

    # Breakdown by error type
    false_positives = errors[errors['predicted_label'] == 1]
    false_negatives = errors[errors['predicted_label'] == 0]

    print(f"\nError Breakdown:")
    print(f"  False Positives (predicted reaction, but none): {len(false_positives)}")
    print(f"  False Negatives (missed reactions): {len(false_negatives)}")

    # Analyze by fly
    if 'fly' in errors.columns:
        print(f"\n{'='*80}")
        print("ERRORS BY FLY:")
        print(f"{'='*80}")
        error_counts = errors.groupby('fly').size().sort_values(ascending=False)
        for fly, count in error_counts.items():
            fly_total = len(df[df['fly'] == fly])
            print(f"  {fly}: {count}/{fly_total} errors ({count/fly_total*100:.1f}% error rate)")

    # Show detailed error information
    print(f"\n{'='*80}")
    print("DETAILED MISCLASSIFICATIONS:")
    print(f"{'='*80}")

    # Define columns to show
    display_cols = ['fly', 'trial', 'label', 'predicted_label', 'prob_reaction']

    # Add any feature columns if they exist
    feature_cols = [col for col in df.columns if col in [
        'global_max', 'local_min', 'local_max', 'local_max_before',
        'local_max_during', 'local_max_during_over_global_min',
        'AUC-During', 'Peak-Value'
    ]]

    if 'label_intensity' in errors.columns:
        display_cols.insert(3, 'label_intensity')

    available_cols = [col for col in display_cols if col in errors.columns]
    available_cols.extend(feature_cols)

    print("\nFalse Positives (model said reaction, actually none):")
    if len(false_positives) > 0:
        print(false_positives[available_cols].to_string(index=False))
    else:
        print("  None")

    print("\nFalse Negatives (model missed reaction):")
    if len(false_negatives) > 0:
        print(false_negatives[available_cols].to_string(index=False))
    else:
        print("  None")

    # Feature analysis for errors
    if feature_cols:
        print(f"\n{'='*80}")
        print("FEATURE PATTERNS IN ERRORS:")
        print(f"{'='*80}")

        correct_samples = df[df['correct'] == True]

        print("\nMean feature values:")
        print(f"{'Feature':<35} {'Correct':>12} {'Errors':>12} {'Difference':>12}")
        print("-" * 80)
        for col in feature_cols:
            if col in errors.columns:
                correct_mean = correct_samples[col].mean()
                error_mean = errors[col].mean()
                diff = error_mean - correct_mean
                print(f"{col:<35} {correct_mean:>12.3f} {error_mean:>12.3f} {diff:>+12.3f}")

    # Probability distribution analysis
    if 'prob_reaction' in errors.columns:
        print(f"\n{'='*80}")
        print("PROBABILITY CONFIDENCE ANALYSIS:")
        print(f"{'='*80}")

        print("\nPrediction confidence for misclassified samples:")
        print(f"  Mean prob_reaction: {errors['prob_reaction'].mean():.3f}")
        print(f"  Min prob_reaction: {errors['prob_reaction'].min():.3f}")
        print(f"  Max prob_reaction: {errors['prob_reaction'].max():.3f}")

        # Classify errors by confidence
        uncertain = errors[(errors['prob_reaction'] > 0.3) & (errors['prob_reaction'] < 0.7)]
        print(f"\n  Uncertain predictions (0.3 < prob < 0.7): {len(uncertain)}")
        if len(uncertain) > 0:
            print(f"    → These are edge cases, might need label review")

        confident_wrong = errors[(errors['prob_reaction'] <= 0.3) | (errors['prob_reaction'] >= 0.7)]
        print(f"  Confident but wrong (prob < 0.3 or > 0.7): {len(confident_wrong)}")
        if len(confident_wrong) > 0:
            print(f"    → These might indicate label errors")

    # Save detailed error report
    output_path = Path(predictions_csv).parent / "error_analysis.csv"
    errors.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed error report saved to: {output_path}")
    print(f"{'='*80}\n")

    return errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <path_to_predictions_test.csv>")
        print("\nExample:")
        print(
            "  python analyze_errors.py "
            f"{artifacts_dir() / 'rf_tuned/2025-11-14T01-47-01Z/predictions_random_forest_test.csv'}"
        )
        sys.exit(1)

    predictions_csv = sys.argv[1]

    if not Path(predictions_csv).exists():
        print(f"Error: File not found: {predictions_csv}")
        sys.exit(1)

    analyze_misclassifications(predictions_csv)
