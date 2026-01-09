#!/usr/bin/env python3
"""Threshold slider analysis tool for multi-class predictions.

This tool allows post-hoc adjustment of classification thresholds to analyze
how different reaction strength cutoffs affect model performance.

Example usage:
    # Train a multiclass model first
    flybehavior-response train --data-csv data.csv --labels-csv labels.csv \\
        --classification-mode multiclass --model mlp --artifacts-dir outputs/multiclass_artifacts

    # Then analyze thresholds
    python threshold_slider.py --predictions-csv outputs/multiclass_artifacts/latest/predictions_mlp_test.csv \\
        --min-threshold 1 --max-threshold 5
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    """Load predictions CSV file."""
    df = pd.read_csv(predictions_csv)

    # Check required columns (actual column names from train.py)
    if 'user_score_odor' not in df.columns:
        raise ValueError(f"predictions_csv must contain 'user_score_odor' column (true labels)")
    if 'predicted_label' not in df.columns:
        raise ValueError(f"predictions_csv must contain 'predicted_label' column (predictions)")

    return df


def apply_threshold(predictions: pd.Series, threshold: int) -> pd.Series:
    """Apply binary threshold to multi-class predictions.

    Args:
        predictions: Multi-class predictions (0-5)
        threshold: Minimum value to be considered 'reactive' (1)

    Returns:
        Binary predictions: 0 = below threshold, 1 = at or above threshold
    """
    return (predictions >= threshold).astype(int)


def compute_metrics(y_true_binary: pd.Series, y_pred_binary: pd.Series) -> dict:
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }


def analyze_thresholds(df: pd.DataFrame, min_threshold: int = 1, max_threshold: int = 5):
    """Analyze model performance across different thresholds.

    Args:
        df: DataFrame with 'user_score_odor' and 'predicted_label' columns
        min_threshold: Minimum threshold to test (inclusive)
        max_threshold: Maximum threshold to test (inclusive)
    """

    print("="*100)
    print("THRESHOLD SLIDER ANALYSIS")
    print("="*100)
    print(f"\nAnalyzing thresholds from {min_threshold} to {max_threshold}")
    print(f"Total samples: {len(df)}")

    # Show original label distribution
    print("\nOriginal label distribution:")
    label_dist = df['user_score_odor'].value_counts().sort_index()
    for label, count in label_dist.items():
        pct = 100 * count / len(df)
        print(f"  Label {label}: {count:4d} ({pct:5.1f}%)")

    # Test each threshold
    results = []

    for threshold in range(min_threshold, max_threshold + 1):
        # Convert predictions to binary based on threshold
        y_true_binary = apply_threshold(df['user_score_odor'], threshold)
        y_pred_binary = apply_threshold(df['predicted_label'], threshold)

        # Compute metrics
        metrics = compute_metrics(y_true_binary, y_pred_binary)

        # Count positives
        n_true_positive = y_true_binary.sum()
        n_pred_positive = y_pred_binary.sum()

        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'true_positives': n_true_positive,
            'pred_positives': n_pred_positive,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        })

    # Display results table
    print("\n" + "="*100)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*100)
    print(f"\n{'Threshold':<11} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'True+':>7} {'Pred+':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    print("-" * 100)

    for r in results:
        print(f"{r['threshold']:<11} {r['accuracy']:>10.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>10.3f} {r['f1']:>10.3f} {r['true_positives']:>7d} "
              f"{r['pred_positives']:>7d} {r['tp']:>6d} {r['fp']:>6d} "
              f"{r['tn']:>6d} {r['fn']:>6d}")

    # Find best thresholds for different metrics
    print("\n" + "="*100)
    print("BEST THRESHOLDS BY METRIC")
    print("="*100)

    results_df = pd.DataFrame(results)

    best_accuracy_idx = results_df['accuracy'].idxmax()
    best_f1_idx = results_df['f1'].idxmax()
    best_precision_idx = results_df['precision'].idxmax()
    best_recall_idx = results_df['recall'].idxmax()

    print(f"\nBest Accuracy:  threshold = {results_df.loc[best_accuracy_idx, 'threshold']:.0f}, "
          f"accuracy = {results_df.loc[best_accuracy_idx, 'accuracy']:.3f}")
    print(f"Best F1 Score:  threshold = {results_df.loc[best_f1_idx, 'threshold']:.0f}, "
          f"F1 = {results_df.loc[best_f1_idx, 'f1']:.3f}")
    print(f"Best Precision: threshold = {results_df.loc[best_precision_idx, 'threshold']:.0f}, "
          f"precision = {results_df.loc[best_precision_idx, 'precision']:.3f}")
    print(f"Best Recall:    threshold = {results_df.loc[best_recall_idx, 'threshold']:.0f}, "
          f"recall = {results_df.loc[best_recall_idx, 'recall']:.3f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Analysis: Performance Metrics vs Threshold', fontsize=14, fontweight='bold')

    # Plot 1: Accuracy, Precision, Recall, F1
    ax = axes[0, 0]
    thresholds = results_df['threshold']
    ax.plot(thresholds, results_df['accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(thresholds, results_df['precision'], 's-', label='Precision', linewidth=2)
    ax.plot(thresholds, results_df['recall'], '^-', label='Recall', linewidth=2)
    ax.plot(thresholds, results_df['f1'], 'd-', label='F1 Score', linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thresholds)

    # Plot 2: Number of positives
    ax = axes[0, 1]
    ax.plot(thresholds, results_df['true_positives'], 'o-', label='True Positives (Ground Truth)', linewidth=2)
    ax.plot(thresholds, results_df['pred_positives'], 's-', label='Predicted Positives', linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Number of Positive Classifications', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thresholds)

    # Plot 3: Confusion matrix components
    ax = axes[1, 0]
    ax.plot(thresholds, results_df['tp'], 'o-', label='True Positives', linewidth=2, color='green')
    ax.plot(thresholds, results_df['fp'], 's-', label='False Positives', linewidth=2, color='red')
    ax.plot(thresholds, results_df['tn'], '^-', label='True Negatives', linewidth=2, color='blue')
    ax.plot(thresholds, results_df['fn'], 'd-', label='False Negatives', linewidth=2, color='orange')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confusion Matrix Components', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thresholds)

    # Plot 4: Precision-Recall tradeoff
    ax = axes[1, 1]
    ax.plot(results_df['recall'], results_df['precision'], 'o-', linewidth=2, markersize=8)
    for i, thresh in enumerate(thresholds):
        ax.annotate(f'{thresh}',
                   (results_df.loc[i, 'recall'], results_df.loc[i, 'precision']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # Save plot
    output_path = Path('threshold_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze classification performance across different thresholds for multi-class predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--predictions-csv',
        type=Path,
        required=True,
        help="Path to predictions CSV file (must contain 'user_score_odor' and 'predicted_label' columns)"
    )
    parser.add_argument(
        '--min-threshold',
        type=int,
        default=1,
        help="Minimum threshold to test (values >= threshold are considered 'reactive')"
    )
    parser.add_argument(
        '--max-threshold',
        type=int,
        default=5,
        help="Maximum threshold to test (values >= threshold are considered 'reactive')"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.predictions_csv.exists():
        print(f"Error: predictions CSV not found: {args.predictions_csv}")
        sys.exit(1)

    if args.min_threshold < 1 or args.max_threshold > 5:
        print(f"Error: thresholds must be in range [1, 5]")
        sys.exit(1)

    if args.min_threshold > args.max_threshold:
        print(f"Error: min_threshold must be <= max_threshold")
        sys.exit(1)

    # Load predictions
    print(f"Loading predictions from: {args.predictions_csv}")
    df = load_predictions(args.predictions_csv)

    # Analyze thresholds
    results = analyze_thresholds(df, args.min_threshold, args.max_threshold)

    # Save results to CSV
    output_csv = Path('threshold_analysis_results.csv')
    results.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    print("\n" + "="*100)
    print("DONE")
    print("="*100)


if __name__ == "__main__":
    main()
