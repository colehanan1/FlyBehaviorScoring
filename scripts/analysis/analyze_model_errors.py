#!/usr/bin/env python
"""Analyze model errors and create detailed error reports.

NOTE: Run this with your conda environment activated:
    conda activate flybehavior
    python scripts/analysis/analyze_model_errors.py --run-dir <dir> --model <model>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_predictions(run_dir: Path, model: str, split: str) -> pd.DataFrame:
    """Load predictions CSV for a specific model and split."""
    pred_file = run_dir / f"predictions_{model}_{split}.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    return pd.read_csv(pred_file)


def analyze_errors(df: pd.DataFrame, split_name: str) -> dict:
    """Analyze prediction errors and return detailed statistics."""

    # Basic metrics
    total = len(df)
    correct = df['correct'].sum()
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0

    # Separate by error type
    false_positives = df[(df['user_score_odor'] == 0) & (df['predicted_label'] == 1)]
    false_negatives = df[(df['user_score_odor'] > 0) & (df['predicted_label'] == 0)]
    true_positives = df[(df['user_score_odor'] > 0) & (df['predicted_label'] == 1)]
    true_negatives = df[(df['user_score_odor'] == 0) & (df['predicted_label'] == 0)]

    # Calculate metrics
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    n_tp = len(true_positives)
    n_tn = len(true_negatives)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    stats = {
        'split': split_name,
        'total_samples': total,
        'correct': int(correct),
        'incorrect': int(incorrect),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': n_tp,
        'true_negatives': n_tn,
        'false_positives': n_fp,
        'false_negatives': n_fn,
    }

    # Analyze false positives by probability
    if 'prob_reaction' in df.columns and n_fp > 0:
        fp_probs = false_positives['prob_reaction'].values
        stats['fp_mean_confidence'] = float(np.mean(fp_probs))
        stats['fp_median_confidence'] = float(np.median(fp_probs))
        stats['fp_min_confidence'] = float(np.min(fp_probs))
        stats['fp_max_confidence'] = float(np.max(fp_probs))

    # Analyze false negatives by probability
    if 'prob_reaction' in df.columns and n_fn > 0:
        fn_probs = false_negatives['prob_reaction'].values
        stats['fn_mean_confidence'] = float(np.mean(fn_probs))
        stats['fn_median_confidence'] = float(np.median(fn_probs))
        stats['fn_min_confidence'] = float(np.min(fn_probs))
        stats['fn_max_confidence'] = float(np.max(fn_probs))

    # Analyze by intensity (for false negatives)
    if 'user_score_odor_intensity' in df.columns and n_fn > 0:
        fn_intensities = false_negatives['user_score_odor_intensity'].value_counts().sort_index()
        stats['fn_by_intensity'] = fn_intensities.to_dict()

    return stats, false_positives, false_negatives, true_positives, true_negatives


def create_error_report(run_dir: Path, model: str, output_dir: Path):
    """Create comprehensive error analysis report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions for all splits
    splits = []
    all_stats = {}

    for split in ['train', 'validation', 'test']:
        try:
            df = load_predictions(run_dir, model, split)
            splits.append(split)

            # Analyze errors
            stats, fp, fn, tp, tn = analyze_errors(df, split)
            all_stats[split] = stats

            # Save error details
            if len(fp) > 0:
                fp_file = output_dir / f"errors_{model}_{split}_false_positives.csv"
                fp.to_csv(fp_file, index=False)
                print(f"✅ Saved {len(fp)} false positives to {fp_file}")

            if len(fn) > 0:
                fn_file = output_dir / f"errors_{model}_{split}_false_negatives.csv"
                fn.to_csv(fn_file, index=False)
                print(f"✅ Saved {len(fn)} false negatives to {fn_file}")

        except FileNotFoundError:
            continue

    # Save summary statistics
    summary_file = output_dir / f"error_summary_{model}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✅ Saved error summary to {summary_file}")

    # Create visualizations
    create_error_visualizations(run_dir, model, splits, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS SUMMARY - {model.upper()}")
    print(f"{'='*60}\n")

    for split in splits:
        stats = all_stats[split]
        print(f"{split.upper()} SET:")
        print(f"  Accuracy:  {stats['accuracy']:.1%}")
        print(f"  Precision: {stats['precision']:.1%}")
        print(f"  Recall:    {stats['recall']:.1%}")
        print(f"  F1 Score:  {stats['f1_score']:.1%}")
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {stats['true_negatives']:3d}")
        print(f"    False Positives: {stats['false_positives']:3d} ⚠️")
        print(f"    False Negatives: {stats['false_negatives']:3d} ⚠️")
        print(f"    True Positives:  {stats['true_positives']:3d}")

        if 'fp_mean_confidence' in stats:
            print(f"\n  False Positive Confidence: {stats['fp_mean_confidence']:.3f} ± {stats['fp_median_confidence']:.3f}")
        if 'fn_mean_confidence' in stats:
            print(f"  False Negative Confidence: {stats['fn_mean_confidence']:.3f} ± {stats['fn_median_confidence']:.3f}")
        if 'fn_by_intensity' in stats:
            print(f"\n  False Negatives by Intensity: {stats['fn_by_intensity']}")
        print()

    return all_stats


def create_error_visualizations(run_dir: Path, model: str, splits: list, output_dir: Path):
    """Create visualizations showing model errors."""

    # Set style
    sns.set_style("whitegrid")

    # 1. Error comparison across splits
    fig, axes = plt.subplots(1, len(splits), figsize=(5*len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    for idx, split in enumerate(splits):
        df = load_predictions(run_dir, model, split)

        # Create confusion matrix
        y_true = (df['user_score_odor'] > 0).astype(int)
        y_pred = df['predicted_label']
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Pred: No', 'Pred: Yes'],
                   yticklabels=['True: No', 'True: Yes'])
        ax.set_title(f'{split.capitalize()} Set\n{model.upper()}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    viz_file = output_dir / f"confusion_matrices_{model}.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved confusion matrices to {viz_file}")
    plt.close()

    # 2. Probability distribution for errors (test set only)
    try:
        test_df = load_predictions(run_dir, model, 'test')
        if 'prob_reaction' in test_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # True positives
            tp = test_df[(test_df['user_score_odor'] > 0) & (test_df['predicted_label'] == 1)]
            if len(tp) > 0:
                axes[0, 0].hist(tp['prob_reaction'], bins=20, color='green', alpha=0.7)
                axes[0, 0].set_title(f'True Positives (n={len(tp)})')
                axes[0, 0].set_xlabel('Predicted Probability')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
                axes[0, 0].legend()

            # True negatives
            tn = test_df[(test_df['user_score_odor'] == 0) & (test_df['predicted_label'] == 0)]
            if len(tn) > 0:
                axes[0, 1].hist(tn['prob_reaction'], bins=20, color='blue', alpha=0.7)
                axes[0, 1].set_title(f'True Negatives (n={len(tn)})')
                axes[0, 1].set_xlabel('Predicted Probability')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
                axes[0, 1].legend()

            # False positives
            fp = test_df[(test_df['user_score_odor'] == 0) & (test_df['predicted_label'] == 1)]
            if len(fp) > 0:
                axes[1, 0].hist(fp['prob_reaction'], bins=20, color='orange', alpha=0.7)
                axes[1, 0].set_title(f'False Positives (n={len(fp)}) ⚠️')
                axes[1, 0].set_xlabel('Predicted Probability')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No False Positives! 🎉',
                               ha='center', va='center', fontsize=14)
                axes[1, 0].set_xlim(0, 1)

            # False negatives
            fn = test_df[(test_df['user_score_odor'] > 0) & (test_df['predicted_label'] == 0)]
            if len(fn) > 0:
                axes[1, 1].hist(fn['prob_reaction'], bins=20, color='red', alpha=0.7)
                axes[1, 1].set_title(f'False Negatives (n={len(fn)}) ⚠️')
                axes[1, 1].set_xlabel('Predicted Probability')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'No False Negatives! 🎉',
                               ha='center', va='center', fontsize=14)
                axes[1, 1].set_xlim(0, 1)

            plt.suptitle(f'Prediction Probability Distribution - {model.upper()} (Test Set)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            prob_file = output_dir / f"probability_distributions_{model}.png"
            plt.savefig(prob_file, dpi=150, bbox_inches='tight')
            print(f"✅ Saved probability distributions to {prob_file}")
            plt.close()
    except FileNotFoundError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Analyze model errors and create detailed reports'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to the run directory containing predictions'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lda', 'logreg', 'random_forest', 'mlp', 'fp_optimized_mlp'],
        help='Model to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for error reports (default: run-dir/error_analysis)'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.run_dir / 'error_analysis'

    print(f"\n{'='*60}")
    print(f"ANALYZING ERRORS FOR {args.model.upper()}")
    print(f"{'='*60}\n")
    print(f"Run directory: {args.run_dir}")
    print(f"Output directory: {args.output_dir}\n")

    create_error_report(args.run_dir, args.model, args.output_dir)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")
    print(f"📁 Check {args.output_dir} for:")
    print(f"  - Error CSVs (false positives & negatives)")
    print(f"  - Visualizations (confusion matrices, probability distributions)")
    print(f"  - JSON summary (detailed statistics)")
    print()


if __name__ == '__main__':
    main()
