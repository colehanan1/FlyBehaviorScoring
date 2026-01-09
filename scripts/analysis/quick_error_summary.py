#!/usr/bin/env python
"""Quick error summary from prediction CSVs - no dependencies needed!"""

import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict


def analyze_predictions(pred_file):
    """Analyze a predictions CSV file."""

    with open(pred_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    correct = sum(1 for r in rows if r['correct'] == 'True')
    incorrect = total - correct

    # Categorize errors
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []

    for row in rows:
        true_label = int(row['user_score_odor'])
        pred_label = int(row['predicted_label'])

        if true_label == 0 and pred_label == 1:
            false_positives.append(row)
        elif true_label > 0 and pred_label == 0:
            false_negatives.append(row)
        elif true_label > 0 and pred_label == 1:
            true_positives.append(row)
        elif true_label == 0 and pred_label == 0:
            true_negatives.append(row)

    n_tp = len(true_positives)
    n_tn = len(true_negatives)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': correct / total if total > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': n_tp,
        'tn': n_tn,
        'fp': n_fp,
        'fn': n_fn,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }


def main():
    parser = argparse.ArgumentParser(description='Quick error summary from predictions')
    parser.add_argument('--run-dir', type=Path, required=True, help='Run directory')
    parser.add_argument('--model', type=str, required=True, help='Model name')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {args.model.upper()}")
    print(f"{'='*70}\n")

    for split in ['train', 'validation', 'test']:
        pred_file = args.run_dir / f"predictions_{args.model}_{split}.csv"

        if not pred_file.exists():
            continue

        results = analyze_predictions(pred_file)

        print(f"{split.upper()} SET ({results['total']} samples):")
        print(f"  Accuracy:  {results['accuracy']:6.1%}")
        print(f"  Precision: {results['precision']:6.1%}")
        print(f"  Recall:    {results['recall']:6.1%}")
        print(f"  F1 Score:  {results['f1']:6.1%}")
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {results['tn']:3d}")
        print(f"    False Positives: {results['fp']:3d} {'⚠️' if results['fp'] > 0 else '✓'}")
        print(f"    False Negatives: {results['fn']:3d} {'⚠️' if results['fn'] > 0 else '✓'}")
        print(f"    True Positives:  {results['tp']:3d}")

        # Show error details
        if results['fp'] > 0:
            print(f"\n  FALSE POSITIVES (predicted Yes, actually No):")
            for row in results['false_positives'][:5]:  # Show first 5
                prob = float(row.get('prob_reaction', 0))
                print(f"    - {row.get('fly', 'unknown')}/{row.get('trial_label', '?')} "
                      f"(confidence: {prob:.2%})")
            if len(results['false_positives']) > 5:
                print(f"    ... and {len(results['false_positives'])-5} more")

        if results['fn'] > 0:
            print(f"\n  FALSE NEGATIVES (predicted No, actually Yes):")
            for row in results['false_negatives'][:5]:  # Show first 5
                intensity = row.get('user_score_odor_intensity', '?')
                prob = float(row.get('prob_reaction', 0))
                print(f"    - {row.get('fly', 'unknown')}/{row.get('trial_label', '?')} "
                      f"(intensity: {intensity}, confidence: {prob:.2%})")
            if len(results['false_negatives']) > 5:
                print(f"    ... and {len(results['false_negatives'])-5} more")

        print()

    print(f"{'='*70}\n")
    print(f"📁 Detailed error CSVs available at:")
    print(f"   {args.run_dir}/predictions_{args.model}_*.csv")
    print()


if __name__ == '__main__':
    main()
