#!/usr/bin/env python3
"""
Find optimal prediction threshold for Random Forest to catch subtle reactions.

This script analyzes the test set predictions and finds the threshold
that maximizes recall (catches more reactions) while maintaining
acceptable precision.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import artifacts_dir  # noqa: E402

def find_optimal_threshold(predictions_csv: str, strategy: str = "f1"):
    """
    Find optimal probability threshold for classification.

    Parameters
    ----------
    predictions_csv : str
        Path to predictions CSV with 'prob_reaction' column
    strategy : str
        Optimization strategy:
        - 'f1': Maximize F1 score (balanced)
        - 'recall': Maximize recall while maintaining precision > 0.85
        - 'precision': Maximize precision while maintaining recall > 0.85
        - 'youden': Maximize Youden's J statistic (sensitivity + specificity - 1)
    """

    df = pd.read_csv(predictions_csv)

    if 'prob_reaction' not in df.columns:
        print("❌ ERROR: No 'prob_reaction' column found in predictions file!")
        print(f"Available columns: {list(df.columns)}")
        return

    if 'label' not in df.columns:
        print("❌ ERROR: No 'label' column found!")
        return

    y_true = df['label'].values
    y_proba = df['prob_reaction'].values

    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR SUBTLE REACTION DETECTION")
    print("="*80)

    print(f"\nTest set size: {len(df)}")
    print(f"Actual reactions: {y_true.sum()} ({y_true.mean():.1%})")
    print(f"Actual non-reactions: {(1-y_true).sum()} ({(1-y_true).mean():.1%})")

    # Current performance (threshold = 0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)

    tp_default = ((y_pred_default == 1) & (y_true == 1)).sum()
    fp_default = ((y_pred_default == 1) & (y_true == 0)).sum()
    fn_default = ((y_pred_default == 0) & (y_true == 1)).sum()
    tn_default = ((y_pred_default == 0) & (y_true == 0)).sum()

    precision_default = tp_default / (tp_default + fp_default) if (tp_default + fp_default) > 0 else 0
    recall_default = tp_default / (tp_default + fn_default) if (tp_default + fn_default) > 0 else 0
    f1_default = 2 * precision_default * recall_default / (precision_default + recall_default) if (precision_default + recall_default) > 0 else 0

    print("\n" + "="*80)
    print("CURRENT PERFORMANCE (threshold = 0.5)")
    print("="*80)
    print(f"Precision: {precision_default:.3f}")
    print(f"Recall:    {recall_default:.3f}")
    print(f"F1 Score:  {f1_default:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn_default:<4} FP: {fp_default}")
    print(f"  FN: {fn_default:<4} TP: {tp_default}")
    print(f"\nMissed reactions (FN): {fn_default}")
    print(f"False alarms (FP):     {fp_default}")

    # Analyze missed reactions
    if fn_default > 0:
        missed = df[(y_proba < 0.5) & (df['label'] == 1)].copy()
        print(f"\n{'='*80}")
        print(f"ANALYSIS OF {len(missed)} MISSED REACTIONS (False Negatives)")
        print(f"{'='*80}")
        print(f"Probability distribution of missed reactions:")
        print(f"  Min:  {missed['prob_reaction'].min():.3f}")
        print(f"  25th: {missed['prob_reaction'].quantile(0.25):.3f}")
        print(f"  50th: {missed['prob_reaction'].quantile(0.50):.3f}")
        print(f"  75th: {missed['prob_reaction'].quantile(0.75):.3f}")
        print(f"  Max:  {missed['prob_reaction'].max():.3f}")

        close_calls = missed[missed['prob_reaction'] > 0.4]
        print(f"\n  → {len(close_calls)} missed reactions had prob > 0.4 (close to threshold)")
        print(f"    These could be caught by lowering threshold!")

    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        youden_j = recall + specificity - 1

        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'youden_j': youden_j,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        })

    results_df = pd.DataFrame(results)

    # Find optimal threshold based on strategy
    if strategy == 'f1':
        best_idx = results_df['f1'].idxmax()
        print(f"\n{'='*80}")
        print(f"OPTIMAL THRESHOLD (Strategy: Maximize F1)")
        print(f"{'='*80}")
    elif strategy == 'recall':
        # Maximize recall while maintaining precision > 0.85
        candidates = results_df[results_df['precision'] >= 0.85]
        if len(candidates) > 0:
            best_idx = candidates['recall'].idxmax()
        else:
            # If can't maintain 0.85 precision, find best F1
            best_idx = results_df['f1'].idxmax()
            print("\n⚠️  Cannot maintain precision ≥ 0.85; falling back to F1 optimization")
        print(f"\n{'='*80}")
        print(f"OPTIMAL THRESHOLD (Strategy: Maximize Recall, precision ≥ 0.85)")
        print(f"{'='*80}")
    elif strategy == 'precision':
        # Maximize precision while maintaining recall > 0.85
        candidates = results_df[results_df['recall'] >= 0.85]
        if len(candidates) > 0:
            best_idx = candidates['precision'].idxmax()
        else:
            best_idx = results_df['f1'].idxmax()
            print("\n⚠️  Cannot maintain recall ≥ 0.85; falling back to F1 optimization")
        print(f"\n{'='*80}")
        print(f"OPTIMAL THRESHOLD (Strategy: Maximize Precision, recall ≥ 0.85)")
        print(f"{'='*80}")
    elif strategy == 'youden':
        best_idx = results_df['youden_j'].idxmax()
        print(f"\n{'='*80}")
        print(f"OPTIMAL THRESHOLD (Strategy: Youden's J)")
        print(f"{'='*80}")
    else:
        best_idx = results_df['f1'].idxmax()
        print(f"\n{'='*80}")
        print(f"OPTIMAL THRESHOLD (Strategy: Maximize F1)")
        print(f"{'='*80}")

    best = results_df.iloc[best_idx]

    print(f"\nOptimal threshold: {best['threshold']:.3f}")
    print(f"\nPerformance at threshold = {best['threshold']:.3f}:")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall:    {best['recall']:.3f}")
    print(f"  F1 Score:  {best['f1']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {int(best['tn']):<4} FP: {int(best['fp'])}")
    print(f"  FN: {int(best['fn']):<4} TP: {int(best['tp'])}")
    print(f"\nMissed reactions (FN): {int(best['fn'])} (was {fn_default})")
    print(f"False alarms (FP):     {int(best['fp'])} (was {fp_default})")

    # Calculate improvement
    fn_reduction = fn_default - int(best['fn'])
    fp_increase = int(best['fp']) - fp_default

    print(f"\n{'='*80}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Threshold change: 0.500 → {best['threshold']:.3f}")
    print(f"\nCatches {fn_reduction} MORE subtle reactions")
    print(f"But adds {fp_increase} MORE false alarms")
    print(f"\nF1 Score change: {f1_default:.3f} → {best['f1']:.3f} ({best['f1'] - f1_default:+.3f})")
    print(f"Recall change:   {recall_default:.3f} → {best['recall']:.3f} ({best['recall'] - recall_default:+.3f})")
    print(f"Precision change: {precision_default:.3f} → {best['precision']:.3f} ({best['precision'] - precision_default:+.3f})")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if best['threshold'] < 0.5:
        print(f"\n✅ Lower threshold to {best['threshold']:.3f} to catch subtle reactions")
        print(f"\nTo apply this threshold when predicting on new data:")
        print(f"```bash")
        print(f"flybehavior-response predict \\")
        print(
            "  --model-path "
            f"{artifacts_dir() / 'rf_corrected_labels_v1/2025-11-14T18-58-28Z/model_random_forest.joblib'} \\"
        )
        print(f"  --data-csv /path/to/unlabeled_data.csv \\")
        print(f"  --threshold {best['threshold']:.3f} \\")
        print(f"  --output predictions_with_custom_threshold.csv")
        print(f"```")
    else:
        print(f"\n✅ Current threshold (0.5) is already optimal")
        print(f"   → Model is well-calibrated")
        print(f"   → To catch more subtle reactions, you need more training data")

    # Alternative thresholds
    print(f"\n{'='*80}")
    print("ALTERNATIVE THRESHOLDS TO CONSIDER")
    print(f"{'='*80}")

    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FN':<6} {'FP':<6}")
    print("-"*80)

    # Show a few key thresholds
    key_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    for thresh in key_thresholds:
        row = results_df[results_df['threshold'].round(2) == thresh]
        if len(row) > 0:
            r = row.iloc[0]
            marker = " ← OPTIMAL" if abs(r['threshold'] - best['threshold']) < 0.01 else ""
            print(f"{r['threshold']:<12.2f} {r['precision']:<12.3f} {r['recall']:<12.3f} {r['f1']:<12.3f} {int(r['fn']):<6} {int(r['fp']):<6}{marker}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Precision-Recall curve
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
    ax.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
    ax.plot(results_df['threshold'], results_df['f1'], label='F1 Score', linewidth=2, linestyle='--')
    ax.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
    ax.axvline(best['threshold'], color='red', linestyle='--', label=f'Optimal ({best["threshold"]:.3f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1 vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FN and FP counts
    ax = axes[0, 1]
    ax.plot(results_df['threshold'], results_df['fn'], label='False Negatives (missed)', linewidth=2, color='orange')
    ax.plot(results_df['threshold'], results_df['fp'], label='False Positives (false alarms)', linewidth=2, color='purple')
    ax.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
    ax.axvline(best['threshold'], color='red', linestyle='--', label=f'Optimal ({best["threshold"]:.3f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Error Counts vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Probability histogram
    ax = axes[1, 0]
    reactions = df[df['label'] == 1]['prob_reaction']
    non_reactions = df[df['label'] == 0]['prob_reaction']

    ax.hist(non_reactions, bins=20, alpha=0.5, label='No Reaction (label=0)', color='blue')
    ax.hist(reactions, bins=20, alpha=0.5, label='Reaction (label=1)', color='red')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, label='Default threshold')
    ax.axvline(best['threshold'], color='red', linestyle='--', linewidth=2, label=f'Optimal ({best["threshold"]:.3f})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution by True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC curve
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir = Path(predictions_csv).parent
    plot_path = output_dir / "threshold_optimization.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plots saved to: {plot_path}")
    print(f"{'='*80}")

    # Save results
    results_path = output_dir / "threshold_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")

    return best['threshold']


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python find_optimal_threshold.py <predictions_test.csv> [strategy]")
        print("\nStrategy options:")
        print("  f1       - Maximize F1 score (default, balanced)")
        print("  recall   - Maximize recall (catch more reactions)")
        print("  precision - Maximize precision (fewer false alarms)")
        print("  youden   - Maximize Youden's J statistic")
        print("\nExample:")
        print("  python find_optimal_threshold.py \\")
        print(
            f"    {artifacts_dir() / 'rf_corrected_labels_v1/2025-11-14T18-58-28Z/predictions_random_forest_test.csv'} \\"
        )
        print("    recall")
        sys.exit(1)

    predictions_csv = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else 'f1'

    if not Path(predictions_csv).exists():
        print(f"❌ ERROR: File not found: {predictions_csv}")
        sys.exit(1)

    optimal_threshold = find_optimal_threshold(predictions_csv, strategy)
