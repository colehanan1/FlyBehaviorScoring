#!/usr/bin/env python3
"""Comprehensive test of all classification modes with all models."""

from pathlib import Path
import sys
import json

import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import data_dir, ensure_src_on_path, outputs_dir, resolve_path  # noqa: E402

ensure_src_on_path()

from flybehavior_response.train import train_models  # noqa: E402

# Configuration from reference file
data_path = resolve_path(data_dir() / "all_envelope_rows_wide.csv")
labels_path = resolve_path(data_dir() / "scoring_results_opto_new_MINIMAL.csv")
run_root = outputs_dir() / "test_all_modes"

FEATURES = ["AUC-During", "TimeToPeak-During", "Peak-Value"]
N_PCS = 10
SEED = 67
ALL_MODELS = ["lda", "logreg", "random_forest", "mlp", "fp_optimized_mlp"]

MODES = [
    ("binary", "Binary (0 vs 1-5)"),
    ("multiclass", "Multi-class (0-5)"),
    ("threshold-2", "Threshold-2 (0-2 vs 3-5)"),
]

print("=" * 120)
print("COMPREHENSIVE CLASSIFICATION MODE TESTING")
print("=" * 120)
print(f"\nConfiguration:")
print(f"  Data: {data_path.name}")
print(f"  Labels: {labels_path.name}")
print(f"  Features: {FEATURES}")
print(f"  n_pcs: {N_PCS}")
print(f"  Seed: {SEED}")
print(f"  Models: {ALL_MODELS}")
print(f"\nTesting {len(MODES)} classification modes × {len(ALL_MODELS)} models = {len(MODES) * len(ALL_MODELS)} total runs")

results_summary = {}

for mode, mode_desc in MODES:
    print("\n" + "=" * 120)
    print(f"CLASSIFICATION MODE: {mode_desc}")
    print("=" * 120)

    try:
        results = train_models(
            data_csv=data_path,
            labels_csv=labels_path,
            features=FEATURES,
            trace_prefixes=None,
            include_traces=True,
            use_raw_pca=True,
            n_pcs=N_PCS,
            models=ALL_MODELS,
            cv=0,  # Skip CV for speed
            artifacts_dir=run_root / mode,
            seed=SEED,
            verbose=False,
            dry_run=False,
            classification_mode=mode,
        )

        # Extract metrics for each model
        mode_results = {}
        for model_name in ALL_MODELS:
            if model_name in results['models']:
                model_metrics = results['models'][model_name]
                train_acc = model_metrics['accuracy']
                test_acc = model_metrics['test']['accuracy']
                train_f1 = model_metrics.get('f1_binary', model_metrics.get('f1_macro', 0))
                test_f1 = model_metrics['test'].get('f1_binary', model_metrics['test'].get('f1_macro', 0))

                # Handle None values for multiclass f1_binary
                if train_f1 is None:
                    train_f1 = model_metrics.get('f1_macro', 0)
                if test_f1 is None:
                    test_f1 = model_metrics['test'].get('f1_macro', 0)

                mode_results[model_name] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'overfitting_gap': train_acc - test_acc,
                }

        results_summary[mode] = mode_results

        print(f"\n✓ {mode_desc} completed successfully!")
        print(f"\nResults by model:")
        print(f"{'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8} {'Train F1':>10} {'Test F1':>10}")
        print("-" * 80)

        for model_name in ALL_MODELS:
            if model_name in mode_results:
                m = mode_results[model_name]
                print(f"{model_name:<20} {m['train_acc']:>10.3f} {m['test_acc']:>10.3f} "
                      f"{m['overfitting_gap']:>8.3f} {m['train_f1']:>10.3f} {m['test_f1']:>10.3f}")

        # Find best model for this mode
        best_model = max(mode_results.items(), key=lambda x: x[1]['test_acc'])
        print(f"\n→ Best model: {best_model[0].upper()} "
              f"(Test Acc: {best_model[1]['test_acc']:.3f}, "
              f"Gap: {best_model[1]['overfitting_gap']:.3f})")

    except Exception as e:
        print(f"\n✗ {mode_desc} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        results_summary[mode] = None

# Final comparison across all modes
print("\n" + "=" * 120)
print("CROSS-MODE COMPARISON")
print("=" * 120)

for model_name in ALL_MODELS:
    print(f"\n{model_name.upper()}:")
    print(f"{'Mode':<30} {'Test Acc':>10} {'Overfitting Gap':>18}")
    print("-" * 60)

    for mode, mode_desc in MODES:
        if results_summary.get(mode) and model_name in results_summary[mode]:
            m = results_summary[mode][model_name]
            print(f"{mode_desc:<30} {m['test_acc']:>10.3f} {m['overfitting_gap']:>18.3f}")

# Overall best performers
print("\n" + "=" * 120)
print("BEST PERFORMERS BY CLASSIFICATION MODE")
print("=" * 120)

for mode, mode_desc in MODES:
    if results_summary.get(mode):
        mode_results = results_summary[mode]

        # Best by test accuracy
        best_acc = max(mode_results.items(), key=lambda x: x[1]['test_acc'])

        # Best generalization (lowest overfitting gap)
        best_gen = min(mode_results.items(), key=lambda x: x[1]['overfitting_gap'])

        print(f"\n{mode_desc}:")
        print(f"  Best Accuracy:       {best_acc[0]:<20} {best_acc[1]['test_acc']:.3f} (gap: {best_acc[1]['overfitting_gap']:+.3f})")
        print(f"  Best Generalization: {best_gen[0]:<20} {best_gen[1]['test_acc']:.3f} (gap: {best_gen[1]['overfitting_gap']:+.3f})")

# Save results to JSON
results_file = Path("classification_modes_comparison.json")
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\n✓ Results saved to: {results_file}")

print("\n" + "=" * 120)
print("KEY INSIGHTS")
print("=" * 120)
print("""
1. BINARY MODE (0 vs 1-5):
   - This is the default mode, treating all reactions (1-5) as positive class
   - Easiest classification task (2 classes)
   - Should show highest accuracies overall

2. MULTICLASS MODE (0-5):
   - Most challenging: predicting exact reaction strength (6 classes)
   - Lower accuracy expected due to fine-grained classification
   - More prone to overfitting with limited data per class
   - Use threshold_slider.py to find optimal post-hoc thresholds

3. THRESHOLD-2 MODE (0-2 vs 3-5):
   - Treats weak reactions (1-2) as non-reactive
   - Only strong reactions (3-5) counted as positive
   - Should show highest accuracy if weak reactions are ambiguous
   - Cleaner separation may reduce overfitting

RECOMMENDATIONS:
- If you care about detecting ANY reaction: use binary mode
- If you need exact reaction strength: use multiclass + threshold slider
- If you only care about strong reactions: use threshold-2 mode
""")

print("=" * 120)
print("DONE")
print("=" * 120)
