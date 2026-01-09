#!/usr/bin/env python3
"""Comprehensive test of all models across ALL classification modes including threshold-1.

This test includes 4 classification modes:
- binary (0 vs 1-5)
- multiclass (0-5)
- threshold-1 (0-1 vs 2-5) - NEW!
- threshold-2 (0-2 vs 3-5)
"""

import json
from pathlib import Path

# Run all models with all FOUR classification modes
data_csv = "/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv"
labels_csv = "/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv"

classification_modes = {
    'binary': 'Binary (0 vs 1-5)',
    'threshold-1': 'Threshold-1 (0-1 vs 2-5)',  # NEW MODE
    'threshold-2': 'Threshold-2 (0-2 vs 3-5)',
    'multiclass': 'Multi-class (0-5)',
}

models = ["lda", "logreg", "random_forest", "hist_gb", "mlp", "fp_optimized_mlp", "xgb"]
features = ["AUC-During", "TimeToPeak-During", "Peak-Value"]

print("="*120)
print("COMPREHENSIVE CLASSIFICATION MODE TEST - ALL 4 MODES")
print("="*120)
print(f"\nConfiguration:")
print(f"  Data CSV: {data_csv}")
print(f"  Labels CSV: {labels_csv}")
print(f"  Features: {features}")
print(f"  Models: {models}")
print(f"  PCA Components: 10")
print(f"  Seed: 67")
print(f"  Cross-validation: 5 folds")
print(f"  Classification Modes: {list(classification_modes.keys())}")
print()

import subprocess
import os

results_summary = {}

for mode_key, mode_name in classification_modes.items():
    print("\n" + "="*120)
    print(f"TESTING: {mode_name}")
    print("="*120)

    artifacts_dir = f"comprehensive_test_all/{mode_key}"

    # Build command
    cmd = [
        "flybehavior-response", "train",
        "--data-csv", data_csv,
        "--labels-csv", labels_csv,
        "--model", "all",
        "--cv", "5",
        "--n-pcs", "10",
        "--seed", "67",
        "--classification-mode", mode_key,
        "--artifacts-dir", artifacts_dir
    ]

    # Add features
    for feat in features:
        cmd.extend(["--feature", feat])

    print(f"\nCommand: {' '.join(cmd)}\n")

    # Set environment variable for matplotlib
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    try:
        # Run training
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per mode
        )

        if result.returncode != 0:
            print(f"✗ {mode_name} FAILED")
            print(f"Error output:\n{result.stderr[-2000:]}")  # Last 2000 chars
            results_summary[mode_key] = {"status": "FAILED", "error": result.stderr[-1000:]}
            continue

        print(f"✓ {mode_name} completed successfully!")

        # Load and parse results
        latest_dir = sorted(Path(artifacts_dir).glob("20*"))[-1]
        metrics_file = latest_dir / "metrics.json"

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Extract model performance
        mode_results = {}
        for model_name in models:
            if model_name in metrics['models']:
                model_metrics = metrics['models'][model_name]

                test_acc = model_metrics['test']['accuracy']
                train_acc = model_metrics.get('accuracy', 0)

                # Get F1 score (binary or macro depending on mode)
                if model_metrics['test'].get('f1_binary') is not None:
                    test_f1 = model_metrics['test']['f1_binary']
                    train_f1 = model_metrics.get('f1_binary', 0)
                    f1_type = 'binary'
                else:
                    test_f1 = model_metrics['test']['f1_macro']
                    train_f1 = model_metrics.get('f1_macro', 0)
                    f1_type = 'macro'

                # Get CV scores if available
                cv_acc = model_metrics.get('cv', {}).get('accuracy')

                mode_results[model_name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'f1_type': f1_type,
                    'cv_accuracy': cv_acc,
                    'overfitting_gap': train_acc - test_acc
                }

        results_summary[mode_key] = {
            "status": "SUCCESS",
            "models": mode_results,
            "artifacts_dir": str(latest_dir)
        }

        print(f"\nResults for {mode_name}:")
        print(f"{'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'CV Acc':>10} {'Test F1':>10} {'Gap':>8}")
        print("-" * 90)
        for model_name, model_data in mode_results.items():
            cv_str = f"{model_data['cv_accuracy']:.3f}" if model_data['cv_accuracy'] else "N/A"
            print(f"{model_name:<20} {model_data['train_accuracy']:>10.3f} "
                  f"{model_data['test_accuracy']:>10.3f} {cv_str:>10} "
                  f"{model_data['test_f1']:>10.3f} {model_data['overfitting_gap']:>8.3f}")

    except subprocess.TimeoutExpired:
        print(f"✗ {mode_name} TIMEOUT (>10 minutes)")
        results_summary[mode_key] = {"status": "TIMEOUT"}
    except Exception as e:
        print(f"✗ {mode_name} ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results_summary[mode_key] = {"status": "ERROR", "error": str(e)}

# Final summary
print("\n" + "="*120)
print("FINAL SUMMARY - ALL 4 CLASSIFICATION MODES")
print("="*120)

for mode_key, mode_name in classification_modes.items():
    print(f"\n{mode_name}:")
    if results_summary[mode_key]["status"] == "SUCCESS":
        print(f"  ✓ Status: {results_summary[mode_key]['status']}")
        print(f"  Artifacts: {results_summary[mode_key]['artifacts_dir']}")

        # Find best model by test accuracy
        best_model = max(
            results_summary[mode_key]['models'].items(),
            key=lambda x: x[1]['test_accuracy']
        )
        print(f"  Best Model: {best_model[0]} (Test Acc: {best_model[1]['test_accuracy']:.3f})")
    else:
        print(f"  ✗ Status: {results_summary[mode_key]['status']}")

# Save results
output_file = Path("comprehensive_test_all_results.json")
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n\nDetailed results saved to: {output_file}")
print("\n" + "="*120)
print("TEST COMPLETE - ALL 4 MODES")
print("="*120)
