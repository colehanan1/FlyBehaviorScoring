#!/usr/bin/env python3
"""Analyze and compare results across all 4 classification modes."""

import json
from pathlib import Path
import pandas as pd

# Find latest metrics for each mode
modes = {
    "binary": "Binary (0 vs 1-5)",
    "threshold-1": "Threshold-1 (0-1 vs 2-5)",
    "threshold-2": "Threshold-2 (0-2 vs 3-5)",
    "multiclass": "Multiclass (0-5)"
}

results = {}

for mode_key, mode_name in modes.items():
    metrics_files = sorted(Path(f"comprehensive_test_all/{mode_key}").glob("*/metrics.json"))
    if metrics_files:
        latest = metrics_files[-1]
        with open(latest) as f:
            data = json.load(f)
        results[mode_key] = {
            "name": mode_name,
            "file": str(latest),
            "models": data["models"]
        }

# Create comparison table
print("=" * 140)
print("COMPREHENSIVE COMPARISON: ALL 4 CLASSIFICATION MODES")
print("=" * 140)
print()

# Compare test accuracy across all modes
print("TEST ACCURACY COMPARISON")
print("-" * 140)
print(f"{'Model':<25} {'Binary':>15} {'Threshold-1':>15} {'Threshold-2':>15} {'Multiclass':>15} {'Best Mode':<20}")
print("-" * 140)

model_names = ["lda", "logreg", "random_forest", "hist_gb", "mlp", "fp_optimized_mlp", "xgb"]

for model in model_names:
    accuracies = {}
    for mode_key in modes.keys():
        if model in results[mode_key]["models"]:
            test_acc = results[mode_key]["models"][model]["test"]["accuracy"]
            accuracies[mode_key] = test_acc
        else:
            accuracies[mode_key] = 0.0

    best_mode = max(accuracies.items(), key=lambda x: x[1])[0]
    best_name = modes[best_mode].split("(")[0].strip()

    print(f"{model:<25} {accuracies.get('binary', 0):.3f}           "
          f"{accuracies.get('threshold-1', 0):.3f}           "
          f"{accuracies.get('threshold-2', 0):.3f}           "
          f"{accuracies.get('multiclass', 0):.3f}           "
          f"{best_name:<20}")

print()
print("=" * 140)
print()

# Compare F1 scores
print("TEST F1 SCORE COMPARISON")
print("-" * 140)
print(f"{'Model':<25} {'Binary':>15} {'Threshold-1':>15} {'Threshold-2':>15} {'Multiclass':>15} {'Best Mode':<20}")
print("-" * 140)

for model in model_names:
    f1_scores = {}
    for mode_key in modes.keys():
        if model in results[mode_key]["models"]:
            test_metrics = results[mode_key]["models"][model]["test"]
            # Use binary F1 if available, otherwise macro
            if test_metrics.get("f1_binary") is not None:
                f1 = test_metrics["f1_binary"]
            else:
                f1 = test_metrics.get("f1_macro", 0.0)
            f1_scores[mode_key] = f1
        else:
            f1_scores[mode_key] = 0.0

    best_mode = max(f1_scores.items(), key=lambda x: x[1])[0]
    best_name = modes[best_mode].split("(")[0].strip()

    print(f"{model:<25} {f1_scores.get('binary', 0):.3f}           "
          f"{f1_scores.get('threshold-1', 0):.3f}           "
          f"{f1_scores.get('threshold-2', 0):.3f}           "
          f"{f1_scores.get('multiclass', 0):.3f}           "
          f"{best_name:<20}")

print()
print("=" * 140)
print()

# Best model per mode
print("BEST MODEL FOR EACH MODE (by Test Accuracy)")
print("-" * 140)

for mode_key, mode_data in results.items():
    best_model = max(
        mode_data["models"].items(),
        key=lambda x: x[1]["test"]["accuracy"]
    )
    print(f"{mode_data['name']:<30}: {best_model[0]:<20} "
          f"(Test Acc: {best_model[1]['test']['accuracy']:.3f}, "
          f"CV Acc: {best_model[1]['cross_validation']['accuracy']:.3f})")

print()
print("=" * 140)
print()

# Threshold-1 detailed analysis
print("THRESHOLD-1 MODE DETAILED ANALYSIS (0-1 vs 2-5)")
print("-" * 140)
print()
print("This mode classifies:")
print("  - Class 0: Labels 0-1 (No reaction or minimal reaction)")
print("  - Class 1: Labels 2-5 (Moderate to strong reaction)")
print()
print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'CV Acc':>10} {'Test F1':>10} {'Overfit Gap':>12}")
print("-" * 100)

for model in model_names:
    if model in results["threshold-1"]["models"]:
        model_data = results["threshold-1"]["models"][model]
        train_acc = model_data["accuracy"]
        test_acc = model_data["test"]["accuracy"]
        cv_acc = model_data["cross_validation"]["accuracy"]
        test_f1 = model_data["test"].get("f1_binary") or model_data["test"]["f1_macro"]
        gap = train_acc - test_acc

        print(f"{model:<25} {train_acc:>10.3f} {test_acc:>10.3f} {cv_acc:>10.3f} "
              f"{test_f1:>10.3f} {gap:>12.3f}")

print()
print("=" * 140)
print()

# Summary insights
print("KEY INSIGHTS:")
print("-" * 140)
print()

# Calculate average accuracies per mode
avg_accuracies = {}
for mode_key, mode_data in results.items():
    accs = [m["test"]["accuracy"] for m in mode_data["models"].values()]
    avg_accuracies[mode_key] = sum(accs) / len(accs)

best_avg_mode = max(avg_accuracies.items(), key=lambda x: x[1])
print(f"1. Best Overall Mode (Average Test Accuracy): {modes[best_avg_mode[0]]} ({best_avg_mode[1]:.3f})")

# Threshold-1 specific insights
t1_data = results["threshold-1"]["models"]
t1_accs = [m["test"]["accuracy"] for m in t1_data.values()]
t1_avg = sum(t1_accs) / len(t1_accs)
print(f"2. Threshold-1 Average Test Accuracy: {t1_avg:.3f}")

# Best threshold-1 model
best_t1 = max(t1_data.items(), key=lambda x: x[1]["test"]["accuracy"])
print(f"3. Best Model for Threshold-1: {best_t1[0]} ({best_t1[1]['test']['accuracy']:.3f})")

# Compare threshold modes
print()
print("4. Threshold Mode Comparison:")
for mode in ["threshold-1", "threshold-2"]:
    avg = avg_accuracies[mode]
    print(f"   - {modes[mode]}: {avg:.3f} average test accuracy")

print()
print("=" * 140)
print()

# Save detailed comparison to CSV
comparison_data = []
for model in model_names:
    row = {"Model": model}
    for mode_key in modes.keys():
        if model in results[mode_key]["models"]:
            row[f"{mode_key}_test_acc"] = results[mode_key]["models"][model]["test"]["accuracy"]
            row[f"{mode_key}_cv_acc"] = results[mode_key]["models"][model]["cross_validation"]["accuracy"]
        else:
            row[f"{mode_key}_test_acc"] = 0.0
            row[f"{mode_key}_cv_acc"] = 0.0
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
output_csv = "comprehensive_mode_comparison.csv"
df.to_csv(output_csv, index=False)
print(f"Detailed comparison saved to: {output_csv}")
print()
print("=" * 140)
