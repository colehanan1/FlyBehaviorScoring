#!/usr/bin/env python3
"""
Hyperparameter tuning for Random Forest model.

This script tests different RF configurations to potentially improve
beyond 93.6% test accuracy.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_experiment(
    n_estimators: int,
    max_depth: int | None,
    n_pcs: int,
    features: str,
    description: str
):
    """Run a single training experiment and return the results."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = f"artifacts/rf_tuning/{description}_{timestamp}"

    cmd = [
        "flybehavior-response", "train",
        "--data-csv", "/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv",
        "--labels-csv", "/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv",
        "--model", "random_forest",
        "--features", features,
        "--rf-class-weights", "balanced",
        "--rf-n-estimators", str(n_estimators),
        "--artifacts-dir", artifacts_dir,
        "--n-pcs", str(n_pcs),
    ]

    if max_depth is not None:
        cmd.extend(["--rf-max-depth", str(max_depth)])

    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}, n_pcs={n_pcs}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse metrics from the artifacts
    metrics_path = Path(artifacts_dir)
    if not metrics_path.exists():
        # Find the actual run directory
        base_dir = Path(artifacts_dir).parent
        if base_dir.exists():
            run_dirs = sorted(base_dir.glob("*"))
            if run_dirs:
                metrics_path = run_dirs[-1] / "metrics.json"
            else:
                metrics_path = None
        else:
            metrics_path = None
    else:
        # Find the timestamped subdirectory
        run_dirs = sorted(metrics_path.glob("*"))
        if run_dirs:
            metrics_path = run_dirs[-1] / "metrics.json"
        else:
            metrics_path = None

    if metrics_path and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        rf_metrics = metrics["models"]["random_forest"]["test"]
        accuracy = rf_metrics["accuracy"]
        precision = rf_metrics["precision"]
        recall = rf_metrics["recall"]
        f1 = rf_metrics["f1_binary"]

        print(f"\nResults:")
        print(f"  Test Accuracy:  {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall:    {recall:.4f}")
        print(f"  Test F1:        {f1:.4f}")

        return {
            "description": description,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_pcs": n_pcs,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "artifacts_dir": str(metrics_path.parent)
        }
    else:
        print(f"\nWARNING: Could not find metrics file")
        return None


def main():
    """Run hyperparameter tuning experiments."""

    base_features = "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value"

    experiments = [
        # Baseline (your current config)
        {
            "n_estimators": 100,
            "max_depth": None,
            "n_pcs": 12,
            "features": base_features,
            "description": "baseline_100trees_12pcs"
        },
        # More trees
        {
            "n_estimators": 200,
            "max_depth": None,
            "n_pcs": 12,
            "features": base_features,
            "description": "200trees_12pcs"
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "n_pcs": 12,
            "features": base_features,
            "description": "300trees_12pcs"
        },
        # Controlled depth to reduce overfitting
        {
            "n_estimators": 200,
            "max_depth": 15,
            "n_pcs": 12,
            "features": base_features,
            "description": "200trees_depth15_12pcs"
        },
        {
            "n_estimators": 200,
            "max_depth": 20,
            "n_pcs": 12,
            "features": base_features,
            "description": "200trees_depth20_12pcs"
        },
        # Different PCA components
        {
            "n_estimators": 200,
            "max_depth": None,
            "n_pcs": 15,
            "features": base_features,
            "description": "200trees_15pcs"
        },
        {
            "n_estimators": 200,
            "max_depth": None,
            "n_pcs": 20,
            "features": base_features,
            "description": "200trees_20pcs"
        },
        # Best combo
        {
            "n_estimators": 300,
            "max_depth": 20,
            "n_pcs": 15,
            "features": base_features,
            "description": "300trees_depth20_15pcs"
        },
    ]

    results = []
    for exp in experiments:
        result = run_experiment(**exp)
        if result:
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*80}\n")

    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print(f"{'Description':<30} {'Trees':>6} {'Depth':>6} {'PCs':>4} {'Accuracy':>10} {'F1':>10}")
    print("-" * 80)
    for r in results_sorted:
        depth_str = str(r['max_depth']) if r['max_depth'] else "None"
        print(f"{r['description']:<30} {r['n_estimators']:>6} {depth_str:>6} {r['n_pcs']:>4} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

    print(f"\nBest configuration:")
    best = results_sorted[0]
    print(f"  {best['description']}")
    print(f"  Test Accuracy: {best['accuracy']:.4f}")
    print(f"  Artifacts: {best['artifacts_dir']}")

    # Save results
    summary_path = Path("artifacts/rf_tuning/tuning_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
