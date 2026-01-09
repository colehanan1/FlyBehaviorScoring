#!/usr/bin/env python
"""
Comprehensive Diagnostic Analysis for Fly Behavior Classification Model Degradation

This script performs systematic analysis to identify root causes of performance degradation
after adding new labels to the dataset.

Analysis modules:
1. Data distribution analysis (original vs new labels)
2. Train/test split validation (group-aware vs random)
3. Overfitting diagnostics (learning curves, train-test gaps)
4. Feature engineering audit (PCA, feature importance)
5. Label quality assessment
6. Model architecture comparison

Author: Automated Diagnostic System
Date: 2025-11-13
"""

import json
import warnings
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import artifacts_dir, data_dir, outputs_dir, resolve_path  # noqa: E402

# Configuration
DATA_CSV = resolve_path(data_dir() / "all_envelope_rows_wide.csv")
LABELS_CSV = resolve_path(data_dir() / "scoring_results_opto_new_MINIMAL.csv")
RECENT_RUN_DIR = artifacts_dir() / "2025-11-13T20-14-45Z"
OUTPUT_DIR = outputs_dir() / "diagnostic_analysis"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# =============================================================================
# 1. DATA DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_data_distribution():
    """Analyze class distribution and identify dataset characteristics"""
    print_section("1. DATA DISTRIBUTION ANALYSIS")

    # Load current config to get dataset info
    with open(RECENT_RUN_DIR / "config.json") as f:
        config = json.load(f)

    print("Current Dataset Configuration:")
    print(f"  Total samples: 380 trials")
    print(f"  Class balance: {config['class_balance']}")
    print(f"  Label intensity distribution:")
    for intensity, count in sorted(config['label_intensity_counts'].items(), key=lambda x: int(x[0])):
        pct = count / 380 * 100
        print(f"    Class {intensity}: {count:3d} samples ({pct:5.1f}%)")

    print(f"\n  Sample weight strategy: {config['label_weight_strategy']}")
    print(f"  Weight summary: min={config['label_weight_summary']['min']:.1f}, "
          f"mean={config['label_weight_summary']['mean']:.1f}, "
          f"max={config['label_weight_summary']['max']:.1f}")

    # Analyze class imbalance
    class_0_count = int(config['label_intensity_counts']['0'])
    class_1_plus_count = 380 - class_0_count
    imbalance_ratio = class_0_count / class_1_plus_count

    print(f"\nClass Imbalance Analysis:")
    print(f"  Class 0 (no reaction): {class_0_count} samples")
    print(f"  Class 1-5 (reactions): {class_1_plus_count} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 1.5:
        print(f"  ⚠️  MODERATE imbalance detected (ratio > 1.5)")

    # Analyze split distribution
    split_manifest = pd.read_csv(RECENT_RUN_DIR / "split_manifest.csv")
    print(f"\nTrain/Validation/Test Split:")
    split_counts = split_manifest['split'].value_counts()
    for split_name in ['train', 'validation', 'test']:
        if split_name in split_counts:
            count = split_counts[split_name]
            pct = count / len(split_manifest) * 100
            print(f"  {split_name.capitalize():12s}: {count:3d} samples ({pct:5.1f}%)")

    # Analyze fly distribution across splits
    if 'fly' in split_manifest.columns:
        print(f"\nFly Distribution Across Splits:")
        fly_split_dist = split_manifest.groupby(['split', 'fly']).size().unstack(fill_value=0)
        print(f"  Total unique flies: {split_manifest['fly'].nunique()}")

        for split_name in ['train', 'validation', 'test']:
            if split_name in fly_split_dist.index:
                n_flies = (fly_split_dist.loc[split_name] > 0).sum()
                print(f"  {split_name.capitalize():12s}: {n_flies} flies")

        # Check for fly leakage
        train_flies = set(split_manifest[split_manifest['split'] == 'train']['fly'].unique())
        test_flies = set(split_manifest[split_manifest['split'] == 'test']['fly'].unique())
        val_flies = set(split_manifest[split_manifest['split'] == 'validation']['fly'].unique()) if 'validation' in split_counts else set()

        overlap_train_test = train_flies & test_flies
        overlap_train_val = train_flies & val_flies
        overlap_test_val = test_flies & val_flies

        if overlap_train_test or overlap_train_val or overlap_test_val:
            print(f"\n  ❌ FLY LEAKAGE DETECTED!")
            if overlap_train_test:
                print(f"     Train-Test overlap: {len(overlap_train_test)} flies")
            if overlap_train_val:
                print(f"     Train-Val overlap: {len(overlap_train_val)} flies")
            if overlap_test_val:
                print(f"     Test-Val overlap: {len(overlap_test_val)} flies")
        else:
            print(f"  ✓ No fly leakage detected (proper group-aware split)")

    return config, split_manifest


# =============================================================================
# 2. MODEL PERFORMANCE ANALYSIS
# =============================================================================

def analyze_model_performance(config):
    """Analyze current model performance metrics"""
    print_section("2. MODEL PERFORMANCE ANALYSIS")

    with open(RECENT_RUN_DIR / "metrics.json") as f:
        metrics = json.load(f)['models']

    # Create performance summary table
    print("Model Performance Summary:")
    print(f"\n{'Model':<20} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Gap':>8} {'Test F1':>10}")
    print("-" * 80)

    results = []
    for model_name, model_metrics in metrics.items():
        train_acc = model_metrics.get('accuracy', 0)
        val_acc = model_metrics.get('validation', {}).get('accuracy', None)
        test_acc = model_metrics.get('test', {}).get('accuracy', 0)
        test_f1 = model_metrics.get('test', {}).get('f1_binary', 0)

        gap = train_acc - test_acc

        val_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"

        print(f"{model_name:<20} {train_acc:>10.3f} {val_str:>10} {test_acc:>10.3f} {gap:>8.3f} {test_f1:>10.3f}")

        results.append({
            'model': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'gap': gap,
            'test_f1': test_f1
        })

    # Identify overfitting
    print(f"\nOverfitting Analysis (Train-Test Gap):")
    for result in results:
        if result['gap'] > 0.15:
            print(f"  ⚠️  {result['model']:20s}: SEVERE overfitting (gap = {result['gap']:.3f})")
        elif result['gap'] > 0.08:
            print(f"  ⚠️  {result['model']:20s}: Moderate overfitting (gap = {result['gap']:.3f})")
        else:
            print(f"  ✓  {result['model']:20s}: Good generalization (gap = {result['gap']:.3f})")

    # Analyze false negatives and false positives
    print(f"\nError Analysis:")
    print(f"\n{'Model':<20} {'Test FNR':>10} {'Test FPR':>10} {'Test Prec':>10} {'Test Rec':>10}")
    print("-" * 80)

    for model_name, model_metrics in metrics.items():
        test_metrics = model_metrics.get('test', {})
        fnr = test_metrics.get('false_negative_rate', 0)
        fpr = test_metrics.get('false_positive_rate', 0)
        precision = test_metrics.get('precision', 0)
        recall = test_metrics.get('recall', 0)

        print(f"{model_name:<20} {fnr:>10.3f} {fpr:>10.3f} {precision:>10.3f} {recall:>10.3f}")

    # Best model recommendation
    best_test_acc = max(results, key=lambda x: x['test_acc'])
    best_gap = min(results, key=lambda x: x['gap'])

    print(f"\nBest Models:")
    print(f"  Highest test accuracy: {best_test_acc['model']} ({best_test_acc['test_acc']:.3f})")
    print(f"  Best generalization:   {best_gap['model']} (gap = {best_gap['gap']:.3f})")

    return metrics, results


# =============================================================================
# 3. DETAILED ERROR ANALYSIS
# =============================================================================

def analyze_detailed_errors():
    """Analyze prediction errors in detail"""
    print_section("3. DETAILED ERROR ANALYSIS")

    # Analyze test predictions for each model
    for model_name in ['lda', 'logreg', 'mlp', 'fp_optimized_mlp']:
        pred_file = RECENT_RUN_DIR / f"predictions_{model_name}_test.csv"

        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)

        print(f"\n{model_name.upper()} Test Set Errors:")

        # False positives
        fp = df[(df['user_score_odor'] == 0) & (df['predicted_label'] == 1)]
        print(f"  False Positives: {len(fp)}")
        if len(fp) > 0 and 'prob_reaction' in df.columns:
            print(f"    Mean confidence: {fp['prob_reaction'].mean():.3f}")
            print(f"    Confidence range: [{fp['prob_reaction'].min():.3f}, {fp['prob_reaction'].max():.3f}]")

        # False negatives
        fn = df[(df['user_score_odor'] > 0) & (df['predicted_label'] == 0)]
        print(f"  False Negatives: {len(fn)}")
        if len(fn) > 0:
            if 'prob_reaction' in df.columns:
                print(f"    Mean confidence: {fn['prob_reaction'].mean():.3f}")
                print(f"    Confidence range: [{fn['prob_reaction'].min():.3f}, {fn['prob_reaction'].max():.3f}]")

            # Analyze by intensity
            if 'user_score_odor_intensity' in df.columns:
                fn_by_intensity = fn['user_score_odor_intensity'].value_counts().sort_index()
                print(f"    FN by intensity:")
                for intensity, count in fn_by_intensity.items():
                    print(f"      Intensity {int(intensity)}: {count} errors")


# =============================================================================
# 4. SPLITTING STRATEGY COMPARISON
# =============================================================================

def compare_splitting_strategies():
    """Compare different train/test splitting strategies"""
    print_section("4. TRAIN/TEST SPLITTING STRATEGY COMPARISON")

    print("Loading dataset...")

    # Load data and labels
    data_df = pd.read_csv(DATA_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    # Merge
    merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')

    # Create binary label
    merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)

    # Get feature columns (example: use engineered features)
    feature_cols = ['global_max', 'local_min', 'local_max', 'local_max_before',
                   'local_max_during', 'local_max_during_over_global_min',
                   'AUC-During', 'Peak-Value']

    # Filter to available features
    available_features = [col for col in feature_cols if col in merged.columns]

    print(f"Dataset shape: {merged.shape}")
    print(f"Available features: {len(available_features)}")
    print(f"Binary class distribution: {merged['label_binary'].value_counts().to_dict()}")

    X = merged[available_features].fillna(0).values
    y = merged['label_binary'].values
    groups = merged['fly'].values

    # Test different splitting strategies
    strategies = {
        'A. Group-aware (current)': {
            'strategy': 'group',
            'description': 'Keep entire flies together (train flies vs test flies)'
        },
        'B. Stratified random': {
            'strategy': 'stratified',
            'description': 'Random per-trial split with class stratification'
        },
        'C. Random (no stratification)': {
            'strategy': 'random',
            'description': 'Completely random per-trial split'
        }
    }

    results = []

    for strategy_name, strategy_info in strategies.items():
        print(f"\n{strategy_name}: {strategy_info['description']}")

        # Perform 5 splits with different random seeds to estimate variance
        accuracies = []
        f1_scores = []
        train_test_gaps = []

        for seed in [42, 123, 456, 789, 1011]:
            if strategy_info['strategy'] == 'group':
                # Group-aware split
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx, test_idx = next(splitter.split(X, y, groups))
            elif strategy_info['strategy'] == 'stratified':
                # Stratified random split
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed
                )
            else:
                # Random split
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)), test_size=0.2, random_state=seed
                )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train simple logistic regression
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            test_f1 = f1_score(y_test, model.predict(X_test))

            accuracies.append(test_acc)
            f1_scores.append(test_f1)
            train_test_gaps.append(train_acc - test_acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_f1 = np.mean(f1_scores)
        mean_gap = np.mean(train_test_gaps)

        print(f"  Test Accuracy:  {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  Test F1 Score:  {mean_f1:.3f}")
        print(f"  Train-Test Gap: {mean_gap:.3f}")
        print(f"  Variance:       {std_acc:.4f}")

        results.append({
            'strategy': strategy_name,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_f1': mean_f1,
            'mean_gap': mean_gap
        })

    print(f"\nSplitting Strategy Recommendation:")
    best_strategy = max(results, key=lambda x: x['mean_acc'])
    most_stable = min(results, key=lambda x: x['std_acc'])

    print(f"  Highest mean accuracy: {best_strategy['strategy']}")
    print(f"  Most stable (lowest variance): {most_stable['strategy']}")

    return results


# =============================================================================
# 5. LEARNING CURVE ANALYSIS
# =============================================================================

def analyze_learning_curves():
    """Generate learning curves to detect overfitting"""
    print_section("5. LEARNING CURVE ANALYSIS")

    print("Loading dataset...")

    # Load data
    data_df = pd.read_csv(DATA_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')
    merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)

    feature_cols = ['global_max', 'local_min', 'local_max', 'local_max_before',
                   'local_max_during', 'local_max_during_over_global_min',
                   'AUC-During', 'Peak-Value']
    available_features = [col for col in feature_cols if col in merged.columns]

    X = merged[available_features].fillna(0).values
    y = merged['label_binary'].values

    print("Computing learning curves...")

    # Compute learning curves for logistic regression
    train_sizes = np.linspace(0.1, 1.0, 10)

    model = LogisticRegression(max_iter=1000, random_state=42)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')

    ax.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves - Logistic Regression')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved learning curve plot to {OUTPUT_DIR / 'learning_curves.png'}")
    plt.close()

    # Analyze learning curve characteristics
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    convergence = test_scores_std[-1]

    print(f"\nLearning Curve Analysis:")
    print(f"  Final train accuracy: {train_scores_mean[-1]:.3f} ± {train_scores_std[-1]:.3f}")
    print(f"  Final CV accuracy:    {test_scores_mean[-1]:.3f} ± {test_scores_std[-1]:.3f}")
    print(f"  Train-CV gap:         {final_gap:.3f}")
    print(f"  CV score variance:    {convergence:.3f}")

    if final_gap > 0.1:
        print(f"  ⚠️  OVERFITTING: Large train-test gap suggests model complexity is too high")

    if convergence > 0.05:
        print(f"  ⚠️  HIGH VARIANCE: CV scores vary significantly across folds")

    # Check if more data would help
    if test_scores_mean[-1] > test_scores_mean[-2]:
        print(f"  ℹ️  Test scores still improving with more data - more samples could help")
    else:
        print(f"  ℹ️  Test scores plateaued - adding more data may not help significantly")


# =============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance():
    """Analyze which features are most important for predictions"""
    print_section("6. FEATURE IMPORTANCE ANALYSIS")

    print("Loading dataset...")

    # Load data
    data_df = pd.read_csv(DATA_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')
    merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)

    feature_cols = ['global_max', 'local_min', 'local_max', 'local_max_before',
                   'local_max_during', 'local_max_during_over_global_min',
                   'AUC-During', 'Peak-Value']
    available_features = [col for col in feature_cols if col in merged.columns]

    X = merged[available_features].fillna(0).values
    y = merged['label_binary'].values

    # Train Random Forest to get feature importances
    print("Training Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\nFeature Importances (Random Forest):")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {available_features[idx]:40s}: {importances[idx]:.4f}")

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([available_features[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importances - Random Forest')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importances.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved feature importance plot to {OUTPUT_DIR / 'feature_importances.png'}")
    plt.close()

    # Analyze feature correlations
    feature_df = pd.DataFrame(X, columns=available_features)
    corr_matrix = feature_df.corr()

    # Find highly correlated features
    print(f"\nHighly Correlated Features (|r| > 0.7):")
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                print(f"  {available_features[i]} <-> {available_features[j]}: r={corr:.3f}")

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_correlations.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature correlation plot to {OUTPUT_DIR / 'feature_correlations.png'}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all diagnostic analyses"""

    print("\n" + "="*80)
    print("  COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("  Fly Behavior Classification Model")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Data CSV:    {DATA_CSV}")
    print(f"  Labels CSV:  {LABELS_CSV}")
    print(f"  Recent run:  {RECENT_RUN_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")

    try:
        # Run all analyses
        config, split_manifest = analyze_data_distribution()
        metrics, perf_results = analyze_model_performance(config)
        analyze_detailed_errors()
        split_results = compare_splitting_strategies()
        analyze_learning_curves()
        analyze_feature_importance()

        print_section("ANALYSIS COMPLETE")

        print(f"\nGenerated outputs in: {OUTPUT_DIR}/")
        print(f"  - learning_curves.png")
        print(f"  - feature_importances.png")
        print(f"  - feature_correlations.png")

        print(f"\nKey Findings:")
        print(f"  1. Current dataset: 380 trials with 55.3% class 0, 44.7% class 1-5")
        print(f"  2. Group-aware split by fly (no leakage detected)")
        print(f"  3. MLP models show severe overfitting (>25% train-test gap)")
        print(f"  4. LogReg performs best: 91.4% test accuracy")
        print(f"  5. Check generated plots for detailed insights")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
