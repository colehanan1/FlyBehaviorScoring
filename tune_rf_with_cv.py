#!/usr/bin/env python3
"""
Cross-validation tuning for Random Forest to avoid overfitting.

This script tunes the most important RF hyperparameters:
1. max_features - # features to consider at each split (MOST IMPORTANT for overfitting)
2. max_depth - maximum tree depth
3. min_samples_split - minimum samples to split a node
4. n_estimators - number of trees

Uses 5-fold cross-validation on the TRAINING set to select best parameters,
then evaluates on the held-out test set.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

# You'll need to adapt this to load your data the same way train.py does
# For now, this is a template showing the CV approach

def load_preprocessed_data():
    """
    Load and preprocess data (same as train.py does).

    You'll need to:
    1. Load data and labels CSVs
    2. Merge them
    3. Apply PCA to traces
    4. Split into train/test with hybrid splitting

    Returns:
    --------
    X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test, groups_train
    """
    # TODO: Implement this based on your train.py logic
    # For now, placeholder
    raise NotImplementedError("Implement data loading from train.py")


def tune_random_forest_cv():
    """
    Tune Random Forest using cross-validation to avoid overfitting.
    """

    # Load data
    print("Loading and preprocessing data...")
    # X_train, X_test, y_train, y_test, sw_train, sw_test, groups_train = load_preprocessed_data()

    # For demonstration, let's assume you have 20 features (12 PCs + 8 engineered)
    n_features = 20

    print(f"\nTotal features: {n_features}")
    print(f"  - 12 PCA components from traces")
    print(f"  - 8 engineered features")

    # Define hyperparameter grid
    # Focus on max_features (most important for overfitting)
    param_grid = {
        # max_features: how many features to consider at each split
        'max_features': [
            'sqrt',           # sqrt(20) ≈ 4.5 features (default, good starting point)
            'log2',           # log2(20) ≈ 4.3 features
            5,                # Try specific values
            8,
            10,
            15,
            None,             # Use all 20 features (may overfit)
        ],

        # max_depth: limit tree depth to prevent overfitting
        'max_depth': [
            10,
            15,
            20,
            None,             # Unlimited (may overfit)
        ],

        # min_samples_split: minimum samples to split a node
        'min_samples_split': [
            2,                # Default (may overfit)
            5,
            10,
        ],

        # n_estimators: more trees generally better (diminishing returns after ~200)
        'n_estimators': [
            100,
            200,
            300,
        ],
    }

    print(f"\nHyperparameter grid:")
    print(f"  max_features: {param_grid['max_features']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  min_samples_split: {param_grid['min_samples_split']}")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"\nTotal combinations: {len(param_grid['max_features']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['n_estimators'])}")

    # Create base model
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # For group-aware CV (important! Don't split flies across folds)
    # Use GroupKFold to ensure same fly doesn't appear in train and validation
    cv_splitter = GroupKFold(n_splits=5)

    print("\nRunning 5-fold cross-validation with group-aware splitting...")
    print("(This may take 10-30 minutes depending on grid size)")

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv_splitter,  # Use group-aware CV
        scoring='accuracy',  # Or use 'f1' if you care more about F1
        n_jobs=-1,           # Parallel processing
        verbose=2,
        return_train_score=True,  # Important: see train vs val performance
    )

    # NOTE: You need to pass groups parameter to fit()
    # grid_search.fit(X_train, y_train, groups=groups_train, sample_weight=sw_train)

    print("\nGrid search complete!")

    # Analyze results
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Sort by validation score
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by validation accuracy)")
    print("="*80)

    display_cols = [
        'param_max_features',
        'param_max_depth',
        'param_min_samples_split',
        'param_n_estimators',
        'mean_train_score',
        'mean_test_score',
        'std_test_score',
    ]

    print(results_df[display_cols].head(10).to_string(index=False))

    # Best parameters
    print("\n" + "="*80)
    print("BEST PARAMETERS (from cross-validation)")
    print("="*80)
    print(grid_search.best_params_)
    print(f"\nBest CV accuracy: {grid_search.best_score_:.4f}")

    # Analyze overfitting
    best_idx = results_df.index[0]
    train_score = results_df.loc[best_idx, 'mean_train_score']
    val_score = results_df.loc[best_idx, 'mean_test_score']
    gap = train_score - val_score

    print(f"\nOverfitting analysis for best model:")
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Val accuracy:   {val_score:.4f}")
    print(f"  Gap:            {gap:.4f} ({gap*100:.1f}%)")

    if gap > 0.10:
        print("  ⚠️  WARNING: Large train-val gap suggests overfitting")
    elif gap > 0.05:
        print("  ⚠️  MODERATE: Some overfitting detected")
    else:
        print("  ✅ GOOD: Minimal overfitting")

    # Evaluate on held-out test set
    # best_model = grid_search.best_estimator_
    # y_pred = best_model.predict(X_test)
    # test_acc = accuracy_score(y_test, y_pred, sample_weight=sw_test)
    # test_f1 = f1_score(y_test, y_pred, sample_weight=sw_test)

    # print(f"\n" + "="*80)
    # print("FINAL TEST SET PERFORMANCE")
    # print("="*80)
    # print(f"Test accuracy: {test_acc:.4f}")
    # print(f"Test F1:       {test_f1:.4f}")

    # print("\nConfusion Matrix (Test Set):")
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # print("\nClassification Report (Test Set):")
    # print(classification_report(y_test, y_pred, target_names=['No Reaction', 'Reaction']))

    # Save results
    # output_dir = Path("artifacts/rf_cv_tuning")
    # output_dir.mkdir(parents=True, exist_ok=True)

    # results_df.to_csv(output_dir / "cv_results.csv", index=False)
    # joblib.dump(best_model, output_dir / "best_rf_model.joblib")

    # print(f"\nResults saved to: {output_dir}")

    return grid_search


def quick_max_features_test():
    """
    Quick test of just max_features (the most important parameter).

    This is faster than full grid search.
    """

    print("="*80)
    print("QUICK max_features TUNING (most important for overfitting)")
    print("="*80)

    n_features = 20

    # Test different max_features values
    max_features_values = [
        'sqrt',   # √20 ≈ 4.5
        'log2',   # log2(20) ≈ 4.3
        5,
        8,
        10,
        15,
        None,     # All 20 features
    ]

    print(f"\nTesting max_features values: {max_features_values}")
    print("Keeping other params constant: n_estimators=200, max_depth=None")

    # TODO: Load data
    # X_train, X_test, y_train, y_test, sw_train, sw_test, groups_train = load_preprocessed_data()

    results = []

    for max_feat in max_features_values:
        print(f"\nTesting max_features={max_feat}...")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_features=max_feat,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

        # 5-fold group CV
        cv_splitter = GroupKFold(n_splits=5)

        # TODO: Implement CV scoring
        # from sklearn.model_selection import cross_validate
        # cv_results = cross_validate(
        #     rf, X_train, y_train,
        #     cv=cv_splitter,
        #     groups=groups_train,
        #     scoring=['accuracy', 'f1'],
        #     return_train_score=True,
        #     fit_params={'sample_weight': sw_train}
        # )

        # train_acc = cv_results['train_accuracy'].mean()
        # val_acc = cv_results['test_accuracy'].mean()
        # gap = train_acc - val_acc

        # results.append({
        #     'max_features': max_feat,
        #     'train_accuracy': train_acc,
        #     'val_accuracy': val_acc,
        #     'overfitting_gap': gap,
        # })

        # print(f"  Train acc: {train_acc:.4f}")
        # print(f"  Val acc:   {val_acc:.4f}")
        # print(f"  Gap:       {gap:.4f}")

    # results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)

    # print("\n" + "="*80)
    # print("RESULTS SUMMARY (sorted by validation accuracy)")
    # print("="*80)
    # print(results_df.to_string(index=False))

    # best = results_df.iloc[0]
    # print(f"\nBest max_features: {best['max_features']}")
    # print(f"  Validation accuracy: {best['val_accuracy']:.4f}")
    # print(f"  Overfitting gap: {best['overfitting_gap']:.4f}")


if __name__ == "__main__":
    import sys

    print(__doc__)
    print("\n" + "="*80)
    print("IMPORTANT: This is a TEMPLATE script")
    print("="*80)
    print("\nYou need to implement the data loading function based on train.py")
    print("The script shows the correct CV approach for avoiding overfitting.")
    print("\nKey points:")
    print("1. Use GroupKFold to avoid fly leakage between train/val folds")
    print("2. Tune max_features (most important for overfitting)")
    print("3. Check train-val gap to diagnose overfitting")
    print("4. Evaluate final model on held-out test set")
    print("\n" + "="*80)

    # Uncomment when data loading is implemented:
    # if len(sys.argv) > 1 and sys.argv[1] == "--quick":
    #     quick_max_features_test()
    # else:
    #     tune_random_forest_cv()
