#!/usr/bin/env python
"""
Compare Group-Aware vs Stratified Random Splitting Strategies

Tests which approach gives better performance and generalization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load data
DATA_CSV = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv")
LABELS_CSV = Path("/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv")

print("="*80)
print("  SPLITTING STRATEGY COMPARISON")
print("="*80)

# Load and merge
data_df = pd.read_csv(DATA_CSV)
labels_df = pd.read_csv(LABELS_CSV)
merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')

print(f"\nDataset: {len(merged)} trials, {merged['fly'].nunique()} flies")

# Create binary label
merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)
print(f"Class balance: {merged['label_binary'].mean():.1%} responders")

# Get features
feature_cols = ['local_min', 'local_max', 'local_max_before', 'local_max_during',
                'local_max_during_over_global_min', 'AUC-During', 'Peak-Value']
available_features = [col for col in feature_cols if col in merged.columns]

print(f"Features: {len(available_features)} available")

X = merged[available_features].fillna(0).values
y = merged['label_binary'].values
groups = merged['fly'].values

# =============================================================================
# STRATEGY 1: Current Group-Aware (Fly-Level) Splitting
# =============================================================================

print("\n" + "="*80)
print("STRATEGY 1: Group-Aware (Keep Flies Together)")
print("="*80)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train_g, X_test_g = X[train_idx], X[test_idx]
y_train_g, y_test_g = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# Check class balance in splits
train_resp_rate = y_train_g.mean()
test_resp_rate = y_test_g.mean()

print(f"\nSplit sizes:")
print(f"  Train: {len(y_train_g)} trials ({len(np.unique(groups_train))} flies)")
print(f"  Test:  {len(y_test_g)} trials ({len(np.unique(groups_test))} flies)")

print(f"\nClass distribution:")
print(f"  Train responder rate: {train_resp_rate:.1%}")
print(f"  Test responder rate:  {test_resp_rate:.1%}")
print(f"  Distribution gap:     {abs(train_resp_rate - test_resp_rate):.1%}")

if abs(train_resp_rate - test_resp_rate) > 0.05:
    print(f"  ⚠️  WARNING: >5% gap in class distribution!")

# Train model
model_group = LogisticRegression(max_iter=1000, random_state=42)
model_group.fit(X_train_g, y_train_g)

train_acc_g = model_group.score(X_train_g, y_train_g)
test_acc_g = model_group.score(X_test_g, y_test_g)
test_f1_g = f1_score(y_test_g, model_group.predict(X_test_g))

print(f"\nPerformance:")
print(f"  Train Accuracy: {train_acc_g:.1%}")
print(f"  Test Accuracy:  {test_acc_g:.1%}")
print(f"  Test F1 Score:  {test_f1_g:.1%}")
print(f"  Overfitting Gap: {(train_acc_g - test_acc_g):.1%}")

# =============================================================================
# STRATEGY 2: Stratified Random (Trial-Level) Splitting
# =============================================================================

print("\n" + "="*80)
print("STRATEGY 2: Stratified Random (Per-Trial, Balanced Classes)")
print("="*80)

X_train_s, X_test_s, y_train_s, y_test_s, groups_train_s, groups_test_s = train_test_split(
    X, y, groups,
    test_size=0.2,
    stratify=y,  # Ensure balanced class distribution
    random_state=42
)

# Check class balance
train_resp_rate_s = y_train_s.mean()
test_resp_rate_s = y_test_s.mean()

print(f"\nSplit sizes:")
print(f"  Train: {len(y_train_s)} trials ({len(np.unique(groups_train_s))} flies)")
print(f"  Test:  {len(y_test_s)} trials ({len(np.unique(groups_test_s))} flies)")

# Check for fly leakage
train_flies_s = set(groups_train_s)
test_flies_s = set(groups_test_s)
overlap = train_flies_s & test_flies_s

print(f"\nFly distribution:")
print(f"  Flies only in train: {len(train_flies_s - test_flies_s)}")
print(f"  Flies only in test:  {len(test_flies_s - train_flies_s)}")
print(f"  Flies in BOTH:       {len(overlap)}")

if overlap:
    print(f"  ⚠️  {len(overlap)} flies appear in both train and test!")
    # Calculate leakage percentage
    total_train_trials = len([g for g in groups_train_s if g in overlap])
    total_test_trials = len([g for g in groups_test_s if g in overlap])
    print(f"     {total_train_trials}/{len(y_train_s)} train trials from shared flies ({total_train_trials/len(y_train_s):.1%})")
    print(f"     {total_test_trials}/{len(y_test_s)} test trials from shared flies ({total_test_trials/len(y_test_s):.1%})")

print(f"\nClass distribution:")
print(f"  Train responder rate: {train_resp_rate_s:.1%}")
print(f"  Test responder rate:  {test_resp_rate_s:.1%}")
print(f"  Distribution gap:     {abs(train_resp_rate_s - test_resp_rate_s):.1%}")

# Train model
model_strat = LogisticRegression(max_iter=1000, random_state=42)
model_strat.fit(X_train_s, y_train_s)

train_acc_s = model_strat.score(X_train_s, y_train_s)
test_acc_s = model_strat.score(X_test_s, y_test_s)
test_f1_s = f1_score(y_test_s, model_strat.predict(X_test_s))

print(f"\nPerformance:")
print(f"  Train Accuracy: {train_acc_s:.1%}")
print(f"  Test Accuracy:  {test_acc_s:.1%}")
print(f"  Test F1 Score:  {test_f1_s:.1%}")
print(f"  Overfitting Gap: {(train_acc_s - test_acc_s):.1%}")

# =============================================================================
# STRATEGY 3: Stratified with Fly Constraint (Hybrid)
# =============================================================================

print("\n" + "="*80)
print("STRATEGY 3: Hybrid - Stratified with Fly Preference")
print("="*80)

# Group flies by response rate
fly_stats = merged.groupby('fly').agg({
    'label_binary': ['count', 'mean']
}).round(3)
fly_stats.columns = ['n_trials', 'response_rate']

# Sort flies by response rate and alternate assignment
fly_stats_sorted = fly_stats.sort_values('response_rate')
flies_sorted = fly_stats_sorted.index.tolist()

# Alternate flies between train and test to balance response rates
train_flies_h = []
test_flies_h = []

for i, fly in enumerate(flies_sorted):
    if i % 5 == 0:  # Every 5th fly goes to test (20% of flies)
        test_flies_h.append(fly)
    else:
        train_flies_h.append(fly)

# Create splits
train_mask_h = merged['fly'].isin(train_flies_h)
test_mask_h = merged['fly'].isin(test_flies_h)

X_train_h = X[train_mask_h]
X_test_h = X[test_mask_h]
y_train_h = y[train_mask_h]
y_test_h = y[test_mask_h]

# Check balance
train_resp_rate_h = y_train_h.mean()
test_resp_rate_h = y_test_h.mean()

print(f"\nSplit sizes:")
print(f"  Train: {len(y_train_h)} trials ({len(train_flies_h)} flies)")
print(f"  Test:  {len(y_test_h)} trials ({len(test_flies_h)} flies)")

print(f"\nClass distribution:")
print(f"  Train responder rate: {train_resp_rate_h:.1%}")
print(f"  Test responder rate:  {test_resp_rate_h:.1%}")
print(f"  Distribution gap:     {abs(train_resp_rate_h - test_resp_rate_h):.1%}")

# Train model
model_hybrid = LogisticRegression(max_iter=1000, random_state=42)
model_hybrid.fit(X_train_h, y_train_h)

train_acc_h = model_hybrid.score(X_train_h, y_train_h)
test_acc_h = model_hybrid.score(X_test_h, y_test_h)
test_f1_h = f1_score(y_test_h, model_hybrid.predict(X_test_h))

print(f"\nPerformance:")
print(f"  Train Accuracy: {train_acc_h:.1%}")
print(f"  Test Accuracy:  {test_acc_h:.1%}")
print(f"  Test F1 Score:  {test_f1_h:.1%}")
print(f"  Overfitting Gap: {(train_acc_h - test_acc_h):.1%}")

# =============================================================================
# FINAL COMPARISON
# =============================================================================

print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Strategy': ['Group-Aware', 'Stratified Random', 'Hybrid'],
    'Test Accuracy': [test_acc_g, test_acc_s, test_acc_h],
    'Test F1': [test_f1_g, test_f1_s, test_f1_h],
    'Train-Test Gap': [train_acc_g - test_acc_g, train_acc_s - test_acc_s, train_acc_h - test_acc_h],
    'Class Balance Gap': [
        abs(train_resp_rate - test_resp_rate),
        abs(train_resp_rate_s - test_resp_rate_s),
        abs(train_resp_rate_h - test_resp_rate_h)
    ],
    'Fly Leakage': ['None', f'{len(overlap)} flies', 'None']
})

print(f"\n{results.to_string(index=False)}")

# Recommend best strategy
best_idx = results['Test Accuracy'].idxmax()
best_strategy = results.loc[best_idx, 'Strategy']

print(f"\n{'='*80}")
print(f"RECOMMENDATION: {best_strategy}")
print(f"{'='*80}")

if best_strategy == 'Group-Aware':
    print(f"\nGroup-aware splitting performs best, but ensure:")
    print(f"  1. Use multiple random seeds to avoid unlucky splits")
    print(f"  2. Check class balance gaps (<5%) before training")
    print(f"  3. Consider hybrid approach if imbalance persists")
elif best_strategy == 'Stratified Random':
    print(f"\nStratified random splitting performs best, BUT:")
    print(f"  ⚠️  {len(overlap)} flies appear in both train and test")
    print(f"  This creates data leakage - model may learn fly-specific patterns")
    print(f"  Use with caution - validate on completely held-out flies")
else:
    print(f"\nHybrid approach balances class distribution while avoiding leakage")
    print(f"  Recommended for production use")

print("\n" + "="*80)
