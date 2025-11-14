#!/usr/bin/env python
"""
Train model with hybrid splitting strategy:
- Assigns flies to train/test by alternating sorted response rates
- Ensures balanced class distribution across splits
- Maintains fly-level holdout (no leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configuration
DATA_CSV = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv")
LABELS_CSV = Path("/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv")
OUTPUT_DIR = Path("artifacts/hybrid_split")
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("  TRAINING WITH HYBRID SPLIT STRATEGY")
print("="*80)

# Load data
print("\nLoading data...")
data_df = pd.read_csv(DATA_CSV)
labels_df = pd.read_csv(LABELS_CSV)
merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')

merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)
print(f"Dataset: {len(merged)} trials, {merged['fly'].nunique()} flies")
print(f"Class balance: {merged['label_binary'].mean():.1%} responders")

# Features
feature_cols = ['local_min', 'local_max', 'local_max_before', 'local_max_during',
                'local_max_during_over_global_min', 'AUC-During', 'Peak-Value']
available_features = [col for col in feature_cols if col in merged.columns]

# Also get trace columns for PCA
trace_cols = [col for col in merged.columns if col.startswith('dir_val_')]
print(f"\nFeatures: {len(available_features)} engineered + {len(trace_cols)} trace time points")

# =============================================================================
# HYBRID SPLITTING: Alternate flies by response rate
# =============================================================================

print("\n" + "="*80)
print("Creating Hybrid Split (Balanced Fly Assignment)")
print("="*80)

# Calculate response rate per fly
fly_stats = merged.groupby('fly').agg({
    'label_binary': ['count', 'mean']
}).round(3)
fly_stats.columns = ['n_trials', 'response_rate']
fly_stats_sorted = fly_stats.sort_values('response_rate')

# Alternate assignment to balance response rates
flies_sorted = fly_stats_sorted.index.tolist()
train_flies = []
test_flies = []

for i, fly in enumerate(flies_sorted):
    if i % 5 == 0:  # Every 5th fly → test (20%)
        test_flies.append(fly)
    else:
        train_flies.append(fly)

# Create splits
train_mask = merged['fly'].isin(train_flies)
test_mask = merged['fly'].isin(test_flies)

train_df = merged[train_mask].copy()
test_df = merged[test_mask].copy()

print(f"\nSplit statistics:")
print(f"  Train: {len(train_df)} trials from {len(train_flies)} flies")
print(f"  Test:  {len(test_df)} trials from {len(test_flies)} flies")
print(f"  Train responder rate: {train_df['label_binary'].mean():.1%}")
print(f"  Test responder rate:  {test_df['label_binary'].mean():.1%}")
print(f"  Balance gap: {abs(train_df['label_binary'].mean() - test_df['label_binary'].mean()):.1%}")

# Save split manifest
manifest = merged[['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label']].copy()
manifest['split'] = 'train'
manifest.loc[test_mask, 'split'] = 'test'
manifest.to_csv(OUTPUT_DIR / 'split_manifest.csv', index=False)
print(f"\n✓ Saved split manifest to {OUTPUT_DIR / 'split_manifest.csv'}")

# =============================================================================
# Prepare features
# =============================================================================

print("\nPreparing features...")

# Combine engineered features and trace data
X_train_eng = train_df[available_features].fillna(0).values
X_test_eng = test_df[available_features].fillna(0).values

X_train_trace = train_df[trace_cols].fillna(0).values
X_test_trace = test_df[trace_cols].fillna(0).values

# PCA on traces
print(f"  Applying PCA to {len(trace_cols)} trace time points → 10 components")
pca = PCA(n_components=10, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_trace)
X_test_pca = pca.transform(X_test_trace)

# Combine
X_train = np.hstack([X_train_eng, X_train_pca])
X_test = np.hstack([X_test_eng, X_test_pca])

# Labels and weights
y_train = train_df['label_binary'].values
y_test = test_df['label_binary'].values

# Sample weights based on intensity
intensity_to_weight = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
train_weights = train_df['user_score_odor'].map(intensity_to_weight).values
test_weights = test_df['user_score_odor'].map(intensity_to_weight).values

print(f"  Final feature dimension: {X_train.shape[1]} features")

# =============================================================================
# Train Logistic Regression
# =============================================================================

print("\n" + "="*80)
print("Training Logistic Regression")
print("="*80)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with balanced class weights
logreg = LogisticRegression(
    class_weight='balanced',  # Equivalent to {0: 1.0, 1: 2.4} given current balance
    max_iter=1000,
    solver='lbfgs',
    random_state=SEED
)

logreg.fit(X_train_scaled, y_train, sample_weight=train_weights)

# Evaluate
train_acc = logreg.score(X_train_scaled, y_train)
test_acc = logreg.score(X_test_scaled, y_test)

y_train_pred = logreg.predict(X_train_scaled)
y_test_pred = logreg.predict(X_test_scaled)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"\nPerformance:")
print(f"  Train Accuracy: {train_acc:.1%}")
print(f"  Test Accuracy:  {test_acc:.1%}")
print(f"  Test Precision: {test_precision:.1%}")
print(f"  Test Recall:    {test_recall:.1%}")
print(f"  Test F1 Score:  {test_f1:.1%}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix (Test):")
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

# Save model
logreg_pipeline = Pipeline([
    ('scaler', scaler),
    ('pca_preprocess', pca),
    ('model', logreg)
])

joblib.dump(logreg_pipeline, OUTPUT_DIR / 'model_logreg_hybrid.joblib')
print(f"\n✓ Saved LogReg model to {OUTPUT_DIR / 'model_logreg_hybrid.joblib'}")

# =============================================================================
# Train Random Forest
# =============================================================================

print("\n" + "="*80)
print("Training Random Forest")
print("="*80)

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,  # Prevent overfitting
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train, sample_weight=train_weights)

# Evaluate
train_acc_rf = rf.score(X_train, y_train)
test_acc_rf = rf.score(X_test, y_test)

y_test_pred_rf = rf.predict(X_test)
test_f1_rf = f1_score(y_test, y_test_pred_rf)
test_precision_rf = precision_score(y_test, y_test_pred_rf)
test_recall_rf = recall_score(y_test, y_test_pred_rf)

print(f"\nPerformance:")
print(f"  Train Accuracy: {train_acc_rf:.1%}")
print(f"  Test Accuracy:  {test_acc_rf:.1%}")
print(f"  Test Precision: {test_precision_rf:.1%}")
print(f"  Test Recall:    {test_recall_rf:.1%}")
print(f"  Test F1 Score:  {test_f1_rf:.1%}")

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
print(f"\nConfusion Matrix (Test):")
print(f"  TN={cm_rf[0,0]:3d}  FP={cm_rf[0,1]:3d}")
print(f"  FN={cm_rf[1,0]:3d}  TP={cm_rf[1,1]:3d}")

# Save model
rf_pipeline = Pipeline([
    ('pca_preprocess', pca),
    ('model', rf)
])

joblib.dump(rf_pipeline, OUTPUT_DIR / 'model_rf_hybrid.joblib')
print(f"\n✓ Saved RF model to {OUTPUT_DIR / 'model_rf_hybrid.joblib'}")

# =============================================================================
# Save predictions
# =============================================================================

print("\n" + "="*80)
print("Saving Predictions")
print("="*80)

# LogReg predictions
train_preds_logreg = train_df.copy()
train_preds_logreg['predicted_label'] = y_train_pred
train_preds_logreg['correct'] = y_train == y_train_pred
train_preds_logreg['prob_reaction'] = logreg.predict_proba(X_train_scaled)[:, 1]

test_preds_logreg = test_df.copy()
test_preds_logreg['predicted_label'] = y_test_pred
test_preds_logreg['correct'] = y_test == y_test_pred
test_preds_logreg['prob_reaction'] = logreg.predict_proba(X_test_scaled)[:, 1]

train_preds_logreg.to_csv(OUTPUT_DIR / 'predictions_logreg_train.csv', index=False)
test_preds_logreg.to_csv(OUTPUT_DIR / 'predictions_logreg_test.csv', index=False)

print(f"✓ Saved LogReg predictions")

# RF predictions
test_preds_rf = test_df.copy()
test_preds_rf['predicted_label'] = y_test_pred_rf
test_preds_rf['correct'] = y_test == y_test_pred_rf
test_preds_rf['prob_reaction'] = rf.predict_proba(X_test)[:, 1]

test_preds_rf.to_csv(OUTPUT_DIR / 'predictions_rf_test.csv', index=False)
print(f"✓ Saved RF predictions")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Test Accuracy': [f"{test_acc:.1%}", f"{test_acc_rf:.1%}"],
    'Test F1': [f"{test_f1:.1%}", f"{test_f1_rf:.1%}"],
    'Test Precision': [f"{test_precision:.1%}", f"{test_precision_rf:.1%}"],
    'Test Recall': [f"{test_recall:.1%}", f"{test_recall_rf:.1%}"]
})

print(f"\n{results.to_string(index=False)}")

print(f"\nFiles saved to: {OUTPUT_DIR}/")
print(f"  - model_logreg_hybrid.joblib")
print(f"  - model_rf_hybrid.joblib")
print(f"  - predictions_*.csv")
print(f"  - split_manifest.csv")

print("\n" + "="*80)
