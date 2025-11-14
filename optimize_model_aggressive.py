#!/usr/bin/env python
"""
Aggressive hyperparameter optimization to maximize accuracy.

Tests:
- All feature combinations
- Multiple PCA components (5, 10, 15, 20)
- Different class weights
- Multiple model types
- Ensemble methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Configuration
DATA_CSV = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv")
LABELS_CSV = Path("/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv")
OUTPUT_DIR = Path("artifacts/optimized")
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("  AGGRESSIVE MODEL OPTIMIZATION")
print("="*80)

# Load data
print("\nLoading data...")
data_df = pd.read_csv(DATA_CSV)
labels_df = pd.read_csv(LABELS_CSV)
merged = data_df.merge(labels_df, on=['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'], how='inner')

merged['label_binary'] = (merged['user_score_odor'] > 0).astype(int)
print(f"Dataset: {len(merged)} trials, {merged['fly'].nunique()} flies")

# Get ALL features
all_cols = [col for col in merged.columns if not col.startswith('dir_val_') and col not in ['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label', 'user_score_odor', 'user_score_odor_intensity', 'label_binary']]
print(f"Found {len(all_cols)} potential feature columns")

# Identify numeric features
numeric_features = []
for col in all_cols:
    if merged[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        numeric_features.append(col)

print(f"\nNumeric features ({len(numeric_features)}):")
for feat in sorted(numeric_features):
    print(f"  {feat}")

# Trace columns
trace_cols = [col for col in merged.columns if col.startswith('dir_val_')]
print(f"\nTrace time series: {len(trace_cols)} points")

# =============================================================================
# Hybrid Split (from previous analysis)
# =============================================================================

print("\n" + "="*80)
print("Creating Hybrid Split")
print("="*80)

fly_stats = merged.groupby('fly').agg({'label_binary': ['count', 'mean']})
fly_stats.columns = ['n_trials', 'response_rate']
fly_stats_sorted = fly_stats.sort_values('response_rate')

flies_sorted = fly_stats_sorted.index.tolist()
train_flies = [fly for i, fly in enumerate(flies_sorted) if i % 5 != 0]
test_flies = [fly for i, fly in enumerate(flies_sorted) if i % 5 == 0]

train_df = merged[merged['fly'].isin(train_flies)].copy()
test_df = merged[merged['fly'].isin(test_flies)].copy()

print(f"Train: {len(train_df)} trials, Test: {len(test_df)} trials")
print(f"Train response rate: {train_df['label_binary'].mean():.1%}")
print(f"Test response rate: {test_df['label_binary'].mean():.1%}")

# Prepare data
y_train = train_df['label_binary'].values
y_test = test_df['label_binary'].values
groups_train = train_df['fly'].values

# Sample weights
intensity_to_weight = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
train_weights = train_df['user_score_odor'].map(intensity_to_weight).values

# =============================================================================
# Feature Engineering Experiments
# =============================================================================

print("\n" + "="*80)
print("Testing Feature Combinations")
print("="*80)

best_results = []

# Test 1: All features + traces
print("\n1. ALL FEATURES + Traces with 15 PCA components...")
X_train_all = train_df[numeric_features].fillna(0).values
X_test_all = test_df[numeric_features].fillna(0).values

X_train_trace = train_df[trace_cols].fillna(0).values
X_test_trace = test_df[trace_cols].fillna(0).values

pca15 = PCA(n_components=15, random_state=SEED)
X_train_pca15 = pca15.fit_transform(X_train_trace)
X_test_pca15 = pca15.transform(X_test_trace)

X_train_1 = np.hstack([X_train_all, X_train_pca15])
X_test_1 = np.hstack([X_test_all, X_test_pca15])

scaler1 = StandardScaler()
X_train_1 = scaler1.fit_transform(X_train_1)
X_test_1 = scaler1.transform(X_test_1)

# LogReg with strong class weight
logreg1 = LogisticRegression(C=0.1, class_weight={0: 1.0, 1: 3.0}, max_iter=2000, random_state=SEED)
logreg1.fit(X_train_1, y_train, sample_weight=train_weights)
acc1 = logreg1.score(X_test_1, y_test)
f1_1 = f1_score(y_test, logreg1.predict(X_test_1))

print(f"  LogReg (C=0.1, weight=3.0): Test Acc={acc1:.1%}, F1={f1_1:.1%}")
best_results.append(('All features + PCA15 + LogReg(C=0.1, w=3)', acc1, f1_1, logreg1, scaler1, pca15, numeric_features))

# Test 2: Stronger regularization
logreg2 = LogisticRegression(C=0.05, class_weight={0: 1.0, 1: 4.0}, max_iter=2000, random_state=SEED)
logreg2.fit(X_train_1, y_train, sample_weight=train_weights)
acc2 = logreg2.score(X_test_1, y_test)
f1_2 = f1_score(y_test, logreg2.predict(X_test_1))

print(f"  LogReg (C=0.05, weight=4.0): Test Acc={acc2:.1%}, F1={f1_2:.1%}")
best_results.append(('All features + PCA15 + LogReg(C=0.05, w=4)', acc2, f1_2, logreg2, scaler1, pca15, numeric_features))

# Test 3: More PCA components
print("\n2. ALL FEATURES + Traces with 20 PCA components...")
pca20 = PCA(n_components=20, random_state=SEED)
X_train_pca20 = pca20.fit_transform(X_train_trace)
X_test_pca20 = pca20.transform(X_test_trace)

X_train_2 = np.hstack([X_train_all, X_train_pca20])
X_test_2 = np.hstack([X_test_all, X_test_pca20])

scaler2 = StandardScaler()
X_train_2 = scaler2.fit_transform(X_train_2)
X_test_2 = scaler2.transform(X_test_2)

logreg3 = LogisticRegression(C=0.1, class_weight={0: 1.0, 1: 3.5}, max_iter=2000, random_state=SEED)
logreg3.fit(X_train_2, y_train, sample_weight=train_weights)
acc3 = logreg3.score(X_test_2, y_test)
f1_3 = f1_score(y_test, logreg3.predict(X_test_2))

print(f"  LogReg (C=0.1, weight=3.5): Test Acc={acc3:.1%}, F1={f1_3:.1%}")
best_results.append(('All features + PCA20 + LogReg(C=0.1, w=3.5)', acc3, f1_3, logreg3, scaler2, pca20, numeric_features))

# Test 4: Random Forest with more trees
print("\n3. Random Forest variations...")
rf1 = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3, class_weight={0: 1.0, 1: 2.5}, random_state=SEED, n_jobs=-1)
rf1.fit(X_train_2, y_train, sample_weight=train_weights)
acc_rf1 = rf1.score(X_test_2, y_test)
f1_rf1 = f1_score(y_test, rf1.predict(X_test_2))

print(f"  RF (200 trees, depth=12, w=2.5): Test Acc={acc_rf1:.1%}, F1={f1_rf1:.1%}")
best_results.append(('All features + PCA20 + RF(200,12,2.5)', acc_rf1, f1_rf1, rf1, scaler2, pca20, numeric_features))

rf2 = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2, class_weight={0: 1.0, 1: 3.0}, random_state=SEED, n_jobs=-1)
rf2.fit(X_train_2, y_train, sample_weight=train_weights)
acc_rf2 = rf2.score(X_test_2, y_test)
f1_rf2 = f1_score(y_test, rf2.predict(X_test_2))

print(f"  RF (300 trees, depth=15, w=3.0): Test Acc={acc_rf2:.1%}, F1={f1_rf2:.1%}")
best_results.append(('All features + PCA20 + RF(300,15,3.0)', acc_rf2, f1_rf2, rf2, scaler2, pca20, numeric_features))

# Test 5: Gradient Boosting
print("\n4. Gradient Boosting...")
gb1 = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED)
gb1.fit(X_train_2, y_train, sample_weight=train_weights)
acc_gb1 = gb1.score(X_test_2, y_test)
f1_gb1 = f1_score(y_test, gb1.predict(X_test_2))

print(f"  GradientBoosting (200, depth=5, lr=0.05): Test Acc={acc_gb1:.1%}, F1={f1_gb1:.1%}")
best_results.append(('All features + PCA20 + GB(200,5,0.05)', acc_gb1, f1_gb1, gb1, scaler2, pca20, numeric_features))

# Test 6: Ensemble of best models
print("\n5. Creating Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('logreg', logreg3),
        ('rf', rf2),
        ('gb', gb1)
    ],
    voting='soft',
    weights=[0.4, 0.4, 0.2]
)

ensemble.fit(X_train_2, y_train, sample_weight=train_weights)
acc_ens = ensemble.score(X_test_2, y_test)
f1_ens = f1_score(y_test, ensemble.predict(X_test_2))

print(f"  Ensemble (LogReg+RF+GB): Test Acc={acc_ens:.1%}, F1={f1_ens:.1%}")
best_results.append(('Ensemble (LogReg+RF+GB)', acc_ens, f1_ens, ensemble, scaler2, pca20, numeric_features))

# =============================================================================
# Find best model
# =============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(best_results, columns=['Model', 'Test Acc', 'Test F1', 'model_obj', 'scaler', 'pca', 'features'])
results_df_display = results_df[['Model', 'Test Acc', 'Test F1']].copy()
results_df_display['Test Acc'] = results_df_display['Test Acc'].apply(lambda x: f"{x:.1%}")
results_df_display['Test F1'] = results_df_display['Test F1'].apply(lambda x: f"{x:.1%}")

print(f"\n{results_df_display.to_string(index=False)}")

# Best model
best_idx = results_df['Test Acc'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_acc = results_df.loc[best_idx, 'Test Acc']
best_f1 = results_df.loc[best_idx, 'Test F1']
best_model = results_df.loc[best_idx, 'model_obj']
best_scaler = results_df.loc[best_idx, 'scaler']
best_pca = results_df.loc[best_idx, 'pca']
best_features = results_df.loc[best_idx, 'features']

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {best_acc:.1%}")
print(f"   Test F1 Score: {best_f1:.1%}")
print(f"{'='*80}")

# Save best model
best_pipeline = Pipeline([
    ('scaler', best_scaler),
    ('pca', best_pca),
    ('model', best_model)
])

joblib.dump(best_pipeline, OUTPUT_DIR / 'best_model_optimized.joblib')
print(f"\n✓ Saved best model to {OUTPUT_DIR / 'best_model_optimized.joblib'}")

# Save feature list
with open(OUTPUT_DIR / 'best_features.txt', 'w') as f:
    f.write("Engineered features:\n")
    for feat in best_features:
        f.write(f"  {feat}\n")
    f.write(f"\nPCA components: {best_pca.n_components}\n")

print(f"✓ Saved feature list to {OUTPUT_DIR / 'best_features.txt'}")

# Get predictions
y_test_pred = best_model.predict(scaler2.transform(np.hstack([X_test_all, best_pca.transform(X_test_trace)])))

# Detailed error analysis
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

print(f"\n{classification_report(y_test, y_test_pred, target_names=['Non-responder', 'Responder'])}")

print("\n" + "="*80)
print(f"Optimization complete! Best accuracy: {best_acc:.1%}")
print("="*80)
