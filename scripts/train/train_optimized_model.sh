#!/bin/bash
# Optimized training script with recommended fixes for false negative reduction
# Based on comprehensive diagnostic analysis

set -e  # Exit on error

echo "=================================================="
echo "  TRAINING OPTIMIZED FLY BEHAVIOR MODEL"
echo "=================================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FLYBEHAVIOR_RESPONSE_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_DIR="${FLYBEHAVIOR_RESPONSE_DATA_DIR:-$REPO_ROOT/data}"
OUTPUTS_DIR="${FLYBEHAVIOR_RESPONSE_OUTPUTS_DIR:-$REPO_ROOT/outputs}"
DATA_CSV="${DATA_DIR}/all_envelope_rows_wide.csv"
LABELS_CSV="${DATA_DIR}/scoring_results_opto_new_MINIMAL.csv"
ARTIFACTS_DIR="${OUTPUTS_DIR}/artifacts"
SEED=42

echo ""
echo "Configuration:"
echo "  Data:    $DATA_CSV"
echo "  Labels:  $LABELS_CSV"
echo "  Seed:    $SEED"
echo ""

# ============================================================================
# RECOMMENDATION #1: Train LogReg with Class Weighting
# ============================================================================

echo "🔧 Training Logistic Regression with optimized class weights..."
echo "   This reduces false negatives by penalizing missed responders 3x more"
echo ""

flybehavior-response train \
    --data-csv "$DATA_CSV" \
    --labels-csv "$LABELS_CSV" \
    --model logreg \
    --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
    --n-pcs 6 \
    --use-raw-pca \
    --logreg-class-weights "0:1.0,1:3.0" \
    --logreg-max-iter 1000 \
    --logreg-solver "lbfgs" \
    --group-column "fly" \
    --test-size 0.2 \
    --seed $SEED \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --verbose

echo ""
echo "✅ Logistic Regression trained successfully!"
echo ""

# ============================================================================
# RECOMMENDATION #2: Train Random Forest for Ensemble
# ============================================================================

echo "🌲 Training Random Forest with conservative max_depth..."
echo "   This prevents overfitting while maintaining high recall"
echo ""

flybehavior-response train \
    --data-csv "$DATA_CSV" \
    --labels-csv "$LABELS_CSV" \
    --model random_forest \
    --features "global_max,local_min,local_max,local_max_before,local_max_during,local_max_during_over_global_min,AUC-During,Peak-Value" \
    --n-pcs 6 \
    --use-raw-pca \
    --rf-n-estimators 100 \
    --rf-max-depth 8 \
    --rf-class-weights "0:1.0,1:3.0" \
    --group-column "fly" \
    --test-size 0.2 \
    --seed $SEED \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --verbose

echo ""
echo "✅ Random Forest trained successfully!"
echo ""

# ============================================================================
# Evaluate and Compare
# ============================================================================

echo "📊 Evaluating models..."
echo ""

# Find most recent run directory
LATEST_RUN=$(ls -t "$ARTIFACTS_DIR" | grep "2025-" | head -1)

echo "Latest run: $LATEST_RUN"
echo ""

# Run quick diagnostics
python "$REPO_ROOT/scripts/analysis/quick_diagnostics.py"

echo ""
echo "=================================================="
echo "  TRAINING COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Check results in: $ARTIFACTS_DIR/$LATEST_RUN/"
echo "  2. Test with lowered threshold (0.35) for predictions"
echo "  3. Run: flybehavior-response predict --threshold 0.35 ..."
echo ""
echo "Expected improvements:"
echo "  - False Negative Rate: 21% → 8-12%"
echo "  - Test Accuracy: ~90-92%"
echo "  - Balanced precision/recall tradeoff"
echo ""
