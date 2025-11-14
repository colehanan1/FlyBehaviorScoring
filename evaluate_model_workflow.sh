#!/bin/bash
# Comprehensive model evaluation workflow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  MODEL EVALUATION WORKFLOW${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if run directory is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide run directory${NC}"
    echo "Usage: ./evaluate_model_workflow.sh <run-dir> [model-name]"
    echo ""
    echo "Example:"
    echo "  ./evaluate_model_workflow.sh artifacts/all_weighted/2025-11-13T22-18-43Z random_forest"
    exit 1
fi

RUN_DIR=$1
MODEL=${2:-random_forest}

# Check if directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo -e "${RED}Error: Run directory not found: $RUN_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Run directory: $RUN_DIR"
echo -e "${GREEN}✓${NC} Model: $MODEL"
echo ""

# Step 1: Analyze errors
echo -e "${YELLOW}Step 1: Analyzing model errors...${NC}"
python analyze_model_errors.py \
    --run-dir "$RUN_DIR" \
    --model "$MODEL" \
    --output-dir "$RUN_DIR/error_analysis"

echo ""

# Step 2: Check for prediction files
echo -e "${YELLOW}Step 2: Checking prediction files...${NC}"
for split in train test validation; do
    pred_file="$RUN_DIR/predictions_${MODEL}_${split}.csv"
    if [ -f "$pred_file" ]; then
        n_rows=$(wc -l < "$pred_file")
        n_rows=$((n_rows - 1))  # Subtract header
        echo -e "${GREEN}✓${NC} Found $split predictions: $n_rows samples"
    else
        echo -e "${YELLOW}⚠${NC}  No $split predictions found"
    fi
done

echo ""

# Step 3: Summary
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  EVALUATION COMPLETE!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}📁 Output files:${NC}"
echo "  $RUN_DIR/error_analysis/error_summary_${MODEL}.json"
echo "  $RUN_DIR/error_analysis/errors_${MODEL}_*_false_*.csv"
echo "  $RUN_DIR/error_analysis/confusion_matrices_${MODEL}.png"
echo "  $RUN_DIR/error_analysis/probability_distributions_${MODEL}.png"
echo ""
echo -e "${GREEN}📊 Next steps:${NC}"
echo "  1. Open the PNG files to see visualizations"
echo "  2. Review the false positive/negative CSVs to see what went wrong"
echo "  3. Check error_summary JSON for detailed statistics"
echo ""
echo -e "${YELLOW}To predict on NEW unlabeled data:${NC}"
echo "  flybehavior-response predict \\"
echo "    --data-csv /path/to/unlabeled_data.csv \\"
echo "    --model-path $RUN_DIR/model_${MODEL}.joblib \\"
echo "    --output-csv predictions_new.csv"
echo ""
