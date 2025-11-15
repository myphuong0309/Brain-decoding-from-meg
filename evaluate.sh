#!/bin/bash
# Brain Speech Classifier Test Set Evaluation Script - Ensemble Mode

set -e

# Parse command line arguments
CKPT_BASE_PATH=${1:-"./output"}
DATA_PATH=${2:-"./assets/data"}
MODEL_INPUT=${3:-23}  # 23 for speech sensors, 306 for all sensors
EVAL_BATCH_SIZE=${4:-32}
OUTPUT_DIR=${5:-"./test_results"}
PATH_NORM=${6:-"assets/norm/time"}
USE_CPU=${7:-""}  # pass "--cpu" if true

echo "=========================================="
echo "Test Set Evaluation"
echo "=========================================="
echo "[INFO] Evaluating on Sherlock1 Sessions 11-12"
echo "[INFO] Using ensemble of all folds"
echo "  Checkpoint base path: $CKPT_BASE_PATH"
echo "  Data path: $DATA_PATH"
echo "  Input size: $MODEL_INPUT"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python scripts/evaluate.py \
  --ckpt_path "$CKPT_BASE_PATH" \
  --data_path "$DATA_PATH" \
  --model_input_size "$MODEL_INPUT" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --path_norm_global_channel_zscore "$PATH_NORM" \
  $USE_CPU

echo ""
echo "[INFO] Ensemble test evaluation completed!"
echo "[INFO] Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/metrics.json"
