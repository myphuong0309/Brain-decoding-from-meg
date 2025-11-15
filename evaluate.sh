#!/bin/bash
# Brain Speech Classifier Evaluation Script

set -e

# Parse command line arguments
CHECKPOINT_PATH=${1:-""}  # Specific checkpoint file (optional)
CKPT_PATH=${2:-"./output"}
DATA_PATH=${3:-"./assets/data"}
FOLD=${4:-""}  # Specific fold number (optional, empty means all folds)
MODEL_INPUT=${5:-23}  # 23 for speech sensors, 306 for all sensors
N_SPLITS=${6:-5}
EVAL_BATCH_SIZE=${7:-32}
OUTPUT_DIR=${8:-"./evaluation_results"}
PATH_NORM=${9:-"assets/norm/time"}
USE_CPU=${10:-""}  # pass "--cpu" if true

echo "[INFO] Starting evaluation with the following configuration:"
echo "  Checkpoint path: $CKPT_PATH"
echo "  Data path: $DATA_PATH"
echo "  Input size: $MODEL_INPUT"
echo "  N-fold CV: $N_SPLITS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

if [ -n "$CHECKPOINT_PATH" ] && [ -n "$FOLD" ]; then
    # Evaluate specific checkpoint and fold
    echo "[INFO] Evaluating fold $FOLD with checkpoint: $CHECKPOINT_PATH"
    python scripts/evaluate.py \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --fold "$FOLD" \
      --data_path "$DATA_PATH" \
      --model_input_size "$MODEL_INPUT" \
      --n_splits "$N_SPLITS" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --output_dir "$OUTPUT_DIR" \
      --path_norm_global_channel_zscore "$PATH_NORM" \
      $USE_CPU
elif [ -n "$FOLD" ]; then
    echo "[ERROR] When specifying --fold, you must also provide --checkpoint_path"
    exit 1
else
    # Evaluate all folds
    echo "[INFO] Evaluating all folds..."
    python scripts/evaluate.py \
      --ckpt_path "$CKPT_PATH" \
      --data_path "$DATA_PATH" \
      --model_input_size "$MODEL_INPUT" \
      --n_splits "$N_SPLITS" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --output_dir "$OUTPUT_DIR" \
      --path_norm_global_channel_zscore "$PATH_NORM" \
      $USE_CPU
fi

echo ""
echo "[INFO] Evaluation completed!"
echo "[INFO] Results saved to: $OUTPUT_DIR"
