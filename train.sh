#!/bin/bash
# Brain Speech Classifier Training Script

set -e

# Parse command line arguments
DATA_PATH=${1:-"./assets/data"}
CKPT_PATH=${2:-"./output"}
EPOCHS=${3:-15}
MODEL_DIM=${4:-256}
MODEL_INPUT=${5:-306}  # 23 for speech sensors, 306 for all sensors
LR=${6:-1e-5}
DROPOUT=${7:-0.2}
LSTM_LAYERS=${8:-2}
WEIGHT_DECAY=${9:-1e-5}
TRAIN_BATCH_SIZE=${10:-32}
EVAL_BATCH_SIZE=${11:-32}
BATCH_NORM=${12:-""}        # pass "--batch_norm" if true
BI_DIRECTIONAL=${13:-""}    # pass "--bi_directional" if true
N_SPLITS=${14:-5}
MONITOR=${15:-"val_f1_macro"}  # "val_f1_macro" or "val_loss"
EARLY_STOPPING_PATIENCE=${16:-10}
EARLY_STOPPING_MIN_DELTA=${17:-0.001}
PATH_NORM=${18:-"assets/norm/time"}

echo "[INFO] Starting training with the following configuration:"
echo "  Data path: $DATA_PATH"
echo "  Checkpoint path: $CKPT_PATH"
echo "  Epochs: $EPOCHS"
echo "  Model dim: $MODEL_DIM"
echo "  Input size: $MODEL_INPUT"
echo "  Learning rate: $LR"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  N-fold CV: $N_SPLITS"
echo ""

python scripts/train.py \
  --data_path "$DATA_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --epochs "$EPOCHS" \
  --model_dim "$MODEL_DIM" \
  --model_input_size "$MODEL_INPUT" \
  --lr "$LR" \
  --dropout_rate "$DROPOUT" \
  --lstm_layers "$LSTM_LAYERS" \
  --weight_decay "$WEIGHT_DECAY" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --n_splits "$N_SPLITS" \
  --monitor "$MONITOR" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" \
  --path_norm_global_channel_zscore "$PATH_NORM" \
  $BATCH_NORM \
  $BI_DIRECTIONAL

echo ""
echo "[INFO] Training completed!"
echo "[INFO] Results saved to: $CKPT_PATH"
