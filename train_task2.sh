#!/bin/bash

# Training script for Task 2: Phoneme Classification (Simple Conv baseline)

python scripts/train_task2.py \
    --data_path ./assets/data_task2 \
    --ckpt_path ./output_task2 \
    --model_input_size 306 \
    --epochs 100 \
    --model_dim 128 \
    --dropout_rate 0.2 \
    --lr 5e-4 \
    --monitor val_f1_macro \
    --weight_decay 1e-5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --label_smoothing 0.05 \
    --grouped_samples 100 \
    --early_stopping_patience 20 \
    --early_stopping_min_delta 0.0005 \
    --path_norm_global_channel_zscore assets/norm_task2/time
