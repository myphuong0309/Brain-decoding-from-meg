#!/bin/bash

# Evaluation script for Task 2: Phoneme Classification
# This script evaluates the trained phoneme classification model on the test set

python scripts/evaluate_task2.py \
    --data_path ./assets/data_task2 \
    --model_dir ./output_task2 \
    --output_dir ./test_results_task2 \
    --batch_size 32 \
    --model_input_size 306 \
    --grouped_samples 100 \
    --path_norm_global_channel_zscore assets/norm_task2/time
