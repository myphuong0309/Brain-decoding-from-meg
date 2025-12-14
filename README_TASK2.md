# Task 2: Phoneme Classification from MEG Signals

This directory contains the implementation for **Task 2: Phoneme Classification**, which classifies 111 different phoneme classes from MEG brain signals. This task is separate from Task 1 (Speech Detection) and uses different data and models.

## Overview

- **Task**: Multiclass classification of phonemes
- **Number of Classes**: 111 phoneme classes (e.g., ah_S, s_B, t_I, etc.)
- **Dataset**: Located in `assets/data_task2/`
- **Train Split**: Sherlock1 sessions 1-10
- **Validation Split**: Sherlock1 session 11
- **Test Split**: Sherlock2 sessions 1-11

## Project Structure

```
task1/  (root directory)
├── assets/
│   ├── data_task2/                    # Task 2 dataset (phoneme classification)
│   │   ├── Sherlock1/
│   │   └── Sherlock2/
│   └── norm_task2/                    # Task 2 normalization statistics
│       └── time/
│           └── global_stats.pt
├── models/
│   ├── model.py                       # Task 1 model (speech detection)
│   └── model_task2.py                 # Task 2 model (phoneme classification) ✨ NEW
├── scripts/
│   ├── train.py                       # Task 1 training
│   ├── train_task2.py                 # Task 2 training ✨ NEW
│   ├── evaluate.py                    # Task 1 evaluation
│   └── evaluate_task2.py              # Task 2 evaluation ✨ NEW
├── utils/
│   ├── processed_data.py              # Task 1 data processing
│   └── processed_data_task2.py        # Task 2 data processing ✨ NEW
├── output_task2/                      # Task 2 model checkpoints (created during training)
├── test_results_task2/                # Task 2 evaluation results (created during testing)
├── train_task2.sh                     # Task 2 training script ✨ NEW
├── evaluate_task2.sh                  # Task 2 evaluation script ✨ NEW
└── compute_stats_task2.py             # Compute normalization stats for task2 ✨ NEW
```

## Quick Start

### 1. Compute Normalization Statistics (Already Done)

```bash
python compute_stats_task2.py
```

This creates `assets/norm_task2/time/global_stats.pt` with per-channel mean and std.

### 2. Train the Model

```bash
./train_task2.sh
```

Or run manually:

```bash
python scripts/train_task2.py \
    --data_path ./assets/data_task2 \
    --ckpt_path ./output_task2 \
    --model_input_size 23 \
    --epochs 30 \
    --model_dim 128 \
    --dropout_rate 0.2 \
    --lstm_layers 2 \
    --bi_directional \
    --lr 1e-5 \
    --label_smoothing 0.1
```

### 3. Evaluate the Model

```bash
./evaluate_task2.sh
```

Or run manually:

```bash
python scripts/evaluate_task2.py \
    --data_path ./assets/data_task2 \
    --model_dir ./output_task2 \
    --output_dir ./test_results_task2 \
    --model_input_size 23
```

## Key Differences from Task 1 (Speech Detection)

| Aspect               | Task 1 (Speech Detection)                 | Task 2 (Phoneme Classification)          |
| -------------------- | ----------------------------------------- | ---------------------------------------- |
| **Objective**        | Binary classification (speech vs silence) | Multiclass classification (111 phonemes) |
| **Data Path**        | `assets/data/`                            | `assets/data_task2/`                     |
| **Output Classes**   | 2 (binary)                                | 111 (multiclass)                         |
| **Loss Function**    | BCEWithLogitsLoss                         | CrossEntropyLoss                         |
| **Label Type**       | Binary labels                             | Phoneme strings mapped to indices        |
| **Normalization**    | `assets/norm/time/`                       | `assets/norm_task2/time/`                |
| **Model Files**      | `model.py`                                | `model_task2.py`                         |
| **Data Processing**  | `processed_data.py`                       | `processed_data_task2.py`                |
| **Output Directory** | `output/`                                 | `output_task2/`                          |

## Model Architecture

The phoneme classification model (`BrainPhonemeClassifier`) uses:

- **Conv1D** layer for temporal feature extraction
- **LSTM** layers (bidirectional) for sequential modeling
- **Attention Pooling** for aggregating temporal information
- **Linear Classifier** outputting 111 classes
- **Label Smoothing** for regularization
- **Dropout** and **Batch Normalization** for robustness

## Evaluation Metrics

The evaluation script computes:

- **Accuracy**: Overall classification accuracy
- **F1 Score** (Macro, Weighted, Micro)
- **Precision** and **Recall** (Macro, Weighted, Micro)
- **Confusion Matrix** (visualized for top 50 classes)
- **Per-Class F1 Scores** (visualized for top 30 classes)
- **Classification Report** (saved to file)

## Output Files

After training and evaluation, you'll find:

**Training Outputs** (`output_task2/`):

- Model checkpoints (`.ckpt` files)
- Training logs
- `phoneme_to_idx.pt` - Phoneme label mapping
- `hparams.yaml` - Hyperparameters

**Evaluation Outputs** (`test_results_task2/`):

- `metrics_task2.json` - Evaluation metrics
- `predictions_task2.npz` - Predictions and logits
- `confusion_matrix_task2.png` - Confusion matrix visualization
- `per_class_f1_task2.png` - Per-class F1 scores
- `classification_report_task2.txt` - Detailed classification report

## Important Notes

⚠️ **Task Separation**: Task 1 and Task 2 are completely independent:

- Different datasets (`data/` vs `data_task2/`)
- Different models (`model.py` vs `model_task2.py`)
- Different normalization statistics (`norm/` vs `norm_task2/`)
- Different output directories (`output/` vs `output_task2/`)

✅ **No Conflicts**: You can safely work on Task 2 without affecting Task 1.

## Training Tips

1. **Label Smoothing**: Set `--label_smoothing 0.1` to prevent overfitting
2. **Batch Size**: Increase if you have enough GPU memory
3. **Learning Rate**: Use `1e-5` for stable convergence with 111 classes
4. **Early Stopping**: Monitors `val_f1_macro` with patience=10
5. **Bidirectional LSTM**: Use `--bi_directional` for better context modeling

## Troubleshooting

**Issue**: Out of memory during training

- **Solution**: Reduce `--train_batch_size` or `--model_dim`

**Issue**: Missing normalization statistics

- **Solution**: Run `python compute_stats_task2.py`

**Issue**: Cannot find checkpoint

- **Solution**: Check `output_task2/` directory exists and contains `.ckpt` files

**Issue**: Unknown phoneme warnings

- **Solution**: This shouldn't happen, but if it does, check the event files in `assets/data_task2/`

## Contact

For questions about Task 2 implementation, refer to the code documentation in:

- `models/model_task2.py`
- `utils/processed_data_task2.py`
- `scripts/train_task2.py`
- `scripts/evaluate_task2.py`
