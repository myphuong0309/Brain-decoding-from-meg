import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

# --- The above three lines must stay at the very top for module import to work ---

import argparse
from datetime import datetime
import torch
import torch.distributed as dist
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from models.model_task2 import BrainPhonemeClassifier
from utils.processed_data_task2 import get_dataloaders_task2
from utils.util import get_logger_and_paths, save_hparams


def run_training(args):
    print(f"\n[INFO] Training phoneme classification with fixed train/val split")
    base_dir, logger_list = get_logger_and_paths(
        args.ckpt_path, args.timestamp)
    save_hparams(base_dir, args)

    print(f"[INFO] Using time domain preprocessing for task2 (phoneme classification)")
    train_loader, val_loader, phoneme_to_idx = get_dataloaders_task2(
        data_path=args.data_path,
        num_workers=4,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        oversample_silence_jitter=None,
        n_input=args.model_input_size,
        grouped_samples=args.grouped_samples,
        path_norm_global_channel_zscore=args.path_norm_global_channel_zscore
    )
    
    num_classes = len(phoneme_to_idx)
    print(f"[INFO] Number of phoneme classes: {num_classes}")
    
    # Save phoneme mapping for later use
    phoneme_mapping_path = os.path.join(base_dir, "phoneme_to_idx.pt")
    torch.save(phoneme_to_idx, phoneme_mapping_path)
    print(f"[INFO] Saved phoneme mapping to {phoneme_mapping_path}")

    model = BrainPhonemeClassifier(
        input_dim=args.model_input_size,
        model_dim=args.model_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        lstm_layers=args.lstm_layers,
        batch_norm=args.batch_norm,
        bi_directional=args.bi_directional,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        phoneme_to_idx=phoneme_to_idx,  # Pass phoneme mapping for adaptive weights
    )

    monitor_metric = "val_loss" if args.monitor == "val_loss" else "val_f1_macro"
    monitor_mode = "min" if args.monitor == "val_loss" else "max"

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        verbose=True,
        strict=True,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger_list,
        callbacks=[early_stopping_callback],
        default_root_dir=base_dir,
        log_every_n_steps=max(1, len(train_loader) // 10),
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)


def main(args):
    seed_everything(42)

    # Ensure checkpoint directory exists
    os.makedirs(args.ckpt_path, exist_ok=True)

    timestamp_file = os.path.join(
        args.ckpt_path, f"Model_task2_timestamp.txt")
    if not dist.is_initialized() or dist.get_rank() == 0:
        args.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(timestamp_file, 'w') as f:
            f.write(args.timestamp)

    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized() and dist.get_rank() != 0:
        with open(timestamp_file, 'r') as f:
            args.timestamp = f.read().strip()

    run_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phoneme classification model (Task 2)")
    parser.add_argument("--data_path", type=str, default="./assets/data_task2")
    parser.add_argument("--ckpt_path", type=str, default="./output_task2")
    parser.add_argument("--model_input_size", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--bi_directional", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--monitor", type=str, default="val_f1_macro")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for cross-entropy loss (default: 0.1)")
    parser.add_argument("--grouped_samples", type=int, default=50,
                        help="Number of samples to group and average for signal averaging (default: 50)")

    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of epochs to wait before early stopping (default: 10)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001,
                        help="Minimum change in monitored quantity to qualify as improvement (default: 0.001)")
    
    # Normalization arguments
    parser.add_argument("--path_norm_global_channel_zscore", type=str, default="assets/norm_task2/time",
                        help="Path to global normalization statistics (default: assets/norm_task2/time)")

    args = parser.parse_args()
    main(args)
