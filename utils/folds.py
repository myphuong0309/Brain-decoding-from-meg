import os
import torch
from pnpl.datasets import LibriBrainSpeech

FULL_RUN_KEYS = []
for j, max_sess in zip(range(1, 3), [11, 11]):
    for i in range(1, max_sess + 1):
        run = "2" if j == 1 and i in [11, 12] else "1"
        FULL_RUN_KEYS.append(("0", str(i), f"Sherlock{j}", run))


def get_fold_keys_train_val(n_splits=5):
    total = len(FULL_RUN_KEYS)
    fold_size = total // n_splits
    all_splits = []
    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        train_keys = []
        for i, key in enumerate(FULL_RUN_KEYS):
            if i < val_start or i >= val_end:
                train_keys.append(key)
        val_keys = FULL_RUN_KEYS[val_start:val_end]
        all_splits.append((fold, train_keys, val_keys))
    return all_splits


def cache_split(data_path, fold, name, run_keys, standardize):
    cache_dir = os.path.join(data_path, "cached_cv10")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_path = os.path.join(cache_dir, f"{name}_fold{fold}.pt")
    if os.path.exists(cache_path):
        print(f"{cache_path} already exists. Skipping.")
        return

    print(f"Creating {cache_path} ...")
    ds = LibriBrainSpeech(
        data_path=data_path,
        include_run_keys=run_keys,
        tmin=0.0,
        tmax=0.1,
        preload_files=True,
        standardize=standardize
    )
    torch.save(ds, cache_path)
    print(f"Saved {cache_path}")


def precache_train_val_only(data_path, n_splits=5):
    print(
        f"[INFO] Precaching train and validation datasets for {n_splits} folds...")
    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] Total samples: {len(FULL_RUN_KEYS)}")

    splits = get_fold_keys_train_val(n_splits)
    for fold, train_keys, val_keys in splits:
        print(f"\n[INFO] Processing fold {fold}:")
        print(
            f"  - Train samples: {len(train_keys)} ({len(train_keys)/len(FULL_RUN_KEYS)*100:.1f}%)")
        print(
            f"  - Val samples: {len(val_keys)} ({len(val_keys)/len(FULL_RUN_KEYS)*100:.1f}%)")
        cache_split(data_path, fold, "train", train_keys, standardize=False)
        cache_split(data_path, fold, "val", val_keys, standardize=True)
    print(f"\n[INFO] Precaching completed for {n_splits} folds!")
    print(f"[INFO] Cached files saved to: {os.path.join(data_path, 'cached')}")
    print(
        f"[INFO] Each fold uses {(n_splits-1)/n_splits*100:.1f}% for training and {1/n_splits*100:.1f}% for validation")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory (contains .fif files)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds (default: 5)")
    args = parser.parse_args()
    precache_train_val_only(args.data_path, args.n_splits)
