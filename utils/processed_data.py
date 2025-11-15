import torch
from pnpl.datasets import LibriBrainSpeech
import random
from torch.utils.data import Dataset, DataLoader
import os
from utils.normalization import SensorNormalizer

SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145, 146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]

ALL_SENSORS = list(range(306))

# Define train/val keys: Sherlock1 sessions 1-10, Sherlock2 sessions 1-12
TRAIN_VAL_RUN_KEYS = []
for i in range(1, 11):  # Sherlock1 sessions 1-10
    run = "1"
    TRAIN_VAL_RUN_KEYS.append(("0", str(i), "Sherlock1", run))
for i in range(1, 13):  # Sherlock2 sessions 1-12 (all run-1)
    run = "1"
    TRAIN_VAL_RUN_KEYS.append(("0", str(i), "Sherlock2", run))

# Define test keys: Sherlock1 sessions 11-12
TEST_RUN_KEYS = [
    ("0", "11", "Sherlock1", "2"),
    ("0", "12", "Sherlock1", "2")
]
        
class FilteredDataset(Dataset):
    def __init__(self, dataset, limit_samples=None, apply_sensors_speech_mask=True,
                 n_input=306):
        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask
        if n_input == 306:
            self.sensors_speech_mask = ALL_SENSORS
        elif n_input == 23:
            self.sensors_speech_mask = SENSORS_SPEECH_MASK
        else:
            raise ValueError(f"n_input must be 23 or 306, got {n_input}")
        self.balanced_indices = random.sample(range(len(dataset.samples)), len(dataset.samples))
        
    def __len__(self):
        return self.limit_samples if self.limit_samples else len(self.balanced_indices)
    
    def __getitem__(self, idx):
        original_idx = self.balanced_indices[idx]
        if self.apply_sensors_speech_mask:
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[original_idx][0][:]
        label_idx = self.dataset[original_idx][1].shape[0] // 2
        label = self.dataset[original_idx][1][label_idx]
        return sensors, label
    
class NormalizedDataset(Dataset):
    def __init__(self, dataset, normalizer):
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sensors, label = self.dataset[idx]
        if self.normalizer is not None:
            sensors = self.normalizer.transform(sensors)
            if not torch.is_tensor(sensors):
                sensors = torch.from_numpy(sensors).float()
        return sensors, label

def get_dataloaders(data_path, num_workers=4, fold=0, n_splits=5,
                       train_batch_size=32, eval_batch_size=32,
                       use_fold_specific_cache=False,
                       oversample_silence_jitter=None, n_cv=10, n_input=306,
                       path_norm_global_channel_zscore="assets/norm/time"):

    if n_input == 306:
        sensors_speech_mask = ALL_SENSORS
    elif n_input == 23:
        sensors_speech_mask = SENSORS_SPEECH_MASK

    def load_cached_dataset(name):
        fold_cache = os.path.join(
            data_path, f"cached_cv{n_cv}", f"{name}_fold{fold}.pt")
        if os.path.exists(fold_cache):
            print(f"[INFO] Loading cached {name} dataset for fold {fold}...")
            return torch.load(fold_cache, weights_only=False)
        else:
            raise FileNotFoundError(
                f"[ERROR] Cached file not found: {fold_cache}")

    # Use TRAIN_VAL_RUN_KEYS for cross-validation split
    full_run_keys = TRAIN_VAL_RUN_KEYS.copy()
    total = len(full_run_keys)
    fold_size = total // n_splits

    val_start = fold * fold_size
    val_end = val_start + fold_size
    val_keys = full_run_keys[val_start:val_end]

    train_keys = [k for i, k in enumerate(full_run_keys) if i not in range(
        val_start, val_end)]

    if use_fold_specific_cache:
        train_data = load_cached_dataset("train")
        val_data = load_cached_dataset("val")
    else:
        dataset_kwargs = {
            "data_path": data_path,
            "tmin": 0.0,
            "tmax": 0.8,
            "preload_files": True,
        }
        if oversample_silence_jitter is not None:
            dataset_kwargs["oversample_silence_jitter"] = oversample_silence_jitter
        train_data = LibriBrainSpeech(
            include_run_keys=train_keys,
            standardize=False,
            **dataset_kwargs
        )
        val_data = LibriBrainSpeech(
            include_run_keys=val_keys,
            standardize=False,
            **{k: v for k, v in dataset_kwargs.items() if k != "oversample_silence_jitter"}
        )
    train_filtered = FilteredDataset(train_data, n_input=n_input)
    val_filtered = FilteredDataset(val_data, n_input=n_input)
    
    normalizer = SensorNormalizer(
        filtered_indices=sensors_speech_mask,
        path_norm_global_channel_zscore=path_norm_global_channel_zscore
    )
    train = NormalizedDataset(train_filtered, normalizer)
    val = NormalizedDataset(val_filtered, normalizer)
    
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_test_dataloader(data_path, num_workers=4, eval_batch_size=32, 
                        n_input=306, path_norm_global_channel_zscore="assets/norm/time"):
    """
    Get test dataloader for Sherlock1 sessions 11-12
    """
    if n_input == 306:
        sensors_speech_mask = ALL_SENSORS
    elif n_input == 23:
        sensors_speech_mask = SENSORS_SPEECH_MASK
    
    dataset_kwargs = {
        "data_path": data_path,
        "tmin": 0.0,
        "tmax": 0.8,
        "preload_files": True,
    }
    
    test_data = LibriBrainSpeech(
        include_run_keys=TEST_RUN_KEYS,
        standardize=False,
        **dataset_kwargs
    )
    
    test_filtered = FilteredDataset(test_data, n_input=n_input)
    
    normalizer = SensorNormalizer(
        filtered_indices=sensors_speech_mask,
        path_norm_global_channel_zscore=path_norm_global_channel_zscore
    )
    test = NormalizedDataset(test_filtered, normalizer)
    
    test_loader = DataLoader(test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return test_loader