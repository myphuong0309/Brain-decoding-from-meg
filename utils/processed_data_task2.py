import torch
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
import random
from torch.utils.data import Dataset, DataLoader
import os
from utils.normalization import SensorNormalizer
import pandas as pd

SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145, 146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]

ALL_SENSORS = list(range(306))

# Define train keys: Sherlock1 sessions 1-10 + Sherlock2 all sessions
TRAIN_RUN_KEYS = []
for i in range(1, 11):  # Sherlock1 sessions 1-10
    run = "1"
    TRAIN_RUN_KEYS.append(("0", str(i), "Sherlock1", run))
for i in range(1, 13):  # Sherlock2 sessions 1-12 (all run-1)
    run = "1"
    TRAIN_RUN_KEYS.append(("0", str(i), "Sherlock2", run))

# Define val keys: Sherlock1 session 11
VAL_RUN_KEYS = [
    ("0", "11", "Sherlock1", "2")
]

# Define test keys: Sherlock2 (simulating that only task2 has different test set)
TEST_RUN_KEYS = [
    ("0", "12", "Sherlock1", "2")
]


def extract_phoneme_labels_from_events(data_path):
    """
    Extract all unique BASE phoneme labels (without position markers) from the event files.
    Returns a sorted list of phoneme labels and a mapping from label to index.
    
    Example: 'aa_B', 'aa_I', 'aa_E' -> all map to 'aa'
    """
    phoneme_set = set()
    
    # Scan through all event files in the task2 dataset
    for sherlock_num in [1, 2]:
        events_dir = os.path.join(data_path, f"Sherlock{sherlock_num}/derivatives/events")
        if not os.path.exists(events_dir):
            continue
            
        for event_file in os.listdir(events_dir):
            if event_file.endswith("_events.tsv"):
                event_path = os.path.join(events_dir, event_file)
                try:
                    df = pd.read_csv(event_path, sep='\t')
                    # Filter only phoneme rows
                    phoneme_rows = df[df['kind'] == 'phoneme']
                    phonemes = phoneme_rows['segment'].unique()
                    
                    # Extract base phoneme (remove position markers like _B, _I, _E, _S)
                    for phoneme in phonemes:
                        # Split by underscore and take the base phoneme
                        base_phoneme = phoneme.split('_')[0] if '_' in phoneme else phoneme
                        phoneme_set.add(base_phoneme)
                except Exception as e:
                    print(f"[WARNING] Error reading {event_file}: {e}")
                    continue
    
    # Sort phonemes for consistent ordering
    phoneme_list = sorted(list(phoneme_set))
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}
    
    print(f"[INFO] Found {len(phoneme_list)} unique base phonemes (without position markers)")
    return phoneme_list, phoneme_to_idx


class FilteredDatasetTask2(Dataset):
    """
    Wrapper for LibriBrainPhoneme dataset that:
    1. Applies sensor masking
    2. Converts phoneme labels to indices using custom mapping
    """
    def __init__(self, dataset, phoneme_to_idx, n_input=306, is_grouped=False):
        self.dataset = dataset
        self.phoneme_to_idx = phoneme_to_idx
        self.num_classes = len(phoneme_to_idx)
        self.is_grouped = is_grouped
        
        if n_input == 306:
            self.sensors_speech_mask = ALL_SENSORS
        elif n_input == 23:
            self.sensors_speech_mask = SENSORS_SPEECH_MASK
        else:
            raise ValueError(f"n_input must be 23 or 306, got {n_input}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get data from underlying dataset
        data_item = self.dataset[idx]
        sensors = data_item[0]  # Shape: (306, T)
        
        # For grouped dataset, label is already an integer index
        # For non-grouped dataset, we need to get phoneme string from samples
        if self.is_grouped:
            # GroupedDataset returns integer label directly
            label = data_item[1].item() if hasattr(data_item[1], 'item') else data_item[1]
        else:
            # Regular LibriBrainPhoneme: get phoneme string from samples
            phoneme_label = self.dataset.samples[idx][-1]
            
            # Extract base phoneme (remove position marker)
            base_phoneme = phoneme_label.split('_')[0] if '_' in phoneme_label else phoneme_label
            
            # Convert base phoneme to our custom class index
            if base_phoneme in self.phoneme_to_idx:
                label = self.phoneme_to_idx[base_phoneme]
            else:
                print(f"[WARNING] Unknown base phoneme: {base_phoneme} (from {phoneme_label}), using class 0")
                label = 0
        
        # Apply sensor masking
        sensors = sensors[self.sensors_speech_mask]
            
        return sensors, label
    

class NormalizedDatasetTask2(Dataset):
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


def get_dataloaders_task2(data_path, num_workers=4,
                       train_batch_size=32, eval_batch_size=32,
                       oversample_silence_jitter=None, n_input=306,
                       grouped_samples=50,
                       path_norm_global_channel_zscore="assets/norm_task2/time"):
    """
    Get train and val dataloaders for phoneme classification task:
    - Train: Sherlock1 sessions 1-10 (with signal averaging)
    - Val: Sherlock1 session 11
    
    Args:
        grouped_samples: Number of phoneme instances to average together (default: 50)
                        Competition uses 100, but 50 is good for training with less data
    """
    # Extract phoneme labels
    phoneme_list, phoneme_to_idx = extract_phoneme_labels_from_events(data_path)
    
    if n_input == 306:
        sensors_speech_mask = ALL_SENSORS
    elif n_input == 23:
        sensors_speech_mask = SENSORS_SPEECH_MASK

    dataset_kwargs = {
        "data_path": data_path,
        "tmin": 0.0,
        "tmax": 0.5,
        "preload_files": True,
        "standardize": False,
    }
    
    if oversample_silence_jitter is not None:
        dataset_kwargs["oversample_silence_jitter"] = oversample_silence_jitter
    
    print(f"[INFO] Loading training data (Sherlock1 sessions 1-10)...")
    train_data = LibriBrainPhoneme(
        include_run_keys=TRAIN_RUN_KEYS,
        **dataset_kwargs
    )
    
    print(f"[INFO] Original training samples: {len(train_data)}")
    
    # Apply signal averaging to improve SNR
    is_grouped = False
    if grouped_samples > 1:
        print(f"[INFO] Grouping {grouped_samples} samples per phoneme for training...")
        train_data = GroupedDataset(
            train_data, 
            grouped_samples=grouped_samples,
            average_grouped_samples=True,
            drop_remaining=False,
            shuffle=True
        )
        is_grouped = True
        print(f"[INFO] Training samples after grouping: {len(train_data)}")
    
    print(f"[INFO] Loading validation data (Sherlock1 session 11)...")
    val_data = LibriBrainPhoneme(
        include_run_keys=VAL_RUN_KEYS,
        **{k: v for k, v in dataset_kwargs.items() if k != "oversample_silence_jitter"}
    )
    
    train_filtered = FilteredDatasetTask2(train_data, phoneme_to_idx, n_input=n_input, is_grouped=is_grouped)
    val_filtered = FilteredDatasetTask2(val_data, phoneme_to_idx, n_input=n_input, is_grouped=False)
    
    normalizer = SensorNormalizer(
        filtered_indices=sensors_speech_mask,
        path_norm_global_channel_zscore=path_norm_global_channel_zscore
    )
    train = NormalizedDatasetTask2(train_filtered, normalizer)
    val = NormalizedDatasetTask2(val_filtered, normalizer)
    
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, phoneme_to_idx


def get_test_dataloader_task2(data_path, phoneme_to_idx, num_workers=4, eval_batch_size=32, 
                        n_input=306, grouped_samples=100,
                        path_norm_global_channel_zscore="assets/norm_task2/time"):
    """
    Get test dataloader for Sherlock2 sessions with signal averaging
    """
    if n_input == 306:
        sensors_speech_mask = ALL_SENSORS
    elif n_input == 23:
        sensors_speech_mask = SENSORS_SPEECH_MASK
    
    dataset_kwargs = {
        "data_path": data_path,
        "tmin": 0.0,
        "tmax": 0.5,
        "preload_files": True,
        "standardize": False,
    }
    
    print(f"[INFO] Loading test data (Sherlock2 sessions)...")
    test_data = LibriBrainPhoneme(
        include_run_keys=TEST_RUN_KEYS,
        **dataset_kwargs
    )
    
    print(f"[INFO] Original test samples: {len(test_data)}")
    
    # Apply signal averaging for testing
    is_grouped = False
    if grouped_samples > 1:
        print(f"[INFO] Grouping {grouped_samples} samples per phoneme for testing...")
        test_data = GroupedDataset(
            test_data, 
            grouped_samples=grouped_samples,
            average_grouped_samples=True,
            drop_remaining=False,
            shuffle=False
        )
        is_grouped = True
        print(f"[INFO] Test samples after grouping: {len(test_data)}")
    
    test_filtered = FilteredDatasetTask2(test_data, phoneme_to_idx, n_input=n_input, is_grouped=is_grouped)
    
    normalizer = SensorNormalizer(
        filtered_indices=sensors_speech_mask,
        path_norm_global_channel_zscore=path_norm_global_channel_zscore
    )
    test = NormalizedDatasetTask2(test_filtered, normalizer)
    
    test_loader = DataLoader(test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
