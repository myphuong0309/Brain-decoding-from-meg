import os
import torch
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm
from torch.serialization import add_safe_globals

def compute_global_statistics(data_path, cache_dir=None, max_samples=None):
    if cache_dir is None:
        cache_dir = "assets/norm/time"
    os.makedirs(cache_dir, exist_ok=True)

    stats_file = os.path.join(cache_dir, "global_stats.pt")

    if os.path.exists(stats_file):
        print(f"[INFO] Loading existing global statistics from {stats_file}")
        try:
            return torch.load(stats_file, weights_only=False)
        except Exception as e:
            print(f"[WARNING] Failed to load with weights_only=False: {e}")
            add_safe_globals([np.ndarray, np.dtype])
            return torch.load(stats_file, weights_only=True)

    print("[INFO] Computing per-channel global statistics from raw .h5 files...")

    n_channels = 306
    sum_values = np.zeros(n_channels)
    sum_squares = np.zeros(n_channels)
    total_timepoints = 0
    processed_files = 0

    h5_files = []

    holdout_dir = os.path.join(
        data_path, "COMPETITION_HOLDOUT/derivatives/serialised")
    if os.path.exists(holdout_dir):
        for root, dirs, files in os.walk(holdout_dir):
            dirs[:] = [d for d in dirs if d != '.cache']
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))

    for sherlock_num in range(1, 3):
        sherlock_dir = os.path.join(
            data_path, f"Sherlock{sherlock_num}/derivatives/serialised")
        if os.path.exists(sherlock_dir):
            for root, dirs, files in os.walk(sherlock_dir):
                dirs[:] = [d for d in dirs if d != '.cache']
                for file in files:
                    if file.endswith('.h5'):
                        h5_files.append(os.path.join(root, file))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_path}")

    print(f"[INFO] Found {len(h5_files)} .h5 files")

    for h5_file in h5_files:
        print(f"[INFO] Processing {os.path.basename(h5_file)}...")
        try:
            with h5py.File(h5_file, 'r') as f:
                meg_data = None
                if 'data' in f:
                    meg_data = f['data'][:]
                elif 'meg' in f:
                    meg_data = f['meg'][:]  # Alternative key
                else:
                    print(f"[WARNING] No MEG data found in {h5_file}")
                    continue
                if not isinstance(meg_data, np.ndarray):
                    meg_data = np.array(meg_data)
                if meg_data.shape[0] != n_channels:
                    print(
                        f"[WARNING] Skipping {h5_file}: expected {n_channels} channels, got {meg_data.shape[0]}")
                    continue
                print(f"[INFO] Loaded data with shape {meg_data.shape}")
                if max_samples and processed_files >= max_samples:
                    break
                sum_values += np.sum(meg_data, axis=1)
                sum_squares += np.sum(meg_data ** 2, axis=1)
                total_timepoints += meg_data.shape[1]
                processed_files += 1
        except Exception as e:
            print(f"[WARNING] Error processing {h5_file}: {e}")
            continue

    if total_timepoints == 0:
        raise ValueError("No valid data found for computing statistics")

    global_mean = sum_values / total_timepoints  # Shape: (n_channels,)
    global_var = (sum_squares / total_timepoints) - (global_mean ** 2)
    global_std = np.sqrt(global_var)  # Shape: (n_channels,)

    global_std[global_std == 0] = 1e-12

    stats = {
        'mean': global_mean.astype(np.float32),  # Per-channel means
        'std': global_std.astype(np.float32),    # Per-channel stds
        'total_timepoints': total_timepoints,
        'n_channels': n_channels,
        'processed_files': processed_files,
        'data_path': data_path,
        'computed_at': str(datetime.now())
    }

    torch.save(stats, stats_file)
    print(f"[INFO] Global statistics saved to {stats_file}")
    print(
        f"[INFO] Per-channel means - min: {global_mean.min():.6e}, max: {global_mean.max():.6e}")
    print(
        f"[INFO] Per-channel stds - min: {global_std.min():.6e}, max: {global_std.max():.6e}")
    print(f"[INFO] Total timepoints processed: {total_timepoints}")
    print(f"[INFO] Number of channels: {n_channels}")
    print(f"[INFO] Number of files processed: {processed_files}")
    return stats


def load_global_statistics(cache_dir=None):
    """
    Load pre-computed global statistics.

    Args:
        cache_dir: Directory containing statistics (default: assets/norm/time)

    Returns:
        dict: Dictionary containing mean, std, and metadata
    """
    if cache_dir is None:
        cache_dir = "assets/norm/time"

    stats_file = os.path.join(cache_dir, "global_stats.pt")

    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"Global statistics file not found: {stats_file}")

    try:
        # Try loading with weights_only=False first (for backward compatibility)
        return torch.load(stats_file, weights_only=False)
    except Exception as e:
        print(f"[WARNING] Failed to load with weights_only=False: {e}")
        # If that fails, try with safe globals for numpy arrays
        add_safe_globals([np.ndarray, np.dtype])
        return torch.load(stats_file, weights_only=True)


class GlobalNormalizer:
    """
    A class to handle global normalization using pre-computed statistics.
    """

    def __init__(self, data_path, cache_dir=None, stats=None):
        """
        Initialize the global normalizer.

        Args:
            data_path: Path to the data directory
            cache_dir: Directory containing statistics
            stats: Pre-computed statistics dict (if None, will load from cache)
        """
        self.data_path = data_path

        if stats is None:
            self.stats = load_global_statistics(cache_dir)
        else:
            self.stats = stats

        self.mean = self.stats['mean']  # Shape: (n_channels,)
        self.std = self.stats['std']    # Shape: (n_channels,)

    def normalize(self, data):
        """
        Normalize data using per-channel global mean and std.

        Args:
            data: numpy array or torch tensor of shape (n_channels, n_timepoints)

        Returns:
            normalized_data: normalized data with same shape
        """
        # Reshape mean and std for broadcasting
        mean_reshaped = self.mean.reshape(-1, 1)  # (n_channels, 1)
        std_reshaped = self.std.reshape(-1, 1)    # (n_channels, 1)

        if torch.is_tensor(data):
            return (data - mean_reshaped) / std_reshaped
        else:
            return (data - mean_reshaped) / std_reshaped

    def normalize_batch(self, batch_data):
        """
        Normalize a batch of data using per-channel global mean and std.

        Args:
            batch_data: torch tensor of shape (batch_size, n_channels, n_timepoints)

        Returns:
            normalized_batch: normalized batch with same shape
        """
        # Reshape mean and std for broadcasting
        mean_reshaped = self.mean.reshape(1, -1, 1)  # (1, n_channels, 1)
        std_reshaped = self.std.reshape(1, -1, 1)    # (1, n_channels, 1)

        return (batch_data - mean_reshaped) / std_reshaped


if __name__ == "__main__":
    data_path = "assets/data"
    stats = compute_global_statistics(data_path, max_samples=10000)
    # print("Computed statistics:", stats)
    print(f"Statistics saved to: assets/norm/time/global_stats.pt")