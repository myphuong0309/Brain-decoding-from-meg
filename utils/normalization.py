import numpy as np
import torch
from utils.compute_global_stats import load_global_statistics

import numpy as np
import torch
from utils.compute_global_stats import load_global_statistics

class SensorNormalizer:
    def __init__(self, sensor_filter='all',
                 filtered_indices=None, path_norm_global_channel_zscore="assets/norm/time"):
        self.sensor_filter = sensor_filter
        self.filtered_indices = filtered_indices
        self.path_norm_global_channel_zscore = path_norm_global_channel_zscore
        self.global_mean = None
        self.global_std = None
        
        try:
            stats = load_global_statistics(
                cache_dir=self.path_norm_global_channel_zscore)
            self.global_mean = stats['mean']
            self.global_std = stats['std']
            print(
                f"[INFO] Loaded global statistics from {self.path_norm_global_channel_zscore}")
        except Exception as e:
            print(f"[ERROR] Failed to load global statistics: {e}")
            raise

    def transform(self, raw_data):
        if torch.is_tensor(raw_data):
            raw_data = raw_data.cpu().numpy()
            
        if self.global_mean is not None and self.global_std is not None:
            filtered_mean = self.global_mean[self.filtered_indices]
            filtered_std = self.global_std[self.filtered_indices]
            mean_reshaped = filtered_mean.reshape(-1, 1)
            std_reshaped = filtered_std.reshape(-1, 1)
            return (raw_data - mean_reshaped) / std_reshaped
        else:
            raise ValueError(
                "Global statistics not available for 'global' mode")

    def transform_batch(self, batch_data):
        if self.global_mean is not None and self.global_std is not None:
            filtered_mean = self.global_mean[self.filtered_indices]
            filtered_std = self.global_std[self.filtered_indices]
            
            mean_reshaped = filtered_mean.reshape(1, -1, 1)
            std_reshaped = filtered_std.reshape(1, -1, 1)
            
            return (batch_data - mean_reshaped) / std_reshaped
        else:
            raise ValueError(
                "Global statistics not available for 'global' mode")