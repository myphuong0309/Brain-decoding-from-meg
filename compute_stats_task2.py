#!/usr/bin/env python3
"""
Compute global normalization statistics for Task 2 (Phoneme Classification)
This script computes per-channel mean and std from the task2 dataset
and saves them for use during training and evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute_global_stats import compute_global_statistics

if __name__ == "__main__":
    data_path = "assets/data_task2"
    cache_dir = "assets/norm_task2/time"
    
    print("="*60)
    print("Computing Global Statistics for Task 2 (Phoneme Classification)")
    print("="*60)
    print(f"Data path: {data_path}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    stats = compute_global_statistics(
        data_path=data_path,
        cache_dir=cache_dir,
        max_samples=None
    )
    
    print("\n" + "="*60)
    print("Statistics Summary:")
    print("="*60)
    print(f"Mean shape: {stats['mean'].shape}")
    print(f"Std shape: {stats['std'].shape}")
    print(f"Total timepoints: {stats['total_timepoints']}")
    print(f"Saved to: {cache_dir}/global_stats.pt")
    print("="*60)
