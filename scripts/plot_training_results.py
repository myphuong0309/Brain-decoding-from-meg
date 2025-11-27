import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def read_all_fold_metrics(output_dir, timestamp, n_folds=5):
    """Read metrics from all folds"""
    all_fold_data = []
    
    for fold_num in range(n_folds):
        fold_dir_name = f"fold_{fold_num}_{timestamp}"
        fold_path = os.path.join(output_dir, fold_dir_name)
        metrics_path = os.path.join(fold_path, 'version_1', 'metrics.csv')
        
        if not os.path.exists(metrics_path):
            print(f"Warning: {metrics_path} not found")
            continue
        
        df = pd.read_csv(metrics_path)
        
        # Separate train and validation metrics
        train_metrics = df[df['train_f1_macro'].notna()][['epoch', 'train_f1_macro', 'train_loss_epoch']].copy()
        val_metrics = df[df['val_f1_macro'].notna()][['epoch', 'val_f1_macro', 'val_loss']].copy()
        
        # Merge them on epoch
        epoch_metrics = pd.merge(train_metrics, val_metrics, on='epoch', how='inner')
        epoch_metrics['fold'] = fold_num
        
        all_fold_data.append(epoch_metrics)
    
    return pd.concat(all_fold_data, ignore_index=True)


def plot_training_curves(df, save_path=None):
    """Plot training and validation curves across all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('5-Fold Cross-Validation Training Curves', fontsize=16, fontweight='bold')
    
    folds = df['fold'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
    
    # Plot 1: Validation F1 Score
    ax = axes[0, 0]
    for i, fold in enumerate(folds):
        fold_data = df[df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['val_f1_macro'], 
                marker='o', label=f'Fold {fold}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Validation F1 Score', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training F1 Score
    ax = axes[0, 1]
    for i, fold in enumerate(folds):
        fold_data = df[df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['train_f1_macro'], 
                marker='s', label=f'Fold {fold}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Training F1 Score', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Loss
    ax = axes[1, 0]
    for i, fold in enumerate(folds):
        fold_data = df[df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['val_loss'], 
                marker='o', label=f'Fold {fold}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Loss
    ax = axes[1, 1]
    for i, fold in enumerate(folds):
        fold_data = df[df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['train_loss_epoch'], 
                marker='s', label=f'Fold {fold}', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    
    return fig


def plot_aggregate_curves(df, save_path=None):
    """Plot mean and std of metrics across folds"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Aggregated Cross-Validation Results (Mean ± Std)', fontsize=16, fontweight='bold')
    
    # Calculate mean and std across folds for each epoch
    agg_data = df.groupby('epoch').agg({
        'val_f1_macro': ['mean', 'std'],
        'train_f1_macro': ['mean', 'std'],
        'val_loss': ['mean', 'std'],
        'train_loss_epoch': ['mean', 'std']
    }).reset_index()
    
    epochs = agg_data['epoch']
    
    # Plot 1: F1 Score
    ax = axes[0]
    
    # Validation F1
    val_f1_mean = agg_data[('val_f1_macro', 'mean')]
    val_f1_std = agg_data[('val_f1_macro', 'std')]
    ax.plot(epochs, val_f1_mean, marker='o', label='Validation', color='#2E86AB', linewidth=2)
    ax.fill_between(epochs, val_f1_mean - val_f1_std, val_f1_mean + val_f1_std, 
                     alpha=0.2, color='#2E86AB')
    
    # Training F1
    train_f1_mean = agg_data[('train_f1_macro', 'mean')]
    train_f1_std = agg_data[('train_f1_macro', 'std')]
    ax.plot(epochs, train_f1_mean, marker='s', label='Training', color='#A23B72', linewidth=2)
    ax.fill_between(epochs, train_f1_mean - train_f1_std, train_f1_mean + train_f1_std, 
                     alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('F1 Score Over Training', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax = axes[1]
    
    # Validation Loss
    val_loss_mean = agg_data[('val_loss', 'mean')]
    val_loss_std = agg_data[('val_loss', 'std')]
    ax.plot(epochs, val_loss_mean, marker='o', label='Validation', color='#2E86AB', linewidth=2)
    ax.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, 
                     alpha=0.2, color='#2E86AB')
    
    # Training Loss
    train_loss_mean = agg_data[('train_loss_epoch', 'mean')]
    train_loss_std = agg_data[('train_loss_epoch', 'std')]
    ax.plot(epochs, train_loss_mean, marker='s', label='Training', color='#A23B72', linewidth=2)
    ax.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, 
                     alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Over Training', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated curves to: {save_path}")
    
    return fig


def plot_final_comparison(csv_path, save_path=None):
    """Plot final comparison of best results across folds"""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Best Performance Comparison Across Folds', fontsize=16, fontweight='bold')
    
    folds = df['fold'].values
    x = np.arange(len(folds))
    width = 0.35
    
    # Plot 1: F1 Scores
    ax = axes[0]
    bars1 = ax.bar(x - width/2, df['best_val_f1'], width, label='Validation', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['best_train_f1'], width, label='Training', 
                   color='#A23B72', alpha=0.8)
    
    # Add mean line
    ax.axhline(y=df['best_val_f1'].mean(), color='#2E86AB', linestyle='--', 
               linewidth=2, label=f'Val Mean: {df["best_val_f1"].mean():.4f}')
    ax.axhline(y=df['best_train_f1'].mean(), color='#A23B72', linestyle='--', 
               linewidth=2, label=f'Train Mean: {df["best_train_f1"].mean():.4f}')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Best F1 Scores per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Loss
    ax = axes[1]
    bars1 = ax.bar(x - width/2, df['best_val_loss'], width, label='Validation', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['best_train_loss'], width, label='Training', 
                   color='#A23B72', alpha=0.8)
    
    # Add mean line
    ax.axhline(y=df['best_val_loss'].mean(), color='#2E86AB', linestyle='--', 
               linewidth=2, label=f'Val Mean: {df["best_val_loss"].mean():.4f}')
    ax.axhline(y=df['best_train_loss'].mean(), color='#A23B72', linestyle='--', 
               linewidth=2, label=f'Train Mean: {df["best_train_loss"].mean():.4f}')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Best Loss per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot training results from all folds')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory containing fold results')
    parser.add_argument('--timestamp', type=str, default='2025-11-15_18-46-12',
                        help='Timestamp of the training run')
    parser.add_argument('--csv_path', type=str, default='./training_results.csv',
                        help='Path to training results CSV file')
    parser.add_argument('--save_dir', type=str, default='./plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Read all fold metrics
    print("Reading metrics from all folds...")
    df = read_all_fold_metrics(args.output_dir, args.timestamp)
    
    if df.empty:
        print("Error: No metrics data found!")
        return
    
    print(f"Loaded data from {len(df['fold'].unique())} folds")
    
    # Plot 1: Individual fold curves
    print("\nGenerating training curves plot...")
    plot_training_curves(df, save_path=os.path.join(args.save_dir, 'training_curves.png'))
    
    # Plot 2: Aggregated curves with mean and std
    print("Generating aggregated curves plot...")
    plot_aggregate_curves(df, save_path=os.path.join(args.save_dir, 'aggregated_curves.png'))
    
    # Plot 3: Final comparison
    print("Generating comparison plot...")
    plot_final_comparison(args.csv_path, save_path=os.path.join(args.save_dir, 'fold_comparison.png'))
    
    print("\n✓ All plots generated successfully!")
    print(f"Plots saved to: {args.save_dir}/")
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()