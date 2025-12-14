import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def read_model_metrics(output_dir, timestamp):
    """Read metrics from the trained model"""
    model_dir_name = f"model_{timestamp}"
    model_path = os.path.join(output_dir, model_dir_name)
    metrics_path = os.path.join(model_path, 'version_1', 'metrics.csv')
    
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return None
    
    df = pd.read_csv(metrics_path)
    
    # Separate train and validation metrics
    train_metrics = df[df['train_f1_macro'].notna()][['epoch', 'train_f1_macro', 'train_loss_epoch']].copy()
    val_metrics = df[df['val_f1_macro'].notna()][['epoch', 'val_f1_macro', 'val_loss']].copy()
    
    # Merge them on epoch
    epoch_metrics = pd.merge(train_metrics, val_metrics, on='epoch', how='inner')
    
    return epoch_metrics


def plot_training_curves(df, save_path=None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation F1 Score
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['val_f1_macro'], 
            marker='o', label='Validation', color='#2E86AB', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Validation F1 Score', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training F1 Score
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_f1_macro'], 
            marker='s', label='Training', color='#A23B72', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Training F1 Score', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Loss
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['val_loss'], 
            marker='o', label='Validation', color='#2E86AB', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Loss
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['train_loss_epoch'], 
            marker='s', label='Training', color='#A23B72', linewidth=2)
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


def plot_combined_metrics(df, save_path=None):
    """Plot combined training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # Debug: print data to verify
    print(f"\nData verification:")
    print(f"Epochs: {list(epochs)}")
    print(f"Val F1: {list(df['val_f1_macro'].round(4))}")
    print(f"Train F1: {list(df['train_f1_macro'].round(4))}")
    
    # Plot 1: F1 Score - Training vs Validation
    ax = axes[0]
    ax.plot(epochs, df['val_f1_macro'], marker='o', label='Validation', 
            color='#2E86AB', linewidth=2, markersize=6)
    ax.plot(epochs, df['train_f1_macro'], marker='s', label='Training', 
            color='#A23B72', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('F1 Score Over Training', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss - Training vs Validation
    ax = axes[1]
    ax.plot(epochs, df['val_loss'], marker='o', label='Validation', 
            color='#2E86AB', linewidth=2, markersize=6)
    ax.plot(epochs, df['train_loss_epoch'], marker='s', label='Training', 
            color='#A23B72', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Over Training', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot training results from model')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory containing model results')
    parser.add_argument('--timestamp', type=str,
                        help='Timestamp of the training run (e.g., 2025-12-14_10-51-22)', default='2025-12-14_14-56-14')
    parser.add_argument('--save_dir', type=str, default='./plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Read model metrics
    print("Reading metrics from model...")
    df = read_model_metrics(args.output_dir, args.timestamp)
    
    if df is None or df.empty:
        print("Error: No metrics data found!")
        print(f"Expected path: {args.output_dir}/model_{args.timestamp}/version_1/metrics.csv")
        return
    
    print(f"Loaded data from {len(df)} epochs")
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"  Best Validation F1: {df['val_f1_macro'].max():.4f} (Epoch {df.loc[df['val_f1_macro'].idxmax(), 'epoch']:.0f})")
    print(f"  Final Validation F1: {df['val_f1_macro'].iloc[-1]:.4f}")
    print(f"  Best Validation Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']:.0f})")
    print(f"  Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    
    # Plot 1: Individual training curves (4 subplots)
    print("\nGenerating training curves plot...")
    plot_training_curves(df, save_path=os.path.join(args.save_dir, 'training_curves.png'))
    
    # Plot 2: Combined metrics (train vs val comparison)
    print("Generating combined metrics plot...")
    plot_combined_metrics(df, save_path=os.path.join(args.save_dir, 'combined_metrics.png'))
    
    print("\n✓ All plots generated successfully!")
    print(f"Plots saved to: {args.save_dir}/")
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()