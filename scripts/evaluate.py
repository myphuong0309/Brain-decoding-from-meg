import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

# --- The above three lines must stay at the very top for module import to work ---

import argparse
import torch
import numpy as np
from glob import glob
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import json
import matplotlib.pyplot as plt
import seaborn as sns
from models.model import BrainSpeechClassifier
from utils.processed_data import get_test_dataloader


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    model = BrainSpeechClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(model, dataloader, device='cuda'):
    """Run inference and collect predictions"""
    model = model.to(device)
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(y.numpy().flatten())
    
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def compute_metrics(y_true, y_pred, y_logits):
    """Compute comprehensive evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Per-class metrics
    precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC
    y_probs = torch.sigmoid(torch.tensor(y_logits)).numpy()
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    except:
        roc_auc = None
        fpr, tpr, thresholds = None, None, None
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'class_0': {
                'precision': float(precision_class[0]),
                'recall': float(recall_class[0]),
                'f1': float(f1_class[0]),
                'support': int(support_class[0])
            },
            'class_1': {
                'precision': float(precision_class[1]),
                'recall': float(recall_class[1]),
                'f1': float(f1_class[1]),
                'support': int(support_class[1])
            }
        }
    }
    
    return metrics, (fpr, tpr, thresholds)


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Silence', 'Speech'],
                yticklabels=['Silence', 'Speech'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot and save ROC curve"""
    if fpr is None or tpr is None:
        print("ROC curve data not available")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (Binary): {metrics['precision']:.4f}")
    print(f"Recall (Binary):    {metrics['recall']:.4f}")
    print(f"F1 Score (Binary):  {metrics['f1']:.4f}")
    print(f"\nPrecision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"\nROC AUC:            {metrics['roc_auc']:.4f}")
    
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n{class_name} (support: {class_metrics['support']}):")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall:    {class_metrics['recall']:.4f}")
        print(f"  F1 Score:  {class_metrics['f1']:.4f}")
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX")
    print("-"*60)
    cm = np.array(metrics['confusion_matrix'])
    print(f"              Predicted")
    print(f"              Silence  Speech")
    print(f"Actual Silence  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Speech   {cm[1][0]:6d}  {cm[1][1]:6d}")
    print("="*60 + "\n")


def find_best_checkpoint(ckpt_base_path):
    """Find the best checkpoint from the model directory"""
    # Look for checkpoints in model directory
    checkpoint_pattern = os.path.join(ckpt_base_path, "**", "*.ckpt")
    checkpoints = glob(checkpoint_pattern, recursive=True)
    
    if not checkpoints:
        print(f"Warning: No checkpoints found in {ckpt_base_path}")
        return None
    
    # If there are multiple checkpoints, try to find the best one
    # Look for files with "best" in name, or use the last epoch
    best_ckpt = None
    for ckpt in checkpoints:
        if 'best' in os.path.basename(ckpt).lower():
            best_ckpt = ckpt
            break
    
    # If no "best" checkpoint, use the one with highest epoch number
    if not best_ckpt:
        # Sort by epoch number (extract from filename like "epoch=14-step=13275.ckpt")
        def extract_epoch(path):
            import re
            match = re.search(r'epoch=(\d+)', os.path.basename(path))
            return int(match.group(1)) if match else -1
        
        checkpoints.sort(key=extract_epoch, reverse=True)
        best_ckpt = checkpoints[0]
    
    print(f"  Found checkpoint: {os.path.basename(best_ckpt)}")
    return best_ckpt


def main(args):
    """Main evaluation function - single model evaluation on test set"""
    print(f"Starting test set evaluation...")
    print(f"Checkpoint base path: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find best checkpoint
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        print(f"Using specified checkpoint: {checkpoint_path}")
    else:
        print(f"Finding best checkpoint in: {args.ckpt_path}")
        checkpoint_path = find_best_checkpoint(args.ckpt_path)
    
    if not checkpoint_path:
        raise ValueError(f"No checkpoint found in {args.ckpt_path}")
    
    # Get test dataloader
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET (Sherlock1 Session 12)")
    print("="*60)
    
    test_loader = get_test_dataloader(
        data_path=args.data_path,
        num_workers=4,
        eval_batch_size=args.eval_batch_size,
        n_input=args.model_input_size,
        path_norm_global_channel_zscore=args.path_norm_global_channel_zscore
    )
    
    # Run evaluation
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}\n")
    
    print(f"Evaluating model...")
    model = load_checkpoint(checkpoint_path)
    logits, y_true = evaluate_model(model, test_loader, device)
    
    # Convert to predictions
    logits = logits.flatten()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs > 0.5).astype(int)
    
    # Compute metrics
    metrics, (fpr, tpr, thresholds) = compute_metrics(y_true, y_pred, logits)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics['checkpoint_path'] = checkpoint_path
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), cm_path)
    
    # Plot ROC curve
    if fpr is not None:
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plot_roc_curve(fpr, tpr, metrics['roc_auc'], roc_path)
    
    print(f"\nTest set evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation on Test Set")
    
    # Model and checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to specific checkpoint file (if not provided, auto-detect)")
    parser.add_argument("--ckpt_path", type=str, default="./output",
                       help="Base path to search for checkpoint")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to dataset")
    parser.add_argument("--model_input_size", type=int, default=23,
                       help="Number of input channels (23 or 306)")
    parser.add_argument("--path_norm_global_channel_zscore", type=str, 
                       default="assets/norm/time",
                       help="Path to normalization statistics")
    
    # Evaluation arguments
    parser.add_argument("--eval_batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU evaluation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args)