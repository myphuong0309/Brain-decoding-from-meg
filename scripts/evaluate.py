import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

# --- The above three lines must stay at the very top for module import to work ---

import argparse
import torch
import numpy as np
from pytorch_lightning import Trainer
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
from utils.processed_data import get_dataloaders


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    model = BrainSpeechClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(model, dataloader, device='cuda'):
    """Run inference and collect predictions"""
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze()
            
            all_logits.extend(logits.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_logits)


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
             label=f'ROC curve (AUC = {roc_auc:.3f})')
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


def evaluate_fold(checkpoint_path, args, fold):
    """Evaluate a single fold"""
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold}")
    print(f"{'='*60}")
    
    # Load model
    model = load_checkpoint(checkpoint_path)
    
    # Get dataloader
    _, val_loader = get_dataloaders(
        data_path=args.data_path,
        num_workers=4,
        fold=fold,
        n_splits=args.n_splits,
        train_batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        n_input=args.model_input_size,
        path_norm_global_channel_zscore=args.path_norm_global_channel_zscore
    )
    
    # Run evaluation
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    y_pred, y_true, y_logits = evaluate_model(model, val_loader, device)
    
    # Compute metrics
    metrics, (fpr, tpr, thresholds) = compute_metrics(y_true, y_pred, y_logits)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    output_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
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
    
    return metrics


def aggregate_fold_results(all_fold_metrics, args):
    """Aggregate results across all folds"""
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS ACROSS ALL FOLDS")
    print(f"{'='*60}")
    
    # Compute mean and std for each metric
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 
                   'precision_macro', 'recall_macro', 'roc_auc']
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_fold_metrics if m[metric_name] is not None]
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
    
    # Print aggregated results
    for metric_name, stats in aggregated.items():
        print(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save aggregated results
    agg_path = os.path.join(args.output_dir, "aggregated_metrics.json")
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated metrics saved to: {agg_path}")
    
    return aggregated


def main(args):
    """Main evaluation function"""
    print(f"Starting evaluation...")
    print(f"Checkpoint path: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_fold_metrics = []
    
    if args.fold is not None:
        # Evaluate single fold
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("--checkpoint_path is required when evaluating a single fold")
        
        metrics = evaluate_fold(checkpoint_path, args, args.fold)
        all_fold_metrics.append(metrics)
    else:
        # Evaluate all folds
        for fold in range(args.n_splits):
            # Find checkpoint for this fold
            fold_dir = args.ckpt_path
            
            # Look for checkpoint files in fold directory
            checkpoint_candidates = []
            for root, dirs, files in os.walk(fold_dir):
                if f"fold_{fold}" in root:
                    for file in files:
                        if file.endswith('.ckpt'):
                            checkpoint_candidates.append(os.path.join(root, file))
            
            if not checkpoint_candidates:
                print(f"Warning: No checkpoint found for fold {fold}, skipping...")
                continue
            
            # Use the first checkpoint found (or you can add logic to select best)
            checkpoint_path = checkpoint_candidates[0]
            print(f"Using checkpoint: {checkpoint_path}")
            
            try:
                metrics = evaluate_fold(checkpoint_path, args, fold)
                all_fold_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating fold {fold}: {e}")
                continue
    
    # Aggregate results if multiple folds
    if len(all_fold_metrics) > 1:
        aggregate_fold_results(all_fold_metrics, args)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Speech Classifier")
    
    # Model and checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to specific checkpoint file (for single fold evaluation)")
    parser.add_argument("--ckpt_path", type=str, default="./",
                       help="Base path for checkpoints (for multi-fold evaluation)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to dataset")
    parser.add_argument("--model_input_size", type=int, default=23,
                       help="Number of input channels (23 or 306)")
    parser.add_argument("--path_norm_global_channel_zscore", type=str, 
                       default="assets/norm/time",
                       help="Path to normalization statistics")
    
    # Evaluation arguments
    parser.add_argument("--fold", type=int, default=None,
                       help="Specific fold to evaluate (if None, evaluates all folds)")
    parser.add_argument("--n_splits", type=int, default=5,
                       help="Total number of folds")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU evaluation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args)
