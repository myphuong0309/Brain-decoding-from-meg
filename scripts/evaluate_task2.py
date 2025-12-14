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
    classification_report
)
import json
import matplotlib.pyplot as plt
import seaborn as sns
from models.model_task2 import BrainPhonemeClassifier
from utils.processed_data_task2 import get_test_dataloader_task2


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    model = BrainPhonemeClassifier.load_from_checkpoint(checkpoint_path)
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


def compute_metrics(y_true, y_pred, num_classes):
    """Compute comprehensive evaluation metrics for multiclass classification"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_weighted': float(f1_weighted),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_micro': float(f1_micro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'confusion_matrix': cm.tolist(),
        'num_classes': num_classes,
        'num_samples': len(y_true)
    }
    
    return metrics


def plot_confusion_matrix(cm, save_path, phoneme_list=None, top_k=50):
    """Plot and save confusion matrix (showing top_k classes for readability)"""
    plt.figure(figsize=(20, 18))
    
    if cm.shape[0] > top_k:
        # Show only top_k most frequent classes
        class_counts = cm.sum(axis=1)
        top_classes = np.argsort(class_counts)[-top_k:]
        cm_subset = cm[top_classes][:, top_classes]
        
        if phoneme_list is not None:
            labels = [phoneme_list[i] for i in top_classes]
        else:
            labels = [str(i) for i in top_classes]
    else:
        cm_subset = cm
        if phoneme_list is not None:
            labels = phoneme_list
        else:
            labels = [str(i) for i in range(cm.shape[0])]
    
    sns.heatmap(cm_subset, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Top {len(labels)} Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def plot_per_class_metrics(y_true, y_pred, phoneme_list, save_path, top_k=30):
    """Plot per-class F1 scores for top_k most frequent classes"""
    precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Get top_k most frequent classes
    top_classes = np.argsort(support_class)[-top_k:]
    
    plt.figure(figsize=(14, 8))
    x_pos = np.arange(len(top_classes))
    
    if phoneme_list is not None:
        labels = [phoneme_list[i] for i in top_classes]
    else:
        labels = [str(i) for i in top_classes]
    
    plt.bar(x_pos, f1_class[top_classes], alpha=0.7)
    plt.xlabel('Phoneme Class')
    plt.ylabel('F1 Score')
    plt.title(f'Per-Class F1 Scores (Top {top_k} Most Frequent)')
    plt.xticks(x_pos, labels, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved per-class F1 scores to: {save_path}")


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find the latest checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_pattern = os.path.join(args.model_dir, "**/*.ckpt")
        checkpoints = glob(checkpoint_pattern, recursive=True)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {args.model_dir}")
        args.checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Using latest checkpoint: {args.checkpoint}")
    
    # Load phoneme mapping
    # Checkpoint path: ./output_task2/model_XXX/version_0/checkpoints/epoch=X.ckpt
    # Phoneme mapping: ./output_task2/model_XXX/phoneme_to_idx.pt
    checkpoint_parts = args.checkpoint.split(os.sep)
    model_dir_idx = checkpoint_parts.index('version_0') - 1 if 'version_0' in checkpoint_parts else -3
    model_base_dir = os.sep.join(checkpoint_parts[:model_dir_idx + 1])
    phoneme_mapping_path = os.path.join(model_base_dir, "phoneme_to_idx.pt")
    
    if not os.path.exists(phoneme_mapping_path):
        # Try alternative location
        alt_model_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint)))
        phoneme_mapping_path = os.path.join(alt_model_base_dir, "phoneme_to_idx.pt")
        if not os.path.exists(phoneme_mapping_path):
            raise FileNotFoundError(f"Phoneme mapping not found at {phoneme_mapping_path}")
    
    phoneme_to_idx = torch.load(phoneme_mapping_path)
    idx_to_phoneme = {v: k for k, v in phoneme_to_idx.items()}
    phoneme_list = [idx_to_phoneme[i] for i in range(len(phoneme_to_idx))]
    num_classes = len(phoneme_to_idx)
    
    print(f"Loaded phoneme mapping with {num_classes} classes")
    
    # Load model
    model = load_checkpoint(args.checkpoint)
    
    # Load test data
    test_loader = get_test_dataloader_task2(
        data_path=args.data_path,
        phoneme_to_idx=phoneme_to_idx,
        num_workers=4,
        eval_batch_size=args.batch_size,
        n_input=args.model_input_size,
        grouped_samples=args.grouped_samples,
        path_norm_global_channel_zscore=args.path_norm_global_channel_zscore
    )
    
    # Run evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on {device}...")
    
    logits, labels = evaluate_model(model, test_loader, device=device)
    predictions = np.argmax(logits, axis=1)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions, num_classes)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Task 2: Phoneme Classification)")
    print("="*50)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Number of classes: {metrics['num_classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print("="*50 + "\n")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics_task2.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions_task2.npz')
    np.savez(predictions_path, 
             predictions=predictions, 
             labels=labels, 
             logits=logits)
    print(f"Saved predictions to: {predictions_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix_task2.png')
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), cm_path, 
                         phoneme_list=phoneme_list, top_k=50)
    
    # Plot per-class metrics
    per_class_path = os.path.join(args.output_dir, 'per_class_f1_task2.png')
    plot_per_class_metrics(labels, predictions, phoneme_list, per_class_path, top_k=30)
    
    # Print detailed classification report for top classes
    print("\nDetailed Classification Report:")
    print("-" * 50)
    
    # Get unique classes that actually appear in the test set
    unique_classes = np.unique(np.concatenate([labels, predictions]))
    target_names_subset = [phoneme_list[i] for i in unique_classes if i < len(phoneme_list)]
    
    report = classification_report(labels, predictions, 
                                   labels=unique_classes,
                                   target_names=target_names_subset,
                                   zero_division=0, digits=4)
    
    # Save full report
    report_path = os.path.join(args.output_dir, 'classification_report_task2.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved full classification report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate phoneme classification model (Task 2)")
    parser.add_argument("--data_path", type=str, default="./assets/data_task2",
                       help="Path to test data")
    parser.add_argument("--model_dir", type=str, default="./output_task2",
                       help="Directory containing model checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint path (if None, uses latest)")
    parser.add_argument("--output_dir", type=str, default="./test_results_task2",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--model_input_size", type=int, default=306,
                       help="Model input size (23 or 306 sensors)")
    parser.add_argument("--grouped_samples", type=int, default=100,
                       help="Number of samples to group for testing (default: 100)")
    parser.add_argument("--path_norm_global_channel_zscore", type=str, 
                       default="assets/norm_task2/time",
                       help="Path to normalization statistics")
    
    args = parser.parse_args()
    main(args)
