import numpy as np
import torch
import json
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt

def compute_detailed_metrics(model, data_loader, device, label_to_id):
    """
    Compute detailed metrics including confusion matrix, per-class accuracy and F1.
    
    Returns:
        dict with confusion matrix, predictions, labels, and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get class names
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_names = [id_to_label[i] for i in range(len(label_to_id))]
    
    # Compute per-class metrics
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Overall metrics
    overall_accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class accuracy (from confusion matrix diagonal)
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_accuracy[class_name] = cm[i, i] / cm[i].sum()
        else:
            per_class_accuracy[class_name] = 0.0
    
    return {
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': class_names,
        'classification_report': report,
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_accuracy': per_class_accuracy
    }

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path.with_suffix('.png'):
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {save_path.with_suffix('.png')}")
    
    plt.close()

    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(save_path.with_suffix('.csv'))
    print(f"Exported confusion matrix to {save_path.with_suffix('.csv')}")

def export_metrics_to_csv(metrics_data, filepath):
    """Export metrics to CSV file."""
    df = pd.DataFrame(metrics_data)
    df.to_csv(filepath, index=False)
    print(f"Exported metrics to {filepath}")

def document_fold_results(val_metrics, train_subset, val_subset, fold, fold_results, fold_dir):
        plot_confusion_matrix(
            val_metrics['confusion_matrix'],
            val_metrics['class_names'],
            f"Fold {fold} - Validation Confusion Matrix",
            fold_dir / "confusion_matrix"
        )
        
        # Prepare per-class metrics for export
        per_class_data = []
        for class_name in val_metrics['class_names']:
            per_class_data.append({
                'fold': fold,
                'class': class_name,
                'accuracy': val_metrics['per_class_accuracy'][class_name],
                'precision': val_metrics['classification_report'][class_name]['precision'],
                'recall': val_metrics['classification_report'][class_name]['recall'],
                'f1_score': val_metrics['classification_report'][class_name]['f1-score'],
                'support': val_metrics['classification_report'][class_name]['support']
            })
        
        export_metrics_to_csv(
            per_class_data,
            fold_dir / "per_class_metrics.csv"
        )
        
        # Save fold summary
        fold_summary = {
            'fold': fold,
            'train_size': len(train_subset),
            'val_size': len(val_subset),
            'best_val_loss': fold_results['best_val_loss'],
            'best_val_acc': fold_results['best_val_acc'],
            'overall_accuracy': val_metrics['overall_accuracy'],
            'macro_f1': val_metrics['macro_f1'],
            'weighted_f1': val_metrics['weighted_f1']
        }
        
        with open(fold_dir / "fold_summary.json", 'w') as f:
            json.dump(fold_summary, f, indent=2)
        
        
        print(f"\nFold {fold} Results:")
        print(f"Best Val Loss: {fold_results['best_val_loss']:.4f}")
        print(f"Best Val Acc: {fold_results['best_val_acc']:.4f}")
        print(f"Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {val_metrics['weighted_f1']:.4f}")

        return per_class_data

def aggregate_fold_metrics(class_names, fold_results, output_dir):
    aggregated_class_metrics = []
    for class_name in class_names:
        class_accuracies = [r['val_metrics']['per_class_accuracy'][class_name] 
                           for r in fold_results]
        class_f1s = [r['val_metrics']['classification_report'][class_name]['f1-score'] 
                    for r in fold_results]
        class_precisions = [r['val_metrics']['classification_report'][class_name]['precision'] 
                           for r in fold_results]
        class_recalls = [r['val_metrics']['classification_report'][class_name]['recall'] 
                        for r in fold_results]
        
        aggregated_class_metrics.append({
            'class': class_name,
            'avg_accuracy': np.mean(class_accuracies),
            'std_accuracy': np.std(class_accuracies),
            'avg_precision': np.mean(class_precisions),
            'std_precision': np.std(class_precisions),
            'avg_recall': np.mean(class_recalls),
            'std_recall': np.std(class_recalls),
            'avg_f1_score': np.mean(class_f1s),
            'std_f1_score': np.std(class_f1s)
        })
        
        print(f"\n{class_name}:")
        print(f"  Accuracy: {np.mean(class_accuracies):.4f} ± {np.std(class_accuracies):.4f}")
        print(f"  Precision: {np.mean(class_precisions):.4f} ± {np.std(class_precisions):.4f}")
        print(f"  Recall: {np.mean(class_recalls):.4f} ± {np.std(class_recalls):.4f}")
        print(f"  F1-Score: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")
    
    # Export aggregated per-class metrics
    export_metrics_to_csv(
        aggregated_class_metrics,
        output_dir / "aggregated_per_class_metrics.csv"
    )

def get_cross_validation_summary(fold_results):
    # Aggregate results across folds
    print("\n" + "="*60 + f"\nCross-Validation Summary\n" + "="*60)

    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    std_val_loss = np.std([r['best_val_loss'] for r in fold_results])
    avg_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_val_acc = np.std([r['best_val_acc'] for r in fold_results])
    avg_macro_f1 = np.mean([r['val_metrics']['macro_f1'] for r in fold_results])
    std_macro_f1 = np.std([r['val_metrics']['macro_f1'] for r in fold_results])
    avg_weighted_f1 = np.mean([r['val_metrics']['weighted_f1'] for r in fold_results])
    std_weighted_f1 = np.std([r['val_metrics']['weighted_f1'] for r in fold_results])
    
    print(f"\nValidation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"Weighted F1: {avg_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    print("\nPer-Fold Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: "
              f"Loss={result['best_val_loss']:.4f}, "
              f"Acc={result['best_val_acc']:.4f}, "
              f"F1={result['val_metrics']['macro_f1']:.4f}")
    
    return {
        'avg_val_loss': float(avg_val_loss),
        'std_val_loss': float(std_val_loss),
        'avg_val_acc': float(avg_val_acc),
        'std_val_acc': float(std_val_acc),
        'avg_macro_f1': float(avg_macro_f1),
        'std_macro_f1': float(std_macro_f1),
        'avg_weighted_f1': float(avg_weighted_f1),
        'std_weighted_f1': float(std_weighted_f1)
}
