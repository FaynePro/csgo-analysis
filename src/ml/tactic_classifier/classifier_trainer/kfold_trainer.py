import numpy as np
import torch
import json

from ml.tactic_classifier.classifier_trainer.feedforward_tactic_classifier import FeedforwardTacticClassifier
from ml.tactic_classifier.classifier_trainer.tactic_feature_dataset import TacticFeatureDataset
from ml.tactic_classifier.classifier_trainer.tactic_classifier_trainer import TacticClassifierTrainer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from ml.tactic_classifier.utils.data_utils import create_output_directory, get_dataset_labels, get_class_weight_tensor, split_train_test, get_model_dimensions
from ml.tactic_classifier.utils.reporting import compute_detailed_metrics, plot_confusion_matrix, export_metrics_to_csv, document_fold_results, get_cross_validation_summary, aggregate_fold_metrics

def export_kfold_summary_to_csv(k_folds, num_epochs, batch_size, 
                                learning_rate, test_split, input_dim, 
                                hidden_dims, num_classes, cross_validation_summary, 
                                test_loss, test_acc, test_metrics, 
                                best_fold, class_names, label_counts,
                                output_dir) -> dict:

    summary = {
        'configuration': {
            'k_folds': k_folds,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'test_split': test_split,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes
        },
        'cross_validation': cross_validation_summary,
        'test_results': {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'test_macro_f1': float(test_metrics['macro_f1']),
            'test_weighted_f1': float(test_metrics['weighted_f1']),
            'best_fold': int(best_fold['fold'])
        },
        'class_names': class_names,
        'label_distribution': label_counts
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

class KfoldTrainer:

    def train_classifier_kfold(
        self,
        data_root_dir: str = "",
        tactics_json_path: str = "",
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        k_folds: int = 5,
        test_split: float = 0.15,
        output_dir: str = "results",
        use_stratified_split: bool = True
    ):
        """
        Train classifier using k-fold cross-validation with comprehensive metrics.
        
        Args:
            data_root_dir: Directory containing the data
            tactics_json_path: Path to tactic definitions
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            k_folds: Number of folds for cross-validation
            test_split: Proportion of data to hold out for final testing
            output_dir: Directory to save results and plots
        """

        run_dir = create_output_directory(output_dir)

        print("\n" + "="*60 + f"\nInitializing {k_folds}-Fold Cross-Validation Training\n" + "="*60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("\nLoading raw data with labels...")
        dataset = TacticFeatureDataset(
            data_root_dir=data_root_dir,
            tactics_json_path=tactics_json_path
        )

        labels, label_counts = get_dataset_labels(dataset)
        class_weights_tensor = get_class_weight_tensor(labels)
        train_val_set, test_set = split_train_test(dataset,test_split,use_stratified_split)
        
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        input_dim, num_classes, hidden_dims = get_model_dimensions(dataset)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        
        print("\n" + "="*60 + f"\nStarting {k_folds}-Fold Cross-Validation\n" + "="*60)
        
        all_fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_set), 1):
            print("\n" + "="*60 + f"\nFOLD {fold}/{k_folds}\n" + "="*60)
            
            # Create fold directory
            fold_dir = run_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            
            train_subset = torch.utils.data.Subset(train_val_set, train_idx)
            val_subset = torch.utils.data.Subset(train_val_set, val_idx)
            
            print(f"Train: {len(train_subset)}, Validation: {len(val_subset)}")
            
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Initialize new model for the fold
            model = FeedforwardTacticClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=0.3
            )
            
            trainer = TacticClassifierTrainer(model, device=device)
            
            history = trainer.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                class_weights=class_weights_tensor,
                patience=10
            )
            
            # Compute detailed metrics for validation set
            val_metrics = compute_detailed_metrics(
                model, val_loader, device, dataset.label_to_id
            )
            
            fold_results.append({
                'fold': fold,
                'history': history,
                'best_val_loss': min(history['val_loss']),
                'best_val_acc': max(history['val_acc']),
                'model': model,
                'val_metrics': val_metrics
            })
            
            per_class_data = document_fold_results(val_metrics, train_subset, val_subset, fold, fold_results[-1], fold_dir)

            all_fold_metrics.append(per_class_data)

        cross_validation_summary = get_cross_validation_summary(fold_results)
        
        # Aggregate per-class metrics across folds
        print("\nPer-Class Average Metrics Across Folds:")
        class_names = fold_results[0]['val_metrics']['class_names']
        
        aggregate_fold_metrics(class_names, fold_results, run_dir)
        
        # Export all fold metrics combined
        all_fold_metrics_flat = [item for sublist in all_fold_metrics for item in sublist]
        export_metrics_to_csv(
            all_fold_metrics_flat,
            run_dir / "all_folds_per_class_metrics.csv"
        )
        
        # Compute average confusion matrix across folds
        avg_cm = np.mean([r['val_metrics']['confusion_matrix'] for r in fold_results], axis=0)
        avg_cm = avg_cm.astype(int)
        
        plot_confusion_matrix(
            avg_cm,
            class_names,
            "Average Confusion Matrix Across All Folds",
            run_dir / "average_confusion_matrix"
        )
        
        # Select best model (based on validation accuracy)
        best_fold = max(fold_results, key=lambda x: x['best_val_acc'])
        best_model = best_fold['model']
        
        print(f"\nBest model from Fold {best_fold['fold']}")
        
        # Evaluate best model on held-out test set
        print("\n" + "="*60 + f"\nFinal Test Set Evaluation\n" + "="*60)
        
        test_metrics = compute_detailed_metrics(
            best_model, test_loader, device, dataset.label_to_id
        )
        
        # Calculate test loss
        best_model.eval()
        test_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = best_model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = test_metrics['overall_accuracy']
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        print("\nPer-Class Test Metrics:")
        for class_name in class_names:
            print(f"\n{class_name}:")
            print(f"  Accuracy: {test_metrics['per_class_accuracy'][class_name]:.4f}")
            print(f"  Precision: {test_metrics['classification_report'][class_name]['precision']:.4f}")
            print(f"  Recall: {test_metrics['classification_report'][class_name]['recall']:.4f}")
            print(f"  F1-Score: {test_metrics['classification_report'][class_name]['f1-score']:.4f}")
            print(f"  Support: {test_metrics['classification_report'][class_name]['support']}")
        
        # Save test set results
        test_dir = run_dir / "test_results"
        test_dir.mkdir(exist_ok=True)
        
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            test_metrics['class_names'],
            "Test Set Confusion Matrix (Best Model)",
            test_dir / "test_confusion_matrix"
        )
        
        # Prepare test per-class metrics
        test_class_data = []
        for class_name in test_metrics['class_names']:
            test_class_data.append({
                'class': class_name,
                'accuracy': test_metrics['per_class_accuracy'][class_name],
                'precision': test_metrics['classification_report'][class_name]['precision'],
                'recall': test_metrics['classification_report'][class_name]['recall'],
                'f1_score': test_metrics['classification_report'][class_name]['f1-score'],
                'support': test_metrics['classification_report'][class_name]['support']
            })
        
        export_metrics_to_csv(
            test_class_data,
            test_dir / "test_per_class_metrics.csv"
        )

        summary = export_kfold_summary_to_csv(k_folds, num_epochs, batch_size, 
                                    learning_rate, test_split, input_dim, 
                                    hidden_dims, num_classes, cross_validation_summary, 
                                    test_loss, test_acc, test_metrics, 
                                    best_fold, class_names, label_counts, run_dir)
        
        print("\n" + "="*60)
        print("K-Fold Cross-Validation Complete!")
        print(f"All results saved to: {run_dir}")
        print("="*60)
        
        return best_model, dataset, fold_results, summary, test_metrics
