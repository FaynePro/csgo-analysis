import torch

from ml.tactic_classifier.classifier_trainer.feedforward_tactic_classifier import FeedforwardTacticClassifier
from ml.tactic_classifier.classifier_trainer.tactic_feature_dataset import TacticFeatureDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
from ml.tactic_classifier.utils.data_utils import get_dataset_labels, get_class_weight_tensor, get_model_dimensions
from ml.tactic_classifier.classifier_trainer.tactic_classifier_trainer import TacticClassifierTrainer

class SingleSplitTrainer:
    def train_classifier(
        self,
        data_root_dir: str = "",
        tactics_json_path: str = "",
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        train_split: float = 0.7,
        val_split: float = 0.15,
    ):
        """
        Args:
            data_root_dir: Directory containing data
            tactics_json_path: Path to tactic definitions
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            train_split: Proportion of data for training (0.0-1.0)
            val_split: Proportion of remaining data for validation
        """
        print("\n" + "="*60)
        print("Initializing Pretrained Classifier Training Pipeline")
        print("="*60)
        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset
        print("\nLoading raw graph data with labels...")
        dataset = TacticFeatureDataset(
            data_root_dir=data_root_dir,
            tactics_json_path=tactics_json_path
        )
        
        labels, label_counts = get_dataset_labels(dataset)
        class_weights_tensor = get_class_weight_tensor(labels)
        
        # Split dataset into train/val/test
        print("\nSplitting data into train/val/test sets...")
        train_size = int(train_split * len(dataset))
        remaining_size = len(dataset) - train_size
        val_size = int(val_split * remaining_size)
        test_size = remaining_size - val_size
        
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        input_dim, num_classes, hidden_dims = get_model_dimensions(dataset)

        model = FeedforwardTacticClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=0.3
        )
        
        print(f"Input dimension: {input_dim} (raw node features: max_nodes={dataset.max_nodes} * 7 features/node)")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Number of classes: {num_classes}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize trainer
        trainer = TacticClassifierTrainer(model, device=device)
        
        # Train with finetuning
        history = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            class_weights=class_weights_tensor,
            patience=10
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        return model, dataset, history