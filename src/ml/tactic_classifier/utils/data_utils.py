import numpy as np
import torch

from collections import Counter
from typing import Tuple, List
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import random_split
from torch.utils.data import Subset, random_split
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from utils.project_root import find_project_root

def create_output_directory(output_dir: str) -> Path:
    """
    Creates a timestamped output directory to store experiment results.
    
    Args:
        output_dir (str): Base directory for saving results.
        
    Returns:
        Path: Path object pointing to the created run directory.
    """
    project_root = find_project_root()
    output_path = project_root / Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"kfold_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    return run_dir

def get_dataset_labels(dataset) -> List[int]:
    """
    Extracts labels from a dataset and prints basic statistics.
    
    Args:
        dataset: A PyTorch Dataset object with elements (features, label).
        
    Returns:
        List[int]: List of labels corresponding to each sample in the dataset.
    """
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")
    
    return labels, label_counts

def get_class_weight_tensor(labels) -> torch.Tensor:
    """
    Computes a tensor of class weights to handle class imbalance.
    
    Args:
        dataset: A PyTorch Dataset object with elements (features, label).
        
    Returns:
        torch.Tensor: Tensor of class weights (dtype=torch.float).
    """    
    unique_labels = np.array(sorted(set(labels)))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    print(f"Class weights: {class_weights}")
    return class_weights_tensor

def get_model_dimensions(dataset) -> Tuple[int, int, List[int]]:
    """
    Determines the model configuration based on the dataset.
    
    Args:
        dataset: A PyTorch Dataset object with elements (features, label) and
                 a 'label_to_id' mapping attribute.
    
    Returns:
        Tuple[int, int, List[int]]: 
            - input_dim: Dimension of input features
            - num_classes: Number of target classes
            - hidden_dims: List of hidden layer sizes
    """
    input_dim = dataset[0][0].shape[0]
    num_classes = len(dataset.label_to_id)
    
    hidden_dims = [
        max(128, int(input_dim * 0.5)),
        max(64, int(input_dim * 0.25)),
        32
    ]
    
    print(f"\nModel Configuration:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Number of classes: {num_classes}")
    
    return input_dim, num_classes, hidden_dims

def split_train_test(dataset, test_size: float = 0.15, stratify: bool = True) -> Tuple[Subset, Subset]:
    """
    Splits a dataset into training+validation and test sets, optionally using stratified sampling.
    
    Args:
        dataset: PyTorch Dataset object with elements (features, label).
        test_size (float, optional): Fraction of dataset to reserve for testing. Defaults to 0.15.
        stratify (bool, optional): Whether to preserve class distribution in the split. Defaults to True.
        
    Returns:
        Tuple[Subset, Subset]: 
            - train_val_set: Subset for training and validation
            - test_set: Subset for final testing
    """
    if stratify:
        # Extract labels for stratification
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        indices = np.arange(len(dataset))

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=targets,
            random_state=42,
            shuffle=True
        )

        train_val_set = Subset(dataset, train_val_idx)
        test_set = Subset(dataset, test_idx)

    else:
        # Compute sizes for non-stratified random split
        test_count = int(test_size * len(dataset))
        train_val_count = len(dataset) - test_count

        train_val_set, test_set = random_split(
            dataset,
            [train_val_count, test_count],
            generator=torch.Generator().manual_seed(42)
        )

    print(f"Train+Val: {len(train_val_set)}, Test: {len(test_set)}")
    return train_val_set, test_set
