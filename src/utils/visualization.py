"""
Utility functions for visualization and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Tuple


def plot_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: List[str],
    title: str,
    save_path: str = None
) -> None:
    """
    Plot and optionally save confusion matrix.
    
    Args:
        targets: Ground truth labels.
        predictions: Model predictions.
        classes: List of class names.
        title: Plot title.
        save_path: Path to save the plot (optional).
    """
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation='vertical')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_misclassified_examples(
    dataset,
    predictions: np.ndarray,
    targets: np.ndarray,
    classes: List[str],
    save_path: str = None,
    num_examples: int = 9
) -> None:
    """
    Plot examples of misclassified images.
    
    Args:
        dataset: Dataset containing the images.
        predictions: Model predictions.
        targets: Ground truth labels.
        classes: List of class names.
        save_path: Path to save the plot (optional).
        num_examples: Number of examples to plot (default: 9).
    """
    misclassified_indices = np.where(predictions != targets)[0]
    if len(misclassified_indices) == 0:
        print("No misclassified examples found.")
        return

    np.random.seed(42)
    sample_indices = np.random.choice(
        misclassified_indices,
        size=min(num_examples, len(misclassified_indices)),
        replace=False
    )

    rows = int(np.ceil(np.sqrt(num_examples)))
    fig, axs = plt.subplots(rows, rows, figsize=(15, 15))
    fig.suptitle("Misclassified Examples", fontsize=16)

    # CIFAR-10 normalization constants
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for i, idx in enumerate(sample_indices):
        if i >= num_examples:
            break
            
        img, _ = dataset[idx]
        ax = axs[i//rows, i%rows] if rows > 1 else axs[i]
        
        # Denormalize image
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(
            f"Pred: {classes[predictions[idx]]}\nTrue: {classes[targets[idx]]}",
            fontsize=10
        )

    # Remove empty subplots
    for i in range(len(sample_indices), rows*rows):
        ax = axs[i//rows, i%rows] if rows > 1 else axs[i]
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_training_history(
    history: dict,
    save_path: str = None
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing training metrics.
        save_path: Path to save the plot (optional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() 