#!/usr/bin/env python3
"""
Post-training quantization and evaluation script for ResNet model.
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.quantizable_resnet import create_resnet18


def get_data_loader(batch_size, train=False):
    """Create CIFAR-10 data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def evaluate_model(model, loader, device):
    """Evaluate model performance and timing."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    start = time.perf_counter()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    end = time.perf_counter()

    accuracy = 100.0 * correct / total
    total_time_ms = (end - start) * 1000
    avg_time_per_batch_ms = total_time_ms / len(loader)
    avg_time_per_image_ms = total_time_ms / total

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return accuracy, avg_time_per_batch_ms, avg_time_per_image_ms, all_preds, all_targets


def get_model_size(model, temp_path="temp.p"):
    """Get model size in MB."""
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / 1e6
    os.remove(temp_path)
    return size_mb


def plot_confusion_matrix(targets, preds, classes, title, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation='vertical')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_misclassified_examples(dataset, preds, targets, classes, save_path=None):
    """Plot misclassified examples."""
    misclassified_indices = np.where(preds != targets)[0]
    if len(misclassified_indices) == 0:
        return

    np.random.seed(42)
    sample_indices = np.random.choice(
        misclassified_indices,
        size=min(9, len(misclassified_indices)),
        replace=False
    )

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Misclassified Examples", fontsize=16)

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for i, idx in enumerate(sample_indices):
        img, _ = dataset[idx]
        ax = axs[i//3, i%3]
        
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f"Pred: {classes[preds[idx]]}\nTrue: {classes[targets[idx]]}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantize and evaluate ResNet model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--quantized-path', type=str, default='models/quantized_resnet18_cifar10.pth')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.quantized_path), exist_ok=True)

    # Set device to CPU for fair comparison
    device = torch.device("cpu")

    # Load the original model
    model = create_resnet18()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create data loader
    test_loader = get_data_loader(args.batch_size)
    classes = test_loader.dataset.classes

    # Evaluate original model
    orig_acc, orig_time_batch, orig_time_img, orig_preds, orig_targets = evaluate_model(
        model, test_loader, device
    )
    orig_size = get_model_size(model)

    # Prepare for quantization
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with a subset of data
    calibration_loader = get_data_loader(args.batch_size, train=True)
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            model(data)
            if i >= 10:  # Calibrate with 10 batches
                break

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    # Save quantized model
    torch.save(model.state_dict(), args.quantized_path)

    # Evaluate quantized model
    quant_acc, quant_time_batch, quant_time_img, quant_preds, quant_targets = evaluate_model(
        model, test_loader, device
    )
    quant_size = get_model_size(model)

    # Calculate improvements
    size_reduction = ((orig_size - quant_size) / orig_size) * 100
    speed_up_batch = orig_time_batch / quant_time_batch
    speed_up_img = orig_time_img / quant_time_img

    # Print results
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<20}{'Accuracy (%)':<15}{'Size (MB)':<15}{'Time/Batch (ms)':<20}{'Time/Image (ms)':<20}")
    print("-" * 85)
    print(f"{'Original':<20}{orig_acc:<15.2f}{orig_size:<15.2f}{orig_time_batch:<20.2f}{orig_time_img:<20.4f}")
    print(f"{'Quantized':<20}{quant_acc:<15.2f}{quant_size:<15.2f}{quant_time_batch:<20.2f}{quant_time_img:<20.4f}")
    print("-" * 85)
    print(f"Size Reduction: {size_reduction:.2f}%")
    print(f"Speed-up (Batch): {speed_up_batch:.2f}x")
    print(f"Speed-up (Image): {speed_up_img:.2f}x")

    # Plot confusion matrices
    plot_confusion_matrix(
        orig_targets, orig_preds, classes,
        "Confusion Matrix - Original Model",
        os.path.join(args.output_dir, "confusion_matrix_original.png")
    )
    plot_confusion_matrix(
        quant_targets, quant_preds, classes,
        "Confusion Matrix - Quantized Model",
        os.path.join(args.output_dir, "confusion_matrix_quantized.png")
    )

    # Plot misclassified examples
    plot_misclassified_examples(
        test_loader.dataset, quant_preds, quant_targets, classes,
        os.path.join(args.output_dir, "misclassified_examples.png")
    )


if __name__ == '__main__':
    main() 