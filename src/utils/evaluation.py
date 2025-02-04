"""
Utility functions for model evaluation and performance metrics.
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model performance and timing.
    
    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader containing evaluation data.
        device: Device to run evaluation on.
    
    Returns:
        Tuple containing:
        - accuracy (%)
        - average time per batch (ms)
        - average time per image (ms)
        - array of predictions
        - array of targets
    """
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


def get_model_size(model: torch.nn.Module, temp_path: str = "temp.p") -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model to measure.
        temp_path: Temporary path to save model.
    
    Returns:
        Model size in MB.
    """
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / 1e6
    os.remove(temp_path)
    return size_mb


def print_performance_comparison(
    orig_metrics: Tuple[float, float, float, float],
    quant_metrics: Tuple[float, float, float, float]
) -> None:
    """
    Print performance comparison between original and quantized models.
    
    Args:
        orig_metrics: Tuple of (accuracy, size, batch_time, image_time) for original model.
        quant_metrics: Tuple of (accuracy, size, batch_time, image_time) for quantized model.
    """
    orig_acc, orig_size, orig_time_batch, orig_time_img = orig_metrics
    quant_acc, quant_size, quant_time_batch, quant_time_img = quant_metrics
    
    size_reduction = ((orig_size - quant_size) / orig_size) * 100
    speed_up_batch = orig_time_batch / quant_time_batch if quant_time_batch > 0 else float('inf')
    speed_up_img = orig_time_img / quant_time_img if quant_time_img > 0 else float('inf')

    print("\n=== Performance Comparison ===")
    print(f"{'Model':<20}{'Accuracy (%)':<15}{'Size (MB)':<15}{'Time/Batch (ms)':<20}{'Time/Image (ms)':<20}")
    print("-" * 85)
    print(f"{'Original':<20}{orig_acc:<15.2f}{orig_size:<15.2f}{orig_time_batch:<20.2f}{orig_time_img:<20.4f}")
    print(f"{'Quantized':<20}{quant_acc:<15.2f}{quant_size:<15.2f}{quant_time_batch:<20.2f}{quant_time_img:<20.4f}")
    print("-" * 85)
    print(f"Size Reduction: {size_reduction:.2f}%")
    print(f"Speed-up (Batch): {speed_up_batch:.2f}x")
    print(f"Speed-up (Image): {speed_up_img:.2f}x") 