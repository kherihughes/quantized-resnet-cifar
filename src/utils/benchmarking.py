"""
Standardized benchmarking utilities for model evaluation.
"""

import torch
import time
import os
import numpy as np
from typing import Dict, Tuple

def measure_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Measure model size accurately including all components.
    
    Returns:
        Dict containing:
        - parameter_size: Size of model parameters
        - buffer_size: Size of model buffers
        - total_size: Total model size in MB
    """
    # Parameter size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    
    # Buffer size (running_mean, running_var, etc.)
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    # Convert to MB
    param_size_mb = param_size / (1024 * 1024)
    buffer_size_mb = buffer_size / (1024 * 1024)
    total_size_mb = param_size_mb + buffer_size_mb
    
    return {
        'parameter_size_mb': param_size_mb,
        'buffer_size_mb': buffer_size_mb,
        'total_size_mb': total_size_mb
    }

def measure_inference_time(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    device: torch.device,
    num_iterations: int = 100,
    batch_size: int = 1,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure inference time accurately with warmup and multiple iterations.
    
    Returns:
        Dict containing:
        - avg_time_ms: Average time per inference in milliseconds
        - std_time_ms: Standard deviation of inference time
        - throughput: Images per second
    """
    model.eval()
    times = []
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Measure
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = (batch_size * 1000) / avg_time  # Images per second
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'throughput': throughput
    }

def evaluate_model_performance(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Comprehensive model evaluation including accuracy and memory usage.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Measure accuracy
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    
    # Measure size
    size_metrics = measure_model_size(model)
    
    # Measure inference time
    time_metrics = measure_inference_time(
        model,
        input_size=(3, 32, 32),  # CIFAR-10 image size
        device=device
    )
    
    return {
        'accuracy': accuracy,
        **size_metrics,
        **time_metrics
    }

def print_benchmark_results(original_metrics: Dict[str, float], quantized_metrics: Dict[str, float]):
    """Print benchmark results in a clear, formatted way."""
    print("\n=== Model Performance Comparison ===")
    print(f"{'Metric':<20} {'Original':<15} {'Quantized':<15} {'Change':<15}")
    print("-" * 65)
    
    metrics_to_show = [
        ('accuracy', 'Accuracy (%)', '{:.2f}'),
        ('total_size_mb', 'Size (MB)', '{:.2f}'),
        ('avg_time_ms', 'Time (ms/img)', '{:.3f}'),
        ('throughput', 'Throughput (img/s)', '{:.1f}')
    ]
    
    for key, name, fmt in metrics_to_show:
        orig_val = original_metrics[key]
        quant_val = quantized_metrics[key]
        
        if key == 'accuracy':
            change = f"{quant_val - orig_val:+.2f}%"
        else:
            change = f"{(quant_val - orig_val) / orig_val * 100:+.1f}%"
        
        print(f"{name:<20} {fmt.format(orig_val):<15} {fmt.format(quant_val):<15} {change:<15}")