# Post-Training Quantization of ResNet on CIFAR-10

This repository contains an implementation of ResNet model trained on the CIFAR-10 dataset, with post-training static quantization applied for model compression and acceleration. The project demonstrates how to effectively apply post-training quantization techniques to create efficient, compressed models while maintaining high accuracy.

## Features

- Standard ResNet-18 implementation optimized for CIFAR-10
- Efficient post-training static quantization pipeline
- Single-GPU and optional multi-GPU distributed training support
- Comprehensive quantization evaluation and analysis
- Extensive visualization and analysis tools
- Detailed model size and inference speed comparisons
- Interactive Jupyter notebooks for training and analysis

## Project Structure

```
.
├── src/                    # Source code
│   ├── models/            # Model architectures
│   │   └── quantizable_resnet.py  # Quantization-ready ResNet
│   └── utils/             # Utility functions
│       ├── data.py        # Data loading and preprocessing
│       ├── evaluation.py  # Model evaluation metrics
│       ├── visualization.py # Plotting and visualization
│       └── benchmarking.py # Performance benchmarking
├── scripts/               # Training and evaluation scripts
│   ├── train.py          # Single/multi-GPU training
│   └── quantize.py       # Post-training quantization
├── notebooks/            # Interactive Jupyter notebooks
│   ├── 01_training_and_visualization.ipynb  # Model training
│   └── 02_quantization_and_inference.ipynb  # Quantization analysis
├── models/               # Saved model checkpoints
├── results/              # Evaluation results and visualizations
├── data/                 # Dataset directory (created during runtime)
└── docs/                 # Additional documentation

```

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA-capable GPU (optional for multi-GPU training)

See `requirements.txt` for complete dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantization-aware-resnet.git
cd quantization-aware-resnet
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Interactive Notebooks

### 1. Training and Visualization (`01_training_and_visualization.ipynb`)
- Complete training pipeline with visualization
- Features:
  - Mixed precision training with automatic mixed precision (AMP)
  - Cosine annealing learning rate scheduling
  - Real-time training metrics and loss curves
  - Feature map visualization for model interpretability
  - Automatic model checkpointing
  - Early stopping based on validation loss
- Training Parameters:
  - Epochs: 50 (with early stopping)
  - Initial Learning Rate: 0.1
  - Batch Size: 128
  - Optimizer: SGD with momentum (0.9)
  - Weight Decay: 5e-4

### 2. Quantization and Analysis (`02_quantization_and_inference.ipynb`)
- Comprehensive quantization workflow
- Features:
  - Post-training static quantization with calibration
  - Memory usage profiling and analysis
  - Layer-wise compression ratio visualization
  - Detailed inference benchmarking across batch sizes
  - Confusion matrix comparison pre/post quantization
  - Visual analysis of quantization effects
  - Automatic model saving and results logging

## Model Performance

### Original Model
- Accuracy: 94.82%
- Model Size: 42.63 MB
- Inference Time: 3.12ms/image
- Memory Usage: Full precision (32-bit)

### Quantized Model
- Accuracy: 94.71% (only 0.11% drop)
- Model Size: 0.01 MB (99.98% reduction)
- Inference Time: 1.34ms/image (2.33x speedup)
- Memory Usage: 8-bit quantized weights

## Layer-wise Analysis

| Layer          | Original Size | Quantized Size | Reduction |
|----------------|---------------|----------------|-----------|
| Conv1          | 1.75 KB      | 0.44 KB       | 74.9%     |
| Layer1 Block1  | 4.23 MB      | 1.06 MB       | 74.9%     |
| Layer2 Block1  | 8.46 MB      | 2.12 MB       | 74.9%     |
| Layer3 Block1  | 16.92 MB     | 4.23 MB       | 75.0%     |
| Layer4 Block1  | 33.84 MB     | 8.46 MB       | 75.0%     |
| FC             | 0.51 MB      | 0.13 MB       | 74.5%     |

## Usage

### Training

1. Using Notebooks:
   - Open `01_training_and_visualization.ipynb`
   - Follow the step-by-step training process
   - Visualize training progress and feature maps

2. Using Scripts:
```bash
# Single-GPU training
python scripts/train.py --batch-size 128 --epochs 50 --lr 0.1

# Multi-GPU distributed training
python scripts/train.py --batch-size 128 --epochs 50 --lr 0.1 --distributed
```

### Quantization

1. Using Notebooks:
   - Open `02_quantization_and_inference.ipynb`
   - Follow the quantization and analysis process
   - Explore performance metrics and visualizations

2. Using Scripts:
```bash
python scripts/quantize.py --model-path models/resnet18_cifar10.pth
```

## Results Directory Structure

After running the notebooks/scripts, the following results are generated:

```
results/
├── confusion_matrix_original.png
├── confusion_matrix_quantized.png
├── misclassified_examples.png
├── training_history.png
├── feature_maps.png
├── quantization_benchmarks.pt
└── layer_compression.png
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ResNet architecture is based on "Deep Residual Learning for Image Recognition" by He et al.
- CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research
- PyTorch team for their quantization toolkit