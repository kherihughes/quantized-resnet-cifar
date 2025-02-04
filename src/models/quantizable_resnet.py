"""
Quantization-friendly ResNet implementation for CIFAR-10.
"""

import torch
import torch.nn as nn
from typing import Type
from torch import Tensor

class QuantizableBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1, downsample: nn.Module = None, use_batch_norm: bool = True) -> None:
        super(QuantizableBasicBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.downsample = downsample
        self.expansion = expansion
        
        # Main layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        # For quantization-friendly addition
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        
        return out

class QuantizableResNet(nn.Module):
    def __init__(self, img_channels: int, num_layers: int, block: Type[QuantizableBasicBlock], num_classes: int = 10, use_batch_norm: bool = True) -> None:
        super(QuantizableResNet, self).__init__()
        layers = [2, 2, 2, 2] if num_layers == 18 else []
        self.expansion = 1
        self.in_channels = 64
        self.use_batch_norm = use_batch_norm
        
        # Initial layers
        self.conv1 = nn.Conv2d(img_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # Main layers
        self.layer1 = self._make_layer(block, 64, layers[0], use_batch_norm=use_batch_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_batch_norm=use_batch_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_batch_norm=use_batch_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_batch_norm=use_batch_norm)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        
        # Quantization layers
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block: Type[QuantizableBasicBlock], out_channels: int, blocks: int, stride: int = 1, use_batch_norm: bool = True) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
            )

        layers = [block(self.in_channels, out_channels, stride, self.expansion, downsample, use_batch_norm=use_batch_norm)]
        self.in_channels = out_channels * self.expansion
        layers += [block(self.in_channels, out_channels, expansion=self.expansion, use_batch_norm=use_batch_norm) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Quantize input
        x = self.quant(x)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Main layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x
        
    def fuse_model(self):
        """Fuse Conv+BN layers for quantization."""
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        for m in self.modules():
            if isinstance(m, QuantizableBasicBlock):
                torch.quantization.fuse_modules(m, ['conv1', 'bn1'], inplace=True)
                torch.quantization.fuse_modules(m, ['conv2', 'bn2'], inplace=True)

def create_resnet18(num_classes: int = 10, **kwargs) -> QuantizableResNet:
    """Helper function to create a ResNet-18 instance."""
    return QuantizableResNet(
        img_channels=3,
        num_layers=18,
        block=QuantizableBasicBlock,
        num_classes=num_classes,
        **kwargs
    ) 