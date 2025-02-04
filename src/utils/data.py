"""
Utility functions for data loading and preprocessing.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """
    Get the transforms for CIFAR-10 dataset.
    
    Args:
        train: Whether to return transforms for training or evaluation.
    
    Returns:
        Composition of transforms.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def get_cifar10_data(
    batch_size: int,
    train: bool = True,
    distributed: bool = False,
    rank: int = None,
    world_size: int = None
) -> DataLoader:
    """
    Create CIFAR-10 data loader.
    
    Args:
        batch_size: Number of samples per batch.
        train: Whether to load training or test data.
        distributed: Whether to use DistributedSampler.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
    
    Returns:
        DataLoader for CIFAR-10 dataset.
    """
    transform = get_cifar10_transforms(train)
    
    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )
    
    if distributed and train and rank is not None and world_size is not None:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        shuffle = False
    else:
        sampler = None
        shuffle = train
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    ) 