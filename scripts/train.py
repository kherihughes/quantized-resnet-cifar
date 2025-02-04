#!/usr/bin/env python3
"""
Training script for ResNet model on CIFAR-10 with optional distributed training support.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.quantizable_resnet import create_resnet18


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_data_loader(batch_size, train=True, distributed=False, rank=None, world_size=None):
    """Create CIFAR-10 data loaders."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if train else transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    if distributed and train:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
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


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, distributed=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(
            loss=f"{running_loss/(total/train_loader.batch_size):.4f}",
            acc=f"{100.*correct/total:.2f}%"
        )

    if distributed:
        metrics = torch.tensor([running_loss, correct, total], device=device)
        torch.distributed.all_reduce(metrics)
        running_loss, correct, total = metrics.tolist()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def train(rank, world_size, args):
    """Main training function."""
    distributed = args.distributed and world_size > 1
    
    if distributed:
        setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if distributed else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_resnet18()
    if distributed:
        model = DDP(model.to(device), device_ids=[rank])
    else:
        model = model.to(device)

    # Create data loaders
    train_loader = get_data_loader(
        args.batch_size,
        train=True,
        distributed=distributed,
        rank=rank if distributed else None,
        world_size=world_size if distributed else None
    )
    val_loader = get_data_loader(args.batch_size, train=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        if distributed and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, distributed
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if rank == 0 or not distributed:
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state_dict, args.save_path)
                print(f"New best model saved to {args.save_path}")

    if distributed:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save-path', type=str, default='models/resnet18_cifar10.pth')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.distributed and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, 1, args)


if __name__ == '__main__':
    main() 