"""
Data Loader Module

This module handles loading and preprocessing the dataset for training.
It creates DataLoaders with appropriate transformations for train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(data_dir='data/eardrum_split', image_size=224, batch_size=8, num_workers=0):
    """
    Create DataLoaders for training, validation, and testing.
    
    This function:
    1. Defines image transformations for training and validation
    2. Creates datasets for each split
    3. Creates DataLoaders with appropriate settings
    
    Args:
        data_dir (str): Path to the dataset directory
        image_size (int): Size to resize images to (default: 224)
        batch_size (int): Number of images per batch (default: 32)
        num_workers (int): Number of worker processes for loading (auto-detected if None)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    
    # Use single-threaded loading for Docker compatibility
    # num_workers=0 prevents multiprocessing issues in containers
    
    # Define transformations for training (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),      # Resize to square
        transforms.RandomHorizontalFlip(),                # Randomly flip horizontally
        transforms.RandomRotation(15),                    # Random rotation up to 15 degrees
        transforms.ToTensor(),                            # Convert to tensor
        transforms.Normalize(                              # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Define transformations for validation and testing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),      # Resize to square
        transforms.ToTensor(),                            # Convert to tensor
        transforms.Normalize(                              # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create datasets for each split
    train_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train', 
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f'{data_dir}/val', 
        transform=val_test_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=f'{data_dir}/test', 
        transform=val_test_transform
    )

    # Create DataLoaders with Docker-compatible settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,           # Shuffle training data
        num_workers=num_workers,  # Multi-threaded for faster loading
        pin_memory=False        # Disable pin_memory for CPU-only training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,          # Don't shuffle validation data
        num_workers=num_workers,  # Multi-threaded for faster loading
        pin_memory=False        # Disable pin_memory for CPU-only training
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,          # Don't shuffle test data
        num_workers=num_workers,  # Multi-threaded for faster loading
        pin_memory=False        # Disable pin_memory for CPU-only training
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders()
    
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        break
