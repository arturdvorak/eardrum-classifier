"""
Dataset Setup Module

This module handles downloading, organizing, and splitting the eardrum dataset.
It's designed to be simple and efficient, following best practices.
"""

import os
import shutil
import random
import zipfile
from pathlib import Path
import pandas as pd

def download_and_unpack_kaggle_dataset():
    """
    Download and unpack the Kaggle dataset.
    
    This function:
    1. Checks if the dataset already exists
    2. Downloads it from Kaggle if needed
    3. Extracts the zip file
    4. Cleans up old data to start fresh
    
    Returns:
        None
    """
    # Dataset information
    kaggle_dataset = "erdalbasaran/eardrum-dataset-otitis-media"
    zip_file = Path("data/raw/eardrum-dataset-otitis-media.zip")
    extract_dir = Path("data/eardrum_dataset")
    
    # Ensure data/raw directory exists
    zip_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean up old data to start fresh
    if extract_dir.exists():
        print("Removing old extracted dataset...")
        shutil.rmtree(extract_dir)

    if Path("data/eardrum_split").exists():
        print("Removing old split dataset...")
        shutil.rmtree("data/eardrum_split")

    # Download dataset if we don't have it
    if not zip_file.exists():
        print("Downloading dataset from Kaggle...")
        os.system(f'kaggle datasets download -d {kaggle_dataset}')
    else:
        print("Found existing zip file. Skipping download.")

    # Extract the zip file
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Dataset extracted to {extract_dir.resolve()}")

def setup_dataset():
    """
    Main function to set up the complete dataset.
    
    This function:
    1. Downloads and extracts the dataset
    2. Organizes images into classes
    3. Splits data into train/validation/test sets
    4. Creates CSV files for easy loading
    5. Saves metadata about the dataset
    
    Returns:
        Path: Path to the processed data directory
    """
    print("Setting up dataset...")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Step 1: Download and extract dataset
    download_and_unpack_kaggle_dataset()
    
    # Step 2: Set up paths and parameters
    src_dir = Path("data/eardrum_dataset")  # Where the extracted data is
    base_dir = Path('data/eardrum_split')             # Where we'll put our organized data
    splits = ['train', 'val', 'test']            # The three data splits
    split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}  # How to split the data
    
    # Step 3: Show what classes we have
    print("\nOriginal class distribution (before filtering):")
    all_classes = [d for d in src_dir.iterdir() if d.is_dir()]
    for cls_dir in all_classes:
        count = len(list(cls_dir.glob("*")))
        print(f"  {cls_dir.name}: {count} images")
    
    # Step 4: Define which classes to exclude (too few samples)
    excluded_classes = {
        'Foreign',            # Only 3 samples - too few
        'PseduoMembran',      # Only 11 samples - too few
        'Earventulation',     # Only 16 samples - too few
        'tympanoskleros',     # Only 28 samples - unstable
        'OtitExterna'         # Only 41 samples - weak performance
    }
    
    # Step 5: Filter to keep only good classes
    classes = [d.name for d in src_dir.iterdir() if d.is_dir() and d.name not in excluded_classes]
    
    print(f"\nExcluded classes: {excluded_classes}")
    print(f"Classes to use: {classes}")
    
    # Step 6: Create folders for each split and class
    for split in splits:
        for cls in classes:
            os.makedirs(base_dir / split / cls, exist_ok=True)
    
    # Step 7: Split the dataset
    print("\nSplitting dataset...")
    for cls in classes:
        # Get all images for this class
        images = list((src_dir / cls).glob('*'))
        random.shuffle(images)  # Randomize order

        # Calculate split points
        total = len(images)
        train_end = int(split_ratio['train'] * total)
        val_end = train_end + int(split_ratio['val'] * total)

        # Split images into groups
        split_files = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Copy images to appropriate folders
        for split, files in split_files.items():
            for img in files:
                dest = base_dir / split / cls / img.name
                shutil.copy(img, dest)
    
    print("Dataset organized and split into train/val/test folders.")
    print("Excluded classes: 'Foreign', 'PseduoMembran', 'Earventulation', 'tympanoskleros', 'OtitExterna'")
    
    # Step 8: Show final counts
    print("\nImage count per class (after split):")
    for split in splits:
        print(f"\nSplit: {split}")
        for cls in classes:
            class_path = base_dir / split / cls
            count = len(list(class_path.glob('*')))
            print(f"  {cls}: {count} images")
    
    # Step 9: Create CSV files for easy loading
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Map class names to more readable names
    class_mapping = {
        'Aom': 'acute_otitis_media',
        'Chornic': 'chronic_otitis_media', 
        'Earwax': 'earwax',
        'Normal': 'normal'
    }
    
    # Create CSV file for each split
    for split in splits:
        image_paths = []
        labels = []
        
        for cls in classes:
            if cls in class_mapping:
                mapped_class = class_mapping[cls]
                class_path = base_dir / split / cls
                
                # Add each image and its label
                for img_file in class_path.glob("*"):
                    image_paths.append(str(img_file))
                    labels.append(mapped_class)
        
        # Save to CSV
        df = pd.DataFrame({
            'image_path': image_paths,
            'label': labels
        })
        
        csv_path = processed_data_dir / f"{split}_split.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} split to {csv_path} with {len(df)} images")
    
    # Step 10: Save metadata
    # Save class mapping
    class_mapping_df = pd.DataFrame([
        {'class_name': k, 'mapped_name': v} for k, v in class_mapping.items()
    ])
    class_mapping_df.to_csv(processed_data_dir / "class_mapping.csv", index=False)
    
    # Save dataset information
    dataset_info = {
        'total_classes': len(classes),
        'classes': classes,
        'excluded_classes': list(excluded_classes),
        'split_ratio': split_ratio,
        'base_dir': str(base_dir)
    }
    
    dataset_info_df = pd.DataFrame([dataset_info])
    dataset_info_df.to_csv(processed_data_dir / "dataset_info.csv", index=False)
    
    print("\nDataset setup completed successfully!")
    print(f"Dataset info saved to {processed_data_dir}")
    
    return processed_data_dir

if __name__ == "__main__":
    # When this script is run directly, set up the dataset
    setup_dataset()
