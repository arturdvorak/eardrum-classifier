"""
Dataset handling for eardrum classification.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os

class EardrumDataset(Dataset):
    """Custom dataset for eardrum images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the data directory
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        self.labels = []
        
        # Map class names to indices
        self.class_to_idx = {
            "Aom": 0,
            "Chronic": 1, 
            "Earwax": 2,
            "Normal": 3,
            "Otitis Externa": 4,
            "Tympanosclerosis": 5
        }
        
        # Load file paths and labels
        for class_name in self.class_to_idx.keys():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    self.image_files.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.image_files)} images from {data_dir}")
    
    def __len__(self):
        """Return the number of images."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single image and label."""
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
