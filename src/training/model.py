"""
Model Definition Module

This module defines the EfficientNetV2 model wrapped in PyTorch Lightning.
It includes F1 scoring and proper training/validation/test steps.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import F1Score

class EfficientNetV2Lightning(pl.LightningModule):
    """
    EfficientNetV2 model wrapped in PyTorch Lightning.
    
    This class:
    1. Loads a pre-trained EfficientNetV2 model
    2. Implements training, validation, and test steps
    3. Uses F1 score as the main metric
    4. Handles optimization automatically
    """
    
    def __init__(self, num_classes, lr=1e-4):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            lr (float): Learning rate (default: 1e-4)
        """
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Create the EfficientNetV2 model
        # Using a pre-trained model from ImageNet-21K
        self.model = timm.create_model(
            "tf_efficientnetv2_s.in21k", 
            pretrained=True, 
            num_classes=num_classes
        )
        
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Define metrics for validation and testing
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Model predictions
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the batch
            
        Returns:
            torch.Tensor: Training loss
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Log training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the batch
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Get predictions
        preds = logits.argmax(dim=1)
        
        # Calculate F1 score
        f1 = self.val_f1(preds, y)
        
        # Log metrics
        self.log("val_loss", loss)
        self.log("val_f1", f1, prog_bar=True)  # Show in progress bar

    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the batch
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Get predictions
        preds = logits.argmax(dim=1)
        
        # Calculate F1 score
        f1 = self.test_f1(preds, y)
        
        # Log metrics
        self.log("test_loss", loss)
        self.log("test_f1", f1, prog_bar=True)  # Show in progress bar

    def configure_optimizers(self):
        """
        Configure the optimizer.
        
        Returns:
            torch.optim.Optimizer: The optimizer to use
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    # Test the model
    print("Testing EfficientNetV2 model...")
    
    # Create a small test model
    model = EfficientNetV2Lightning(num_classes=4)
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
