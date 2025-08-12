"""
Training strategies for fine-tuning the EfficientNetV2 model.
"""

import torch
import torch.nn as nn

class TrainingStrategy:
    """Base class for training strategies."""
    
    def apply(self, model):
        """Apply the strategy to the model."""
        raise NotImplementedError

class FreezeBackboneStrategy(TrainingStrategy):
    """Freeze backbone, only train classification head."""
    
    def apply(self, model):
        """Freeze all layers except the final classification layer."""
        for param in model.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classification layer
        for param in model.model.classifier.parameters():
            param.requires_grad = True
        
        return model

class LastNBlocksStrategy(TrainingStrategy):
    """Unfreeze last N blocks of the model."""
    
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
    
    def apply(self, model):
        """Unfreeze last N blocks."""
        # Freeze all layers first
        for param in model.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        # This is a simplified approach - you might need to adjust based on actual model structure
        if hasattr(model.model, 'blocks'):
            total_blocks = len(model.model.blocks)
            for i in range(total_blocks - self.num_blocks, total_blocks):
                for param in model.model.blocks[i].parameters():
                    param.requires_grad = True
        
        # Always unfreeze classifier
        for param in model.model.classifier.parameters():
            param.requires_grad = True
        
        return model

class FullTrainingStrategy(TrainingStrategy):
    """Train all layers."""
    
    def apply(self, model):
        """Unfreeze all layers."""
        for param in model.model.parameters():
            param.requires_grad = True
        return model

def get_training_strategies():
    """Get all available training strategies."""
    return {
        "freeze_backbone": FreezeBackboneStrategy(),
        "last1+head": LastNBlocksStrategy(1),
        "last2+head": LastNBlocksStrategy(2),
        "last3+head": LastNBlocksStrategy(3),
        "last4+head": LastNBlocksStrategy(4),
        "full": FullTrainingStrategy()
    }
