"""
Eardrum Classification Package

This package contains all the core functionality for eardrum classification:
- Models: Neural network architectures
- Data: Dataset handling and transforms
- Training: Training strategies and utilities
- Utils: Helper functions and logging
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from .models.efficientnet import EfficientNetV2Lightning
from .data.dataset import EardrumDataset
from .data.transforms import get_transforms
from .data.data_setup import setup_dataset
from .training.strategies import get_training_strategies
from .training.trainer import train_models
from .training.evaluator import evaluate_models
from .utils.logging_config import setup_logging

__all__ = [
    "EfficientNetV2Lightning",
    "EardrumDataset", 
    "get_transforms",
    "setup_dataset",
    "get_training_strategies",
    "train_models",
    "evaluate_models",
    "setup_logging"
]
