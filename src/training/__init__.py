"""
Training module for the eardrum classification project.

This module contains:
- Model definition (EfficientNetV2)
- Training strategies
- Training loop with PyTorch Lightning
"""

from .strategies import get_available_strategies

__all__ = ["get_available_strategies"]
