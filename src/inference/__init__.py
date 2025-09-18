"""
Inference Module

This module contains all inference-related functionality for the eardrum classification model.
It includes model export to ONNX format and ONNX Runtime inference engine.
"""

# Only import what's needed for inference to avoid PyTorch dependency in container
try:
    from .onnx_inference import ONNXInference
    __all__ = ['ONNXInference']
except ImportError:
    # If ONNX Runtime is not available, only export the model exporter
    try:
        from .model_exporter import export_to_onnx
        __all__ = ['export_to_onnx']
    except ImportError:
        __all__ = []
