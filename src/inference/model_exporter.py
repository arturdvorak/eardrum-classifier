"""
Model Export Module

This module handles exporting PyTorch models to ONNX format for optimized inference.
It includes validation and optimization for production deployment.
"""

import os
import torch
import onnx
import json
from pathlib import Path
from typing import Optional, Tuple
import logging

# Import the model class
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.training.model import EfficientNetV2Lightning

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_best_model(checkpoint_path: str, num_classes: int = 4) -> EfficientNetV2Lightning:
    """
    Load the best trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file
        num_classes (int): Number of output classes (default: 4)
        
    Returns:
        EfficientNetV2Lightning: Loaded model in evaluation mode
    """
    try:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = EfficientNetV2Lightning.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes
        )
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 11,
    optimize: bool = True
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file
        output_path (str): Path where to save the ONNX model
        input_shape (Tuple): Input shape (batch_size, channels, height, width)
        opset_version (int): ONNX opset version (default: 11)
        optimize (bool): Whether to optimize the ONNX model
        
    Returns:
        str: Path to the exported ONNX model
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load the model
        model = load_best_model(checkpoint_path)
        
        # Create dummy input tensor
        dummy_input = torch.randn(input_shape)
        logger.info(f"Created dummy input with shape: {dummy_input.shape}")
        
        # Export to ONNX
        logger.info("Exporting model to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        logger.info(f"Model exported to: {output_path}")
        
        # Validate the ONNX model
        logger.info("Validating ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation successful")
        
        # Optimize the model if requested
        if optimize:
            logger.info("Optimizing ONNX model...")
            try:
                # Try the new optimization method
                from onnxruntime.transformers import optimizer
                optimized_model = optimizer.optimize_model(output_path)
                optimized_output_path = output_path.replace('.onnx', '_optimized.onnx')
                optimized_model.save_model_to_file(optimized_output_path)
                logger.info(f"Optimized model saved to: {optimized_output_path}")
            except ImportError:
                logger.warning("ONNX optimization not available, skipping optimization")
                logger.info("Model exported without optimization")
        
        # Save model metadata
        metadata = {
            'input_shape': input_shape,
            'opset_version': opset_version,
            'optimized': optimize,
            'checkpoint_path': checkpoint_path,
            'export_timestamp': str(torch.datetime.now() if hasattr(torch, 'datetime') else '2024-01-01'),
            'class_names': ['Aom', 'Earwax', 'Normal', 'Chornic']
        }
        
        metadata_path = output_path.replace('.onnx', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to: {metadata_path}")
        logger.info("ONNX export completed successfully!")
        
        return output_path
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise

def export_best_models_from_checkpoints(
    checkpoints_file: str,
    output_dir: str = "models/onnx"
) -> dict:
    """
    Export all best models from the checkpoints JSON file.
    
    Args:
        checkpoints_file (str): Path to the checkpoints JSON file
        output_dir (str): Directory to save ONNX models
        
    Returns:
        dict: Mapping of strategy names to ONNX model paths
    """
    try:
        # Load checkpoints
        with open(checkpoints_file, 'r') as f:
            checkpoints = json.load(f)
        
        logger.info(f"Found {len(checkpoints)} model strategies to export")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        onnx_models = {}
        
        for strategy, checkpoint_path in checkpoints.items():
            logger.info(f"Exporting strategy: {strategy}")
            
            # Create output filename
            onnx_filename = f"eardrum_classifier_{strategy}.onnx"
            onnx_path = os.path.join(output_dir, onnx_filename)
            
            # Export to ONNX
            try:
                export_to_onnx(checkpoint_path, onnx_path)
                onnx_models[strategy] = onnx_path
                logger.info(f"Successfully exported {strategy}")
            except Exception as e:
                logger.error(f"Failed to export {strategy}: {e}")
                continue
        
        # Save the mapping
        mapping_file = os.path.join(output_dir, "model_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(onnx_models, f, indent=2)
        
        logger.info(f"Exported {len(onnx_models)} models to {output_dir}")
        logger.info(f"Model mapping saved to: {mapping_file}")
        
        return onnx_models
        
    except Exception as e:
        logger.error(f"Batch export failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    
    args = parser.parse_args()
    
    # Export single model
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=(args.batch_size, 3, 224, 224),
        optimize=args.optimize
    )
