"""
ONNX Runtime Inference Engine

This module provides a high-performance inference engine using ONNX Runtime
for the eardrum classification model.
"""

import os
import json
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

# Optional PyTorch imports for preprocessing
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback transforms using PIL and numpy
    from PIL import Image as PILImage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXInference:
    """
    ONNX Runtime inference engine for eardrum classification.
    
    This class provides:
    - Model loading and initialization
    - Image preprocessing
    - Single and batch inference
    - Performance monitoring
    """
    
    def __init__(
        self, 
        model_path: str, 
        providers: Optional[List[str]] = None,
        metadata_path: Optional[str] = None
    ):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path (str): Path to the ONNX model file
            providers (List[str]): ONNX Runtime providers (default: CPU)
            metadata_path (str): Path to model metadata JSON file
        """
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        self.metadata = self._load_metadata(metadata_path)
        
        # Initialize ONNX Runtime session
        self.session = self._initialize_session()
        
        # Set up image preprocessing
        self.transform = self._setup_preprocessing()
        
        # Class names
        self.class_names = self.metadata.get('class_names', ['Aom', 'Earwax', 'Normal', 'Chornic'])
        
        logger.info(f"ONNX Inference engine initialized with {len(self.class_names)} classes")
        logger.info(f"Model: {model_path}")
        logger.info(f"Providers: {self.providers}")
    
    def _load_metadata(self, metadata_path: Optional[str]) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        if metadata_path is None:
            # Try to find metadata file automatically
            metadata_path = self.model_path.replace('.onnx', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_path}")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        # Default metadata
        return {
            'class_names': ['Aom', 'Earwax', 'Normal', 'Chornic'],
            'input_shape': [1, 3, 224, 224],
            'opset_version': 11
        }
    
    def _initialize_session(self) -> ort.InferenceSession:
        """Initialize ONNX Runtime session with optimizations."""
        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = True
            
            # Create session
            session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            logger.info("ONNX Runtime session initialized successfully")
            return session
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise
    
    def _setup_preprocessing(self):
        """Set up image preprocessing pipeline."""
        if TORCH_AVAILABLE:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Return a simple function for PIL-based preprocessing
            return self._pil_preprocess
    
    def _pil_preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL-based preprocessing without PyTorch."""
        # Resize image
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize
        img_array = (img_array - mean) / std
        
        # Convert to CHW format (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            
            # Apply preprocessing
            if TORCH_AVAILABLE:
                tensor = self.transform(image)
                # Add batch dimension
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                return tensor.numpy()
            else:
                # Use PIL-based preprocessing
                tensor = self.transform(image)
                # Add batch dimension
                return np.expand_dims(tensor, axis=0)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def predict_single(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(None, {'input': input_tensor})
            logits = outputs[0]
            
            # Convert to probabilities
            if TORCH_AVAILABLE:
                probabilities = torch.softmax(torch.tensor(logits), dim=1)
                # Get prediction
                prediction_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction_idx].item()
            else:
                # Use numpy for softmax
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                # Get prediction
                prediction_idx = np.argmax(probabilities, axis=1)[0]
                confidence = probabilities[0][prediction_idx]
            
            # Create result
            if TORCH_AVAILABLE:
                prob_dict = {
                    name: prob.item() 
                    for name, prob in zip(self.class_names, probabilities[0])
                }
            else:
                prob_dict = {
                    name: float(prob) 
                    for name, prob in zip(self.class_names, probabilities[0])
                }
            
            result = {
                'prediction': self.class_names[prediction_idx],
                'prediction_idx': int(prediction_idx),
                'confidence': float(confidence),
                'probabilities': prob_dict
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        try:
            # Preprocess all images
            input_tensors = []
            for image in images:
                tensor = self.preprocess_image(image)
                input_tensors.append(tensor)
            
            # Stack tensors for batch processing
            batch_tensor = np.vstack(input_tensors)
            
            # Run batch inference
            outputs = self.session.run(None, {'input': batch_tensor})
            logits = outputs[0]
            
            # Convert to probabilities
            if TORCH_AVAILABLE:
                probabilities = torch.softmax(torch.tensor(logits), dim=1)
            else:
                # Use numpy for softmax
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Process each prediction
            results = []
            for i in range(len(images)):
                if TORCH_AVAILABLE:
                    pred_idx = torch.argmax(probabilities[i], dim=0).item()
                    confidence = probabilities[i][pred_idx].item()
                    prob_dict = {
                        name: prob.item() 
                        for name, prob in zip(self.class_names, probabilities[i])
                    }
                else:
                    pred_idx = np.argmax(probabilities[i], axis=0)
                    confidence = probabilities[i][pred_idx]
                    prob_dict = {
                        name: float(prob) 
                        for name, prob in zip(self.class_names, probabilities[i])
                    }
                
                result = {
                    'prediction': self.class_names[pred_idx],
                    'prediction_idx': int(pred_idx),
                    'confidence': float(confidence),
                    'probabilities': prob_dict
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance stats."""
        try:
            # Get input/output info
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            
            # Get session providers
            providers = self.session.get_providers()
            
            return {
                'model_path': self.model_path,
                'input_shape': input_info.shape,
                'input_type': input_info.type,
                'output_shape': output_info.shape,
                'output_type': output_info.type,
                'providers': providers,
                'class_names': self.class_names,
                'metadata': self.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark the model performance.
        
        Args:
            num_runs (int): Number of runs for benchmarking
            
        Returns:
            Dict containing performance metrics
        """
        try:
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                self.session.run(None, {'input': dummy_input})
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                self.session.run(None, {'input': dummy_input})
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            
            return {
                'mean_inference_time': float(np.mean(times)),
                'std_inference_time': float(np.std(times)),
                'min_inference_time': float(np.min(times)),
                'max_inference_time': float(np.max(times)),
                'throughput_fps': float(1.0 / np.mean(times)),
                'num_runs': num_runs
            }
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {'error': str(e)}

# Convenience function for easy usage
def create_inference_engine(
    model_path: str, 
    use_gpu: bool = False
) -> ONNXInference:
    """
    Create an ONNX inference engine with optimal settings.
    
    Args:
        model_path (str): Path to ONNX model
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        ONNXInference: Configured inference engine
    """
    providers = ['CPUExecutionProvider']
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logger.info("GPU acceleration enabled")
    
    return ONNXInference(model_path, providers=providers)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Inference Engine")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = create_inference_engine(args.model, use_gpu=args.gpu)
    
    # Print model info
    info = engine.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with image if provided
    if args.image:
        result = engine.predict_single(args.image)
        print(f"\nPrediction: {result}")
    
    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        stats = engine.benchmark()
        print("Performance Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
