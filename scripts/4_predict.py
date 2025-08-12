#!/usr/bin/env python3
"""
Prediction Script for Eardrum Classification

This script handles:
1. Single image prediction
2. Batch prediction on multiple images
3. Loading best model from checkpoint
4. Saving prediction results
"""

import argparse
import logging
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

# Import from the package
from eardrum_classifier import EfficientNetV2Lightning, get_transforms, setup_logging

class EardrumPredictor:
    """Class for making predictions on eardrum images."""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Initialize predictor with a trained model."""
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        _, self.transform = get_transforms()
        
        # Class names
        self.class_names = [
            "Aom", "Chronic", "Earwax", "Normal", 
            "Otitis Externa", "Tympanosclerosis"
        ]
        
        self.logger.info("Predictor initialized successfully")
    
    def _load_model(self, checkpoint_path: str) -> EfficientNetV2Lightning:
        """Load model from checkpoint."""
        try:
            model = EfficientNetV2Lightning.load_from_checkpoint(
                checkpoint_path,
                num_classes=6
            )
            self.logger.info(f"Model loaded from: {checkpoint_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, image_path: str) -> Dict:
        """Predict on a single image."""
        self.logger.info(f"Predicting on: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
            
            # Get results
            predicted_class = self.class_names[prediction.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            class_probabilities = {
                class_name: prob.item() 
                for class_name, prob in zip(self.class_names, probabilities[0])
            }
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'class_probabilities': class_probabilities,
                'prediction_rankings': self._get_prediction_rankings(probabilities[0])
            }
            
            self.logger.info(f"Prediction: {predicted_class} (confidence: {confidence_score:.4f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {image_path}: {e}")
            raise
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict on multiple images."""
        self.logger.info(f"Predicting on {len(image_paths)} images")
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Skipping {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def _get_prediction_rankings(self, probabilities: torch.Tensor) -> List[Tuple[str, float]]:
        """Get class predictions ranked by probability."""
        class_probs = [(self.class_names[i], prob.item()) for i, prob in enumerate(probabilities)]
        return sorted(class_probs, key=lambda x: x[1], reverse=True)
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save prediction results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            self.logger.info(f"Predictions saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {e}")
            raise

def find_best_checkpoint() -> str:
    """Find the best performing checkpoint."""
    checkpoints_file = Path("logs/best_checkpoints.json")
    
    if not checkpoints_file.exists():
        raise FileNotFoundError("Best checkpoints file not found. Run training first.")
    
    with open(checkpoints_file, 'r') as f:
        best_checkpoints = json.load(f)
    
    # Find strategy with highest F1 score (you might want to load from evaluation results)
    # For now, return the first available checkpoint
    if best_checkpoints:
        return list(best_checkpoints.values())[0]
    else:
        raise ValueError("No checkpoints found")

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Eardrum Classification Prediction")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to multiple images")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="predictions.json", 
                       help="Output file for predictions")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.image and not args.images:
        parser.error("Either --image or --images must be specified")
    
    try:
        # Find checkpoint if not specified
        if not args.checkpoint:
            args.checkpoint = find_best_checkpoint()
            logger.info(f"Using best checkpoint: {args.checkpoint}")
        
        # Initialize predictor
        predictor = EardrumPredictor(args.checkpoint, args.device)
        
        # Make predictions
        if args.image:
            # Single image prediction
            result = predictor.predict_single(args.image)
            predictions = [result]
            
        else:
            # Batch prediction
            predictions = predictor.predict_batch(args.images)
        
        # Save results
        predictor.save_predictions(predictions, args.output)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("PREDICTION RESULTS")
        logger.info("="*50)
        
        for pred in predictions:
            if 'error' in pred:
                logger.info(f"❌ {pred['image_path']}: {pred['error']}")
            else:
                logger.info(f"✅ {pred['image_path']}: {pred['predicted_class']} "
                          f"(confidence: {pred['confidence']:.4f})")
        
        logger.info("="*50)
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
