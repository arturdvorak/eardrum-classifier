#!/usr/bin/env python3
"""
Evaluation Script for Eardrum Classification

This script handles:
1. Loading trained models from checkpoints
2. Evaluating on test dataset
3. Comparing performance across strategies
4. Generating confusion matrices and reports
5. Saving evaluation results
"""

import logging
import json
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import from the package
from eardrum_classifier import (
    EfficientNetV2Lightning, 
    EardrumDataset, 
    get_transforms, 
    setup_logging
)

def load_test_data():
    """Load test dataset."""
    logger = logging.getLogger(__name__)
    
    # Get transforms
    _, test_transform = get_transforms()
    
    # Load test dataset
    data_dir = Path("data/processed/eardrum_split")
    test_dataset = EardrumDataset(
        data_dir / "test",
        transform=test_transform
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Test data loaded: {len(test_dataset)} samples")
    
    return test_loader

def load_model_from_checkpoint(checkpoint_path):
    """Load a trained model from checkpoint."""
    logger = logging.getLogger(__name__)
    
    try:
        model = EfficientNetV2Lightning.load_from_checkpoint(
            checkpoint_path,
            num_classes=6
        )
        model.eval()
        logger.info(f"Model loaded from: {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise

def evaluate_model(model, test_loader, device):
    """Evaluate a single model on test data."""
    logger = logging.getLogger(__name__)
    
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    
    # Generate classification report
    class_names = ["Aom", "Chronic", "Earwax", "Normal", "Otitis Externa", "Tympanosclerosis"]
    report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results

def plot_confusion_matrix(cm, class_names, strategy_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {strategy_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_strategies():
    """Evaluate all training strategies."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evaluation of all strategies...")
    
    try:
        # Setup MLflow
        mlflow.set_tracking_uri("file:logs/mlruns")
        mlflow.start_run(run_name="eardrum_classification_evaluation")
        
        # Load test data
        test_loader = load_test_data()
        
        # Load best checkpoints info
        checkpoints_file = Path("logs/best_checkpoints.json")
        if not checkpoints_file.exists():
            logger.error("Best checkpoints file not found. Run training first.")
            return
        
        with open(checkpoints_file, 'r') as f:
            best_checkpoints = json.load(f)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Evaluate each strategy
        all_results = {}
        strategy_comparison = []
        
        for strategy, checkpoint_path in best_checkpoints.items():
            logger.info(f"Evaluating strategy: {strategy}")
            
            try:
                # Load model
                model = load_model_from_checkpoint(checkpoint_path)
                
                # Evaluate model
                results = evaluate_model(model, test_loader, device)
                all_results[strategy] = results
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    f"{strategy}_test_accuracy": results['accuracy'],
                    f"{strategy}_test_f1": results['f1'],
                    f"{strategy}_test_precision": results['precision'],
                    f"{strategy}_test_recall": results['recall']
                })
                
                # Save confusion matrix plot
                plot_path = f"logs/confusion_matrix_{strategy}.png"
                plot_confusion_matrix(
                    results['confusion_matrix'],
                    results['class_names'],
                    strategy,
                    plot_path
                )
                
                # Add to comparison table
                strategy_comparison.append({
                    'Strategy': strategy,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision': f"{results['precision']:.4f}",
                    'Recall': f"{results['recall']:.4f}",
                    'F1-Score': f"{results['f1']:.4f}"
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate strategy {strategy}: {e}")
                continue
        
        # Create comparison table
        if strategy_comparison:
            comparison_df = pd.DataFrame(strategy_comparison)
            comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
            
            # Save comparison table
            comparison_path = "logs/strategy_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            
            # Display results
            logger.info("\n" + "="*60)
            logger.info("STRATEGY COMPARISON RESULTS")
            logger.info("="*60)
            logger.info(comparison_df.to_string(index=False))
            logger.info("="*60)
            
            # Save detailed results
            save_detailed_results(all_results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        mlflow.end_run()

def save_detailed_results(all_results):
    """Save detailed evaluation results."""
    logger = logging.getLogger(__name__)
    
    # Save results summary
    summary = {}
    for strategy, results in all_results.items():
        summary[strategy] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1']
        }
    
    # Save to JSON
    with open("logs/evaluation_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Detailed results saved to logs/evaluation_results.json")

if __name__ == "__main__":
    setup_logging()
    evaluate_all_strategies()
