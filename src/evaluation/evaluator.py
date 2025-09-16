"""
Model Evaluation Module

This module evaluates all trained models on the test dataset.
It calculates comprehensive metrics and visualizes results.
"""

import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import mlflow

# Import our custom modules
from ..training.model import EfficientNetV2Lightning

def load_test_dataset():
    """
    Load the test dataset with proper transformations.
    
    Returns:
        tuple: (test_dataset, test_loader)
    """
    # Define transformations for testing (same as validation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                    # Resize to square
        transforms.ToTensor(),                            # Convert to tensor
        transforms.Normalize(                              # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root='data/eardrum_split/test', 
        transform=test_transform
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    return test_dataset, test_loader

def evaluate_model(strategy, ckpt_path):
    """
    Evaluate a single model strategy.
    
    This function:
    1. Loads the model from checkpoint
    2. Runs inference on test data
    3. Calculates performance metrics
    4. Creates visualizations
    5. Logs results to MLflow
    
    Args:
        strategy (str): Name of the training strategy
        ckpt_path (str): Path to the model checkpoint
    """
    
    print(f"\nEvaluating strategy: {strategy}")
    print("=" * 50)
    
    # Determine device to use
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {ckpt_path}")
    
    # Load test dataset
    test_dataset, test_loader = load_test_dataset()
    
    # Load model from checkpoint
    model = EfficientNetV2Lightning.load_from_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and labels
    all_preds, all_labels = [], []
    
    print("Running inference on test dataset...")
    
    # Run inference
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Get model predictions
            logits = model(x)
            preds = logits.argmax(dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Show progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Always use fixed label list for consistent evaluation
    labels = list(range(len(test_dataset.classes)))
    
    print("Calculating metrics...")
    
    # Compute performance metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, labels=labels, average=None, zero_division=0)
    
    # Display results
    print(f"\nResults for {strategy}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score (macro): {f1_macro:.4f}")
    
    print(f"\nF1 Score per class:")
    for idx, score in enumerate(f1_per_class):
        print(f"  {test_dataset.classes[idx]}: {score:.4f}")
    
    # Generate classification report
    print(f"\nClassification Report:")
    report_txt = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=test_dataset.classes,
        zero_division=0
    )
    print(report_txt)
    
    # Create confusion matrix visualization
    print("Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {strategy}")
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = f"confusion_matrix_{strategy}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Log results to MLflow
    print("Logging results to MLflow...")
    mlflow.set_tracking_uri("file://mlruns")
    mlflow.set_experiment("eardrum_finetune_strategies")
    
    with mlflow.start_run(run_name=f"eval_{strategy}", nested=True):
        # Log parameters
        mlflow.log_param("strategy", strategy)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_macro", f1_macro)
        
        # Log F1 scores per class
        for idx, score in enumerate(f1_per_class):
            class_metric_name = f"class_{test_dataset.classes[idx]}_f1"
            mlflow.log_metric(class_metric_name, score)
        
        # Save and log classification report
        report_path = f"classification_report_{strategy}.txt"
        with open(report_path, "w") as f:
            f.write(report_txt)
        mlflow.log_artifact(report_path)
        
        # Log confusion matrix
        mlflow.log_artifact(cm_path)
        
        # Log model checkpoint used for evaluation
        mlflow.log_artifact(ckpt_path)
    
    print(f"Evaluation of {strategy} completed successfully!")
    
    # Clean up temporary files
    os.remove(report_path)
    os.remove(cm_path)
    
    return {
        'strategy': strategy,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class
    }

def evaluate_all_strategies():
    """
    Evaluate all trained model strategies.
    
    This function:
    1. Finds the latest checkpoint file
    2. Loads all checkpoint paths
    3. Evaluates each strategy
    4. Shows summary of results
    """
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Find the latest checkpoint file
    checkpoint_files = glob.glob("checkpoints/best_checkpoints_*.json")
    if not checkpoint_files:
        raise FileNotFoundError("No best_checkpoints_*.json files found in checkpoints/ folder.")
    
    # Get the most recent checkpoint file
    latest_ckpt_file = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading latest checkpoint file: {latest_ckpt_file}")
    
    # Load checkpoint paths
    with open(latest_ckpt_file, "r") as f:
        best_ckpt_paths = json.load(f)
    
    print(f"Found {len(best_ckpt_paths)} strategies to evaluate")
    print()
    
    # Evaluate each strategy
    results = []
    for strategy, ckpt_path in best_ckpt_paths.items():
        try:
            result = evaluate_model(strategy, ckpt_path)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {strategy}: {e}")
            continue
    
    # Show summary of all results
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if results:
        # Sort by F1 score
        results.sort(key=lambda x: x['f1_macro'], reverse=True)
        
        print(f"\nRanked by F1 Score (Macro):")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['strategy']:15} - F1: {result['f1_macro']:.4f}, Acc: {result['accuracy']:.4f}")
        
        # Find best strategy
        best_result = results[0]
        print(f"\nBest Strategy: {best_result['strategy']}")
        print(f"  F1 Score: {best_result['f1_macro']:.4f}")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  Recall: {best_result['recall']:.4f}")
    
    print(f"\nEvaluation completed successfully!")
    return results

if __name__ == "__main__":
    # When this script is run directly, evaluate all strategies
    evaluate_all_strategies()
