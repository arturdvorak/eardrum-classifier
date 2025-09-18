#!/usr/bin/env python3
"""
Eardrum Classification Project - Complete Pipeline

This is the comprehensive entry point script that runs the complete machine learning pipeline.
It handles dataset download, preprocessing, training, and evaluation in one command.

Pipeline Steps:
1. Dataset Setup: Download and organize the eardrum dataset from Kaggle
2. Data Preprocessing: Split into train/val/test and apply transformations
3. Model Training: Train EfficientNetV2 with multiple fine-tuning strategies
4. Model Evaluation: Test all models and find the best performing one

Usage:
    python main.py                    # Run complete pipeline
    python main.py --step data        # Only download and setup data
    python main.py --step train       # Only train models
L    # Only evaluate models
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Import our custom modules
from src.data.dataset_setup import setup_dataset
from src.training.trainer import train_all_strategies
from src.evaluation.evaluator import evaluate_all_strategies

def main():
    """
    Main function that orchestrates the complete ML pipeline.
    
    This function:
    1. Parses command line arguments
    2. Runs the requested pipeline steps in order
    3. Provides clear progress updates
    4. Handles errors gracefully
    5. Reports final results
    """
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(
        description="Eardrum Classification Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run complete pipeline
    python main.py --step data        # Only setup dataset
    python main.py --step train       # Only train models
    python main.py --step evaluate    # Only evaluate models
    python main.py --verbose          # Show detailed output
        """
    )
    
    # Pipeline step selection
    parser.add_argument(
        "--step", 
        choices=["data", "train", "evaluate", "all"], 
        default="all", 
        help="Which pipeline step to run (default: all)"
    )
    
    # Verbosity control
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Show detailed output and debug information"
    )
    
    # Training epochs control
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10, use 1 for debugging)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Start pipeline
    print("=" * 70)
    print("EARDRUM CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"Running step: {args.step}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Dataset Setup (Download & Preprocessing)
        if args.step in ["data", "all"]:
            print("STEP 1: DATASET SETUP")
            print("-" * 50)
            print("• Downloading dataset from Kaggle...")
            print("• Extracting and organizing files...")
            print("• Splitting into train/validation/test sets...")
            print("• Creating CSV files for data loading...")
            
            # Run dataset setup
            processed_data_dir = setup_dataset()
            
            print(f"Dataset setup completed successfully!")
            print(f"   Data saved to: {processed_data_dir}")
            print()
        
        # Step 2: Model Training
        if args.step in ["train", "all"]:
            print("STEP 2: MODEL TRAINING")
            print("-" * 50)
            print("• Loading dataset and creating data loaders...")
            print("• Training with 6 different fine-tuning strategies:")
            print("  - freeze_backbone: Only train classifier")
            print("  - last1+head: Train last block + classifier")
            print("  - last2+head: Train last 2 blocks + classifier")
            print("  - last3+head: Train last 3 blocks + classifier")
            print("  - last4+head: Train last 4 blocks + classifier")
            print("  - full: Train entire model")
            print("• Using automatic learning rate tuning...")
            print("• Tracking experiments with MLflow...")
            
            # Run training
            best_checkpoints = train_all_strategies(epochs=args.epochs)
            
            print(f"Model training completed successfully!")
            print(f"   Trained {len(best_checkpoints)} strategies")
            print(f"   Checkpoints saved to: checkpoints/")
            print()
        
        # Step 3: Model Evaluation
        if args.step in ["evaluate", "all"]:
            print("STEP 3: MODEL EVALUATION")
            print("-" * 50)
            print("• Loading test dataset...")
            print("• Evaluating all trained models...")
            print("• Calculating performance metrics...")
            print("• Creating confusion matrices...")
            print("• Logging results to MLflow...")
            
            # Run evaluation
            results = evaluate_all_strategies()
            
            print(f"Model evaluation completed successfully!")
            if results:
                best_result = max(results, key=lambda x: x['f1_macro'])
                print(f"   Best strategy: {best_result['strategy']}")
                print(f"   Best F1 Score: {best_result['f1_macro']:.4f}")
                print(f"   Best Accuracy: {best_result['accuracy']:.4f}")
            print()
        
        # Pipeline completion
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total execution time: {duration/60:.1f} minutes")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Results:")
        print(f"   - Dataset: data/eardrum_split/ (train/val/test)")
        print(f"   - Models: checkpoints/ (best models saved)")
        print(f"   - Experiments: mlruns/ (MLflow tracking)")
        print()
        print("Next steps:")
        print(f"   - View results: mlflow ui")
        print(f"   - Use web app: python web/app.py")
        print(f"   - Analyze data: jupyter notebook notebooks/")
        print("=" * 70)
        
        return 0
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Solution: Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
        
    except FileNotFoundError as e:
        print(f"File Not Found: {e}")
        print("Solution: Make sure you're in the project root directory")
        return 1
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print()
        print("Debugging information:")
        print(f"   - Step that failed: {args.step}")
        print(f"   - Error type: {type(e).__name__}")
        if args.verbose:
            print(f"   - Full traceback:")
            traceback.print_exc()
        print()
        print("Common solutions:")
        print("   - Check internet connection (for dataset download)")
        print("   - Ensure sufficient disk space (>5GB needed)")
        print("   - Verify Kaggle API credentials are set up")
        print("   - Try running individual steps: --step data, --step train, --step evaluate")
        
        return 1

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This ensures proper exit codes are returned to the system.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user (Ctrl+C)")
        print("You can resume by running individual steps:")
        print("   python main.py --step train     # If data setup was completed")
        print("   python main.py --step evaluate  # If training was completed")
        sys.exit(1)
