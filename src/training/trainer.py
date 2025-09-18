"""
Main Trainer Module

This module handles training all fine-tuning strategies with learning rate tuning.
It uses PyTorch Lightning for training and MLflow for experiment tracking.
"""

import json
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.tuner import Tuner
import mlflow

# Import our custom modules
from .model import EfficientNetV2Lightning
from .strategies import set_finetune_strategy, get_available_strategies
from ..data.data_loader import get_dataloaders

def train_all_strategies(epochs=10):
    """
    Train models with all fine-tuning strategies.
    
    This function:
    1. Loads the dataset
    2. Finds optimal learning rate for each strategy using Tuner.lr_find()
    3. Trains each strategy with the optimized learning rate
    4. Saves checkpoints and logs results
    5. Tracks experiments with MLflow
    """
    
    # Set random seed for reproducible results
    pl.seed_everything(42, workers=True)
    
    print("Loading dataset...")
    # Get data loaders with optimized settings
    # Use larger batch size for better GPU utilization
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
        batch_size=batch_size
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print()
    
    # Get all available strategies
    strategies = get_available_strategies()
    num_classes = len(train_dataset.classes)
    
    # Dictionary to store best checkpoint paths
    best_ckpt_paths = {}
    
    print(f"Training {len(strategies)} strategies...")
    print("=" * 60)
    
    # Train each strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"\nStrategy {i}/{len(strategies)}: {strategy}")
        print("-" * 40)
        
        # Create a new model for this strategy
        model = EfficientNetV2Lightning(num_classes=num_classes)
        
        # Apply the fine-tuning strategy
        set_finetune_strategy(model, strategy)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Set up MLflow logging
        mlflow_logger = MLFlowLogger(
            experiment_name="eardrum_finetune_strategies",
            tracking_uri="file:mlruns",
            run_name=strategy
        )
        
        # Set up callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", 
            patience=3, 
            mode="min"
        )
        
        checkpoint = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            filename=f"{strategy}-best-f1-{{epoch:02d}}-{{val_f1:.4f}}"
        )
        
        # Determine precision and device settings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = "16-mixed" if device == "cuda" else 32
        print(f"Using device: {device}, precision: {precision}")
        
        # Create trainer with optimized settings
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            precision=precision,
            callbacks=[early_stop, checkpoint],
            logger=mlflow_logger,
            default_root_dir="checkpoints",  # Save all temporary files in checkpoints folder
            gradient_clip_val=1.0,  # Gradient clipping for stability
            accumulate_grad_batches=1,  # No gradient accumulation for speed
            log_every_n_steps=10,  # Log more frequently for better monitoring
            val_check_interval=0.5,  # Validate twice per epoch
            enable_progress_bar=True,
            enable_model_summary=True,
            fast_dev_run=False,  # Set to True for quick testing
            # Memory optimizations
            strategy="auto",  # Let Lightning choose the best strategy
            sync_batchnorm=False,  # Disable for single GPU
            # Performance optimizations
            benchmark=True if device == "cuda" else False,  # Enable cudnn benchmark
            deterministic=False,  # Allow non-deterministic for speed
        )
        
        # Find optimal learning rate using PyTorch Lightning Tuner
        print("Finding optimal learning rate...")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        new_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {new_lr:.2e}")
        model.hparams.lr = new_lr
        
        # Train the model
        print("Training model...")
        trainer.fit(model, train_loader, val_loader)
        
        # Save best checkpoint path
        ckpt_path = checkpoint.best_model_path
        print(f"Best checkpoint saved at: {ckpt_path}")
        best_ckpt_paths[strategy] = ckpt_path
        
        # Log checkpoint and model to MLflow (temporarily disabled for Docker compatibility)
        # TODO: Fix MLflow artifact logging in Docker environment
        # mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, ckpt_path)
        
        # Log model to MLflow (temporarily disabled for Docker compatibility)
        # example_input = torch.randn(1, 3, 224, 224).numpy()
        # with mlflow.start_run(run_id=mlflow_logger.run_id):
        #     mlflow.pytorch.log_model(
        #         model,
        #         artifact_path="model",
        #         input_example=example_input
        #     )
        
        print(f"Strategy '{strategy}' completed successfully!")
    
    # Save best checkpoint paths to JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = f"checkpoints/best_checkpoints_{timestamp}.json"
    
    with open(json_path, "w") as f:
        json.dump(best_ckpt_paths, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best checkpoint paths saved to: {json_path}")
    print("\nCheckpoint paths:")
    for strategy, path in best_ckpt_paths.items():
        print(f"  {strategy}: {path}")
    
    return best_ckpt_paths

if __name__ == "__main__":
    # When this script is run directly, train all strategies
    train_all_strategies()
