#!/usr/bin/env python3
"""
Training Script for Eardrum Classification

This script handles:
1. Loading and preparing data
2. Training models with different fine-tuning strategies
3. Model checkpointing and early stopping
4. MLflow experiment tracking
"""

import logging
import mlflow
import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import from the package
from eardrum_classifier import (
    EfficientNetV2Lightning, 
    EardrumDataset, 
    get_transforms, 
    get_training_strategies, 
    setup_logging
)

def setup_mlflow():
    """Setup MLflow tracking."""
    logger = logging.getLogger(__name__)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:logs/mlruns")
    
    # Start MLflow run
    mlflow.start_run(run_name="eardrum_classification_training")
    
    logger.info("MLflow tracking initialized")

def load_data():
    """Load and prepare data loaders."""
    logger = logging.getLogger(__name__)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    data_dir = Path("data/processed/eardrum_split")
    
    train_dataset = EardrumDataset(
        data_dir / "train",
        transform=train_transform
    )
    
    val_dataset = EardrumDataset(
        data_dir / "val", 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Data loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    return train_loader, val_loader

def train_strategy(strategy_name, model, train_loader, val_loader):
    """Train a model with a specific fine-tuning strategy."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training with strategy: {strategy_name}")
    
    # Get strategy configuration
    strategy_config = get_training_strategies()[strategy_name]
    
    # Apply strategy to model
    model = strategy_config.apply(model)
    
    # Setup callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_f1",
            patience=5,
            mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=f"models/checkpoints/{strategy_name}",
            filename=f"{strategy_name}-best-f1-{{epoch:02d}}-val_f1={{val_f1:.4f}}",
            monitor="val_f1",
            mode="max",
            save_top_k=1
        )
    ]
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        val_check_interval=1.0
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    logger.info(f"Training completed for strategy: {strategy_name}")
    
    return model, trainer

def train_models():
    """Main training function for all strategies."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        train_loader, val_loader = load_data()
        
        # Training strategies to try
        strategies = [
            "freeze_backbone",
            "last1+head", 
            "last2+head",
            "last3+head",
            "last4+head",
            "full"
        ]
        
        trained_models = {}
        
        for strategy in strategies:
            logger.info(f"Training strategy: {strategy}")
            
            # Create model
            model = EfficientNetV2Lightning(
                num_classes=6,  # Aom, Chronic, Earwax, Normal, Otitis Externa, Tympanosclerosis
                lr=1e-4
            )
            
            # Train model
            trained_model, trainer = train_strategy(strategy, model, train_loader, val_loader)
            trained_models[strategy] = trained_model
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                f"{strategy}_final_val_f1": trainer.callback_metrics.get("val_f1", 0.0),
                f"{strategy}_final_val_loss": trainer.callback_metrics.get("val_loss", 0.0)
            })
        
        logger.info("All training strategies completed successfully!")
        
        # Save best checkpoints info
        save_best_checkpoints_info(trained_models)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        mlflow.end_run()

def save_best_checkpoints_info(trained_models):
    """Save information about best checkpoints for each strategy."""
    logger = logging.getLogger(__name__)
    
    checkpoints_info = {}
    
    for strategy, model in trained_models.items():
        # Find best checkpoint file
        checkpoint_dir = Path(f"models/checkpoints/{strategy}")
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoint_files:
                # Get the most recent checkpoint
                best_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                checkpoints_info[strategy] = str(best_checkpoint)
    
    # Save to JSON file
    import json
    with open("logs/best_checkpoints.json", "w") as f:
        json.dump(checkpoints_info, f, indent=2)
    
    logger.info("Best checkpoints information saved")

if __name__ == "__main__":
    setup_logging()
    train_models()
