#!/usr/bin/env python3
"""
Eardrum Classification Pipeline Runner

This script orchestrates the entire pipeline from data setup to model evaluation.
It can run individual steps or the complete pipeline based on command line arguments.
"""

import argparse
import logging
from pathlib import Path

# Import from the package
from eardrum_classifier import setup_dataset, train_models, evaluate_models, setup_logging

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description="Eardrum Classification Pipeline")
    parser.add_argument("--step", choices=["data", "train", "evaluate", "all"], 
                       default="all", help="Pipeline step to execute")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Eardrum Classification Pipeline")
    logger.info(f"Executing step: {args.step}")
    
    try:
        if args.step in ["data", "all"]:
            logger.info("Step 1: Setting up dataset...")
            setup_dataset()
            
        if args.step in ["train", "all"]:
            logger.info("Step 2: Training models...")
            train_models()
            
        if args.step in ["evaluate", "all"]:
            logger.info("Step 3: Evaluating models...")
            evaluate_models()
            
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
