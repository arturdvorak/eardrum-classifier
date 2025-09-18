#!/usr/bin/env python3
"""
Model Export Script

This script exports the best trained PyTorch model to ONNX format for production inference.
It automatically finds the best model from checkpoints and exports it with optimizations.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference.model_exporter import export_best_models_from_checkpoints, export_to_onnx

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX format")
    parser.add_argument(
        "--checkpoints-file", 
        default="checkpoints/best_checkpoints_20250918-142120.json",
        help="Path to checkpoints JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        default="models/onnx",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--strategy", 
        choices=["freeze_backbone", "last1+head", "last2+head", "last3+head", "last4+head", "full"],
        help="Export specific strategy only (default: export all)"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Optimize ONNX models for better performance"
    )
    parser.add_argument(
        "--use-best-only", 
        action="store_true",
        help="Export only the best performing model (full strategy)"
    )
    
    args = parser.parse_args()
    
    print("Starting model export to ONNX format...")
    print(f"Checkpoints file: {args.checkpoints_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Optimize: {args.optimize}")
    
    # Check if checkpoints file exists
    if not os.path.exists(args.checkpoints_file):
        print(f"Checkpoints file not found: {args.checkpoints_file}")
        print("Available checkpoint files:")
        checkpoint_dir = Path("checkpoints")
        for file in checkpoint_dir.glob("best_checkpoints_*.json"):
            print(f"  - {file}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.strategy:
            # Export specific strategy
            print(f"Exporting strategy: {args.strategy}")
            
            # Load checkpoints
            with open(args.checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
            
            if args.strategy not in checkpoints:
                print(f"Strategy '{args.strategy}' not found in checkpoints")
                print(f"Available strategies: {list(checkpoints.keys())}")
                return 1
            
            checkpoint_path = checkpoints[args.strategy]
            output_path = os.path.join(args.output_dir, f"eardrum_classifier_{args.strategy}.onnx")
            
            export_to_onnx(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                optimize=args.optimize
            )
            
            print(f"Successfully exported {args.strategy} to {output_path}")
            
        elif args.use_best_only:
            # Export only the best model (full strategy)
            print("Exporting best model (full strategy)...")
            
            with open(args.checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
            
            if 'full' not in checkpoints:
                print("'full' strategy not found in checkpoints")
                return 1
            
            checkpoint_path = checkpoints['full']
            output_path = os.path.join(args.output_dir, "eardrum_classifier_full.onnx")
            
            export_to_onnx(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                optimize=args.optimize
            )
            
            print(f"Successfully exported best model to {output_path}")
            
        else:
            # Export all models
            print("Exporting all model strategies...")
            
            onnx_models = export_best_models_from_checkpoints(
                checkpoints_file=args.checkpoints_file,
                output_dir=args.output_dir
            )
            
            print(f"Successfully exported {len(onnx_models)} models:")
            for strategy, path in onnx_models.items():
                print(f"  - {strategy}: {path}")
        
        print("\nModel export completed successfully!")
        print(f"ONNX models saved to: {args.output_dir}")
        
        # Show file sizes
        print("\nModel file sizes:")
        for file in Path(args.output_dir).glob("*.onnx"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"Export failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
