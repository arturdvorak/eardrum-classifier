#!/usr/bin/env python3
"""
Inference Deployment Script

This script handles the complete deployment of the ONNX Runtime inference service:
1. Exports PyTorch models to ONNX format
2. Builds Docker containers
3. Starts the inference service
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if required tools are installed."""
    print("Checking requirements...")
    
    required_tools = ['docker', 'docker-compose', 'python']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            print(f"{tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{tool} is not installed")
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"\nMissing required tools: {', '.join(missing_tools)}")
        print("Please install them before running this script.")
        return False
    
    return True

def export_models(checkpoints_file, output_dir, strategy=None, optimize=True):
    """Export PyTorch models to ONNX format."""
    print("Exporting models to ONNX format...")
    
    # Check if checkpoints file exists
    if not os.path.exists(checkpoints_file):
        print(f"Checkpoints file not found: {checkpoints_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build export command
    cmd = f"python export_model.py --checkpoints-file {checkpoints_file} --output-dir {output_dir}"
    
    if strategy:
        cmd += f" --strategy {strategy}"
    else:
        cmd += " --use-best-only"  # Export only the best model by default
    
    if optimize:
        cmd += " --optimize"
    
    return run_command(cmd, "Model export")

def build_containers():
    """Build Docker containers."""
    print("Building Docker containers...")
    
    # Build training container
    if not run_command("docker-compose build eardrum-classifier", "Training container build"):
        return False
    
    # Build inference container
    if not run_command("docker-compose build eardrum-inference", "Inference container build"):
        return False
    
    return True

def start_services():
    """Start all services."""
    print("Starting services...")
    
    return run_command("docker-compose up -d", "Services startup")

def stop_services():
    """Stop all services."""
    print("Stopping services...")
    
    return run_command("docker-compose down", "Services shutdown")

def show_status():
    """Show service status and endpoints."""
    print("\nService Status:")
    
    # Check if containers are running
    result = subprocess.run("docker-compose ps", shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    print("\nAvailable Endpoints:")
    print("  - Training Service: http://localhost:5000 (if MLflow UI is running)")
    print("  - Inference API: http://localhost:8000")
    print("  - API Documentation: http://localhost:8000/docs")
    print("  - Health Check: http://localhost:8000/health")
    
    print("\nUsage Examples:")
    print("  # Test health check")
    print("  curl http://localhost:8000/health")
    print("")
    print("  # Test prediction (replace with actual image path)")
    print("  curl -X POST 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'image=@test_image.jpg'")

def main():
    parser = argparse.ArgumentParser(description="Deploy ONNX Runtime inference service")
    parser.add_argument("--checkpoints-file", default="checkpoints/best_checkpoints_20250918-142120.json", help="Path to checkpoints JSON file")
    parser.add_argument("--output-dir", default="models/onnx", help="Output directory for ONNX models")
    parser.add_argument("--strategy", help="Export specific strategy only")
    parser.add_argument("--no-optimize", action="store_true", help="Skip ONNX optimization")
    parser.add_argument("--build-only", action="store_true", help="Only build containers, don't start services")
    parser.add_argument("--start-only", action="store_true", help="Only start services, don't build")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--status", action="store_true", help="Show service status")
    
    args = parser.parse_args()
    
    print("Eardrum Classification - ONNX Runtime Deployment")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Stop services if requested
    if args.stop:
        return 0 if stop_services() else 1
    
    # Show status if requested
    if args.status:
        show_status()
        return 0
    
    # Export models (unless start-only)
    if not args.start_only:
        if not export_models(
            checkpoints_file=args.checkpoints_file,
            output_dir=args.output_dir,
            strategy=args.strategy,
            optimize=not args.no_optimize
        ):
            return 1
    
    # Build containers (unless start-only)
    if not args.start_only:
        if not build_containers():
            return 1
    
    # Start services (unless build-only)
    if not args.build_only:
        if not start_services():
            return 1
        
        # Show status after starting
        show_status()
    
    print("\nDeployment completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
