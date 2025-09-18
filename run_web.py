#!/usr/bin/env python3
"""
Web Interface Runner

This script runs the Streamlit web interface locally for testing.
Make sure the inference API is running before starting the web interface.
"""

import subprocess
import sys
import os
import requests
import time

def check_api():
    """Check if the inference API is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    print("Eardrum Classification Web Interface")
    print("=" * 50)
    
    # Check if API is running
    print("Checking API connection...")
    if not check_api():
        print("Inference API is not running!")
        print("Please start the inference service first:")
        print("  docker-compose up -d eardrum-inference")
        print("  or")
        print("  python -m src.inference.api")
        return 1
    
    print("API is running!")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Streamlit is not installed!")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-web.txt"])
        print("Requirements installed")
    
    # Run streamlit
    print("\nStarting web interface...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Set environment variable for local API connection
    env = os.environ.copy()
    env["API_BASE_URL"] = "http://localhost:8000"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "web/app.py", 
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ], env=env)
    except KeyboardInterrupt:
        print("\nWeb interface stopped")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
