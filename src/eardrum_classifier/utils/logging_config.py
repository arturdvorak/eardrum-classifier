"""
Logging configuration for the eardrum classification project.
"""

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {logging.getLevelName(level)}")
    logging.info(f"Log file: {log_dir / 'pipeline.log'}")
