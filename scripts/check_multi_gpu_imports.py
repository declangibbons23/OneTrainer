#!/usr/bin/env python3
"""
Script to check that all multi-GPU related modules can be imported correctly.
This helps identify any missing dependencies or import issues.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_import(module_name):
    """
    Attempt to import a module and log the result.
    """
    try:
        __import__(module_name)
        logger.info(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error when importing {module_name}: {e}")
        return False

def main():
    logger.info("Checking multi-GPU related imports...")
    
    # Base Python packages
    check_import("torch")
    check_import("torch.distributed")
    check_import("torch.multiprocessing")
    
    # UI related imports
    check_import("modules.ui.TooltipLabel")
    check_import("modules.ui.MultiGPUFrame")
    check_import("customtkinter")  # Required for UI components
    
    # Core multi-GPU functionality
    check_import("modules.util.distributed_util")
    check_import("modules.trainer.DistributedTrainer")
    check_import("modules.util.config.TrainConfig")
    
    # Display system info
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
    
    logger.info("Import check complete")

if __name__ == "__main__":
    main()