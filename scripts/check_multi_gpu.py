#!/usr/bin/env python3
"""
Multi-GPU diagnostic script for OneTrainer.
This script checks if your system is properly set up for multi-GPU training.
"""

import os
import sys
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import script_imports first
try:
    # First try from the expected location
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    sys.path.append(str(root_dir))
    
    try:
        from util.import_util import script_imports
        script_imports()
    except ImportError:
        # If that fails, try directly
        sys.path.append(str(script_dir))

except Exception as e:
    logger.error(f"Error setting up Python path: {e}")
    logger.error("This script should be run from the OneTrainer directory.")

# Import checks
logger.info("=== OneTrainer Multi-GPU System Check ===")
logger.info(f"Python version: {platform.python_version()}")
logger.info(f"Platform: {platform.platform()}")

# Check PyTorch and CUDA
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes (version {torch.version.cuda})")
        gpu_count = torch.cuda.device_count()
        logger.info(f"Number of GPUs detected: {gpu_count}")
        
        if gpu_count < 2:
            logger.warning("Multi-GPU training requires at least 2 GPUs, but only 1 was detected.")
        else:
            logger.info("✓ Multiple GPUs detected - Good")
            
        # Print GPU info
        logger.info("\nGPU Information:")
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            try:
                mem_info = torch.cuda.get_device_properties(i).total_memory / 1024**3  # in GB
                logger.info(f"    Memory: {mem_info:.2f} GB")
            except:
                logger.info("    Memory: Unknown")
    else:
        logger.error("CUDA is not available. Multi-GPU training requires CUDA support.")
except ImportError:
    logger.error("PyTorch is not installed or failed to import. Please install PyTorch with CUDA support.")

# Check distributed package
try:
    import torch.distributed as dist
    
    # Check NCCL backend
    try:
        if hasattr(torch.distributed, 'is_nccl_available'):
            if torch.distributed.is_nccl_available():
                logger.info("✓ NCCL backend is available - Good")
            else:
                logger.warning("NCCL backend is not available. It's recommended for GPU training.")
        else:
            # Older PyTorch versions
            logger.info("? Unable to check NCCL backend availability")
    except:
        logger.warning("Failed to check NCCL backend availability")
    
    # Check GLOO backend
    try:
        if hasattr(torch.distributed, 'is_gloo_available'):
            if torch.distributed.is_gloo_available():
                logger.info("✓ GLOO backend is available - Good")
            else:
                logger.warning("GLOO backend is not available. It's an alternative to NCCL.")
        else:
            # Older PyTorch versions
            logger.info("? Unable to check GLOO backend availability")
    except:
        logger.warning("Failed to check GLOO backend availability")
        
except ImportError:
    logger.error("PyTorch distributed module failed to import.")

# Check for OneTrainer modules
try:
    from modules.trainer.DistributedTrainer import DistributedTrainer
    from modules.util.distributed_util import setup_distributed
    logger.info("✓ OneTrainer distributed modules are available - Good")
except ImportError as e:
    logger.error(f"OneTrainer distributed modules not found: {e}")
    logger.error("Make sure you're running this script from the root OneTrainer directory.")

# Check if running as root (which can cause issues)
if os.name == 'posix' and os.geteuid() == 0:
    logger.warning("You are running as root, which might cause issues with multi-GPU training.")

logger.info("\n=== Result ===")
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    logger.info("Your system appears ready for multi-GPU training!")
    logger.info("You can enable multi-GPU training in the UI or run start-multi-gpu.sh/bat")
else:
    logger.info("Your system is not properly configured for multi-GPU training.")
    if torch.cuda.device_count() < 2:
        logger.info("Reason: At least 2 GPUs are required, but only " + 
                  f"{torch.cuda.device_count()} {'was' if torch.cuda.device_count() == 1 else 'were'} detected.")
    elif not torch.cuda.is_available():
        logger.info("Reason: CUDA is not available.")
    else:
        logger.info("Reason: Unknown issue with your setup.")

# Performance recommendations
logger.info("\n=== Performance Recommendations ===")
logger.info("1. Use the NCCL backend for best GPU-to-GPU communication performance")
logger.info("2. Enable distributed data loading for optimal throughput")
logger.info("3. Consider using automatic learning rate scaling based on GPU count")
logger.info("4. For memory-limited GPUs, try reducing batch size per GPU")
logger.info("5. Make sure all GPUs are of the same model for best performance")