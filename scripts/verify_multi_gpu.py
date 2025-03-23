#!/usr/bin/env python3
"""
Script to verify multi-GPU training setup in OneTrainer
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def check_cuda():
    """Check CUDA availability and GPU count"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return False
        
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        logger.error(f"Multi-GPU training requires at least 2 GPUs, but only {gpu_count} detected")
        return False
        
    logger.info(f"Found {gpu_count} CUDA GPUs:")
    for i in range(gpu_count):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return True

def check_distributed():
    """Check distributed training requirements"""
    try:
        import torch.distributed as dist
        logger.info("PyTorch distributed package available")
        
        # Check NCCL backend
        if hasattr(dist, 'is_nccl_available') and dist.is_nccl_available():
            logger.info("NCCL backend available (recommended)")
        else:
            logger.warning("NCCL backend not available, will fall back to GLOO")
            
        return True
    except ImportError:
        logger.error("PyTorch distributed package not available")
        return False

def check_trainer():
    """Check OneTrainer distributed components"""
    try:
        from modules.trainer.DistributedTrainer import DistributedTrainer
        from modules.util.config.TrainConfig import TrainConfig
        
        # Create dummy config
        config = TrainConfig.default_values()
        config.enable_multi_gpu = True
        config.distributed_backend = "nccl"
        config.world_size = torch.cuda.device_count()
        
        # Try to create trainer
        trainer = DistributedTrainer(config, None, None, local_rank=0)
        logger.info("Successfully created DistributedTrainer")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DistributedTrainer: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks"""
    logger.info("Starting multi-GPU verification...")
    
    checks = [
        ("CUDA/GPU Check", check_cuda),
        ("Distributed Check", check_distributed),
        ("Trainer Check", check_trainer)
    ]
    
    all_passed = True
    for name, check_fn in checks:
        logger.info(f"\nRunning {name}...")
        try:
            if check_fn():
                logger.info(f"{name} PASSED")
            else:
                logger.error(f"{name} FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"{name} FAILED with error: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\nAll checks PASSED - Multi-GPU training should work")
        return 0
    else:
        logger.error("\nSome checks FAILED - Please fix issues before using multi-GPU training")
        return 1

if __name__ == "__main__":
    sys.exit(main())