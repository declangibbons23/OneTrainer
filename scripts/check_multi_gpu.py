#!/usr/bin/env python3
"""
Check Multi-GPU Setup

This script checks if your system is properly set up for multi-GPU training.
It verifies that PyTorch is installed, CUDA is available, and multiple GPUs are detected.
"""

import os
import sys
import traceback

print("Checking Multi-GPU Setup")
print("========================")

# Check Python version
print(f"\n[1] Python Version: {sys.version.split()[0]}")

# Check PyTorch installation
try:
    import torch
    print(f"\n[2] PyTorch Version: {torch.__version__}")
except ImportError:
    print("\n[2] PyTorch: Not installed!")
    print("    Please install PyTorch with CUDA support:")
    print("    https://pytorch.org/get-started/locally/")
    sys.exit(1)

# Check CUDA availability
try:
    if torch.cuda.is_available():
        print("\n[3] CUDA: Available")
        print(f"    CUDA Version: {torch.version.cuda}")
    else:
        print("\n[3] CUDA: Not available!")
        print("    Please ensure you have a CUDA-capable GPU and PyTorch was installed with CUDA support.")
        sys.exit(1)
except Exception as e:
    print(f"\n[3] CUDA: Error checking CUDA: {e}")
    sys.exit(1)

# Check GPU count
try:
    gpu_count = torch.cuda.device_count()
    print(f"\n[4] GPUs: {gpu_count} detected")
    
    if gpu_count < 2:
        print("    Warning: Multi-GPU training requires at least 2 GPUs")
        print("    You can still use the UI, but training will run on a single GPU")
    
    # Print GPU info
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
        print(f"    GPU {i}: {gpu_name} ({memory:.2f} GB)")
except Exception as e:
    print(f"\n[4] GPUs: Error checking GPUs: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Check distributed module
try:
    import torch.distributed as dist
    print("\n[5] torch.distributed: Available")
except ImportError:
    print("\n[5] torch.distributed: Not available!")
    print("    This is unusual with standard PyTorch installations.")
    print("    Please reinstall PyTorch with CUDA support.")
    sys.exit(1)

# Check DDP implementation
try:
    from torch.nn.parallel import DistributedDataParallel
    print("\n[6] DistributedDataParallel: Available")
except ImportError:
    print("\n[6] DistributedDataParallel: Not available!")
    print("    This is unusual with standard PyTorch installations.")
    print("    Please reinstall PyTorch.")
    sys.exit(1)

# Check OneTrainer multi-GPU implementation
try:
    from modules.trainer.DistributedTrainer import DistributedTrainer
    from modules.util.distributed_util import setup_distributed, cleanup_distributed
    print("\n[7] OneTrainer Multi-GPU: Implemented")
except ImportError:
    print("\n[7] OneTrainer Multi-GPU: Not available!")
    print("    Make sure scripts/train_multi_gpu.py exists and the implementation is complete.")
    sys.exit(1)

print("\n========================")
if gpu_count >= 2:
    print("Your system is ready for multi-GPU training!")
    print("\nTo use multi-GPU training:")
    print("1. Use the Multi-GPU panel in the training UI")
    print("2. Or run: ./start-multi-gpu.sh (Linux/macOS) or start-multi-gpu.bat (Windows)")
else:
    print("Your system only has 1 GPU. Multi-GPU training requires at least 2 GPUs.")
    print("You can still use the UI, but training will run on a single GPU.")