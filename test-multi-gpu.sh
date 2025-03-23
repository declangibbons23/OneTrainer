#!/bin/bash

# Script to test if multi-GPU training works on this system

echo "========================================"
echo "       MULTI-GPU TRAINING TEST"
echo "========================================"
echo

# Include the lib.include.sh for run_python_in_active_env
source "$(dirname "$0")/lib.include.sh"

# Check CUDA
echo "Checking CUDA availability..."
run_python_in_active_env - << 'EOF'
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s)")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if gpu_count < 2:
        print("Multi-GPU training requires at least 2 GPUs")
    else:
        print("System has enough GPUs for multi-GPU training")
else:
    print("CUDA is not available")
    print("Multi-GPU training requires CUDA support")
EOF

echo
echo "Testing simple multi-GPU model..."
echo

# Run the test script directly with Python
run_python_in_active_env scripts/check_multi_gpu.py

echo
echo "If no errors occurred and you see output from multiple GPU ranks,"
echo "your system is correctly set up for multi-GPU training."
echo