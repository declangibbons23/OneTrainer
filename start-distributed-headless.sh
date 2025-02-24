#!/bin/bash

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python is not found in PATH"
    exit 1
fi

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is not available"
    exit 1
fi

# Get number of available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "Found only $GPU_COUNT GPU(s). Multi-GPU training requires at least 2 GPUs."
    exit 1
fi

echo "Starting headless distributed training with $GPU_COUNT GPUs..."

# Set environment variables for NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Launch distributed training using torchrun
# --nnodes=1: We're running on a single machine
# --nproc_per_node=$GPU_COUNT: Launch one process per GPU
# --master_port=29500: Default port for distributed training
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    scripts/train_headless.py "$@"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Error during distributed training"
    exit $exit_code
fi

exit 0
