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

# Launch distributed training using torchrun
torchrun --nproc_per_node="$GPU_COUNT" scripts/train_headless.py "$@"

if [ $? -ne 0 ]; then
    echo "Error during distributed training"
    exit 1
fi

exit 0
