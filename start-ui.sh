#!/bin/bash

# Source lib.include.sh for common functions
source lib.include.sh

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

# Set environment variables for NCCL if multiple GPUs are available
if [ "$GPU_COUNT" -gt 1 ]; then
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
fi

# Launch UI
CONDA_RUN_ARGS="--no-capture-output"
CONDA_ENV="conda_env"

# Check version
conda run $CONDA_RUN_ARGS --prefix $CONDA_ENV python scripts/util/version_check.py 3.10 3.13

# Start UI
if [ "$GPU_COUNT" -gt 1 ]; then
    # Multi-GPU mode
    conda run $CONDA_RUN_ARGS --prefix $CONDA_ENV torchrun \
        --nnodes=1 \
        --nproc_per_node=$GPU_COUNT \
        --master_port=29500 \
        scripts/train_ui.py "$@"
else
    # Single-GPU mode
    conda run $CONDA_RUN_ARGS --prefix $CONDA_ENV python scripts/train_ui.py "$@"
fi

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Error during training"
    exit $exit_code
fi

exit 0
