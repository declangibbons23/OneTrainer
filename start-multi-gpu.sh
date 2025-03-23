#!/bin/bash

# Make the script executable
chmod +x "$(dirname "$0")/start-multi-gpu.sh"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python and try again."
    exit 1
fi

# Check if PyTorch is installed
if ! python -c "import torch" &> /dev/null; then
    echo "PyTorch not found. Please install PyTorch and try again."
    exit 1
fi

# Check if CUDA is available
if ! python -c "import torch; cuda_available = torch.cuda.is_available(); print('CUDA available:', cuda_available); print('Number of GPUs:', torch.cuda.device_count() if cuda_available else 0); exit(0 if cuda_available else 1)"; then
    echo "CUDA is not available. Multi-GPU training requires CUDA support."
    echo "Please install CUDA and PyTorch with CUDA support, or use single-GPU training."
    exit 1
fi
echo

# Get number of available GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "Only $NUM_GPUS GPU detected. Multi-GPU training requires at least 2 GPUs."
    echo "Falling back to single GPU training."
    bash "$(dirname "$0")/start-ui.sh"
    exit 0
fi

echo "Found $NUM_GPUS GPUs available for training."
echo

# Check if user wants to use all GPUs
read -p "Use all $NUM_GPUS GPUs? (y/n): " USE_ALL_GPUS
if [[ "$USE_ALL_GPUS" == "n" || "$USE_ALL_GPUS" == "N" ]]; then
    read -p "Enter number of GPUs to use: " NUM_GPUS_TO_USE
else
    NUM_GPUS_TO_USE=$NUM_GPUS
fi

# Check if config file is provided
read -p "Enter path to training config file: " CONFIG_PATH

if [ -z "$CONFIG_PATH" ]; then
    echo "Config file path is required."
    exit 1
fi

# Check if the user wants to use torchrun or torch.multiprocessing
read -p "Use torchrun for launching? Recommended. (y/n): " USE_TORCHRUN

if [[ "$USE_TORCHRUN" == "n" || "$USE_TORCHRUN" == "N" ]]; then
    echo "Launching with torch.multiprocessing.spawn"
    python -m scripts.train_multi_gpu --config-path "$CONFIG_PATH" --num-gpus $NUM_GPUS_TO_USE --spawn
else
    echo "Launching with torchrun"
    python -m scripts.train_multi_gpu --config-path "$CONFIG_PATH" --num-gpus $NUM_GPUS_TO_USE
fi