#!/usr/bin/env bash

set -e

# Use the same library as start-ui.sh to properly set up the environment
source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Prepare the environment (activates conda env)
prepare_runtime_environment

# Now run Python commands in the proper environment
echo "Checking for CUDA support..."
if ! run_python_in_active_env -c "import torch; print('CUDA available:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
    echo "CUDA is not available. Multi-GPU training requires CUDA support."
    echo "Please install CUDA and PyTorch with CUDA support, or use single-GPU training."
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(run_python_in_active_env -c "import torch; print(torch.cuda.device_count())")
echo "CUDA available: Yes"
echo "Number of GPUs: $NUM_GPUS"

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
    run_python_in_active_env scripts/train_multi_gpu.py "--config-path=$CONFIG_PATH" "--num-gpus=$NUM_GPUS_TO_USE" "--spawn"
else
    echo "Launching with torchrun"
    run_python_in_active_env scripts/train_multi_gpu.py "--config-path=$CONFIG_PATH" "--num-gpus=$NUM_GPUS_TO_USE"
fi