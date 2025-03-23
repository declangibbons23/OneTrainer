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

# Execute a Python command directly and capture its output
get_gpu_count() {
    run_python_in_active_env -c "import torch; print(torch.cuda.device_count())" 2>/dev/null
}

# Get number of available GPUs
NUM_GPUS=$(get_gpu_count)
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
    # Direct approach without variable substitution
    run_python_in_active_env << EOF
import sys
sys.path.append(".")
from scripts.train_multi_gpu import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, default="$CONFIG_PATH")
parser.add_argument("--num-gpus", type=int, default=$NUM_GPUS_TO_USE)
parser.add_argument("--spawn", action="store_true", default=True)
args = parser.parse_args()

main(args)
EOF
else
    echo "Launching with torchrun"
    # Direct approach without variable substitution
    run_python_in_active_env << EOF
import sys
import os
import subprocess
sys.path.append(".")

# Execute torchrun
cmd = [
    sys.executable,
    "-m", "torch.distributed.run",
    f"--nproc_per_node=$NUM_GPUS_TO_USE",
    f"--master_port=12355",
    "scripts/train_multi_gpu.py",
    f"--config-path=$CONFIG_PATH",
    f"--num-gpus=$NUM_GPUS_TO_USE",
]

print(f"Executing: {' '.join(cmd)}")
process = subprocess.run(cmd)
sys.exit(process.returncode)
EOF
fi