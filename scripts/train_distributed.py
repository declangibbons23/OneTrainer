#!/usr/bin/env python3
"""
Script for launching distributed training across multiple GPUs.

This script initializes the distributed environment, sets up the 
required process group, and launches the training process.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import distributed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.DistributedConfig import Backend, DataDistributionStrategy
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit


def setup_distributed_environment(rank, world_size, args):
    """
    Set up the distributed environment for a single process.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Set process-specific environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    # Configure GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    # Set backend
    backend = args.backend
    
    # Configure NCCL for better performance if using NVLink
    if backend == Backend.NCCL and args.detect_nvlink:
        distributed.configure_nccl_for_nvlink()
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=args.timeout),
    )
    
    # Print information about the distributed setup
    if rank == 0:
        print(f"Initialized distributed environment with {world_size} processes")
        print(f"Master: {args.master_addr}:{args.master_port}, Backend: {backend}")
    
    # Synchronize all processes
    dist.barrier()


def load_config(args):
    """
    Load and configure the training configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured TrainConfig object
    """
    # Load the base training config
    config_path = args.config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object from dictionary
    config = TrainConfig()
    config.from_dict(config_dict)
    
    # Add or update distributed configuration
    if not hasattr(config, 'distributed'):
        from modules.util.config.DistributedConfig import DistributedConfig
        config.distributed = DistributedConfig()
    
    # Enable distributed training
    config.distributed.enabled = True
    
    # Set distributed configuration from arguments
    config.distributed.backend = args.backend
    config.distributed.master_addr = args.master_addr
    config.distributed.master_port = args.master_port
    config.distributed.timeout = args.timeout
    config.distributed.detect_nvlink = args.detect_nvlink
    config.distributed.data_loading_strategy = args.data_strategy
    config.distributed.latent_caching_strategy = args.cache_strategy
    config.distributed.find_unused_parameters = args.find_unused_parameters
    config.distributed.gradient_as_bucket_view = args.gradient_as_bucket_view
    config.distributed.bucket_cap_mb = args.bucket_cap_mb
    config.distributed.static_graph = args.static_graph
    
    # Set device based on local rank
    local_rank = distributed.get_local_rank()
    config.train_device = f"cuda:{local_rank}"
    
    return config


def train_worker(rank, world_size, args):
    """
    Worker function for each distributed process.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Set up distributed environment
    setup_distributed_environment(rank, world_size, args)
    
    # Load and configure training
    config = load_config(args)
    
    # Create dummy callbacks and commands
    # These could be extended to support remote monitoring
    callbacks = TrainCallbacks()
    commands = TrainCommands()
    
    # Create trainer and start training
    trainer = GenericTrainer(config, callbacks, commands)
    
    try:
        # Start trainer
        trainer.start()
        
        # Run training loop
        if rank == 0:
            print(f"Starting distributed training with {world_size} processes")
        trainer.train()
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Main function to parse arguments and launch training."""
    parser = argparse.ArgumentParser(description="Distributed training script for OneTrainer")
    
    # Basic arguments
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the training configuration JSON file")
    
    # Distributed specific arguments
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], default="nccl",
                        help="Distributed backend to use (nccl for GPU, gloo for CPU)")
    parser.add_argument("--master_addr", type=str, default="localhost",
                        help="Master node address")
    parser.add_argument("--master_port", type=str, default="12355",
                        help="Master node port")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout for operations in seconds")
    parser.add_argument("--detect_nvlink", action="store_true", default=True,
                        help="Detect and optimize for NVLink")
    parser.add_argument("--data_strategy", type=str, 
                        choices=["distributed", "centralized"], default="distributed",
                        help="Strategy for data loading (distributed=each GPU loads a subset, centralized=rank 0 loads all)")
    parser.add_argument("--cache_strategy", type=str, 
                        choices=["distributed", "centralized"], default="distributed",
                        help="Strategy for latent caching (distributed=each GPU caches a subset, centralized=rank 0 caches all)")
    parser.add_argument("--find_unused_parameters", action="store_true", default=False,
                        help="Find unused parameters in forward pass (slower, but needed for some models)")
    parser.add_argument("--gradient_as_bucket_view", action="store_true", default=True,
                        help="Use gradient bucket view to reduce memory usage")
    parser.add_argument("--bucket_cap_mb", type=int, default=25,
                        help="Maximum bucket size in MiB")
    parser.add_argument("--static_graph", action="store_true", default=False,
                        help="Use static graph optimization")
    
    args = parser.parse_args()
    
    # Check if distributed training is available
    if not distributed.is_distributed_available():
        print("Error: Distributed training is not available. Need CUDA and multiple GPUs.")
        sys.exit(1)
    
    # Get world size (number of GPUs)
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"Error: Distributed training requires at least 2 GPUs, but only {world_size} found.")
        sys.exit(1)
    
    print(f"Launching distributed training with {world_size} GPUs")
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
