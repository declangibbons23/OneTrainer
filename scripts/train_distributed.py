from util.import_util import script_imports

script_imports()

import json
import os
import sys
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed import (
    DistributedBackend,
    launch_distributed_from_args,
    init_distributed,
    cleanup_distributed,
    is_distributed_available,
    configure_nccl_for_nvlink
)


def parse_distributed_args():
    parser = TrainArgs.create_parser()
    
    # Add distributed training specific arguments
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Enable multi-GPU training')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                        help='Distributed backend (nccl or gloo)')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='Master node address (for multi-node training)')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Master port (for multi-node training)')
    
    args = parser.parse_args()
    return args


def train_worker(rank, local_rank, world_size, args=None):
    """
    Worker function that runs on each distributed process.
    """
    # Set up environment for this process
    torch.cuda.set_device(local_rank)
    
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
        train_config.from_dict(config_dict)
    
    # Enable distributed training in config
    train_config.distributed.enabled = True
    train_config.distributed.backend = DistributedBackend(args.backend)
    train_config.distributed.master_addr = args.master_addr
    train_config.distributed.master_port = int(args.master_port)
    
    # Set device-specific settings
    train_config.train_device = f"cuda:{local_rank}"
    
    # Load secrets if available
    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict = json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    # Only show progress on rank 0
    if rank != 0:
        callbacks = TrainCallbacks(silent=True)

    trainer = GenericTrainer(train_config, callbacks, commands)

    # Print information about this process
    if rank == 0:
        print(f"Starting distributed training with {world_size} processes")
        print(f"Backend: {args.backend}")
        if train_config.distributed.detect_nvlink:
            print("NVLink detection enabled")

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        cleanup_distributed()
        raise

    # Only save on rank 0 or if backup_before_save is enabled
    if rank == 0 and (not canceled or train_config.backup_before_save):
        trainer.end()
    
    # Cleanup
    cleanup_distributed()


def main():
    # Parse arguments
    args = parse_distributed_args()
    
    # Check if distributed training is possible
    if args.multi_gpu and not is_distributed_available():
        print("Multi-GPU training requested but torch.cuda.device_count() <= 1 or CUDA not available")
        print("Falling back to single GPU training")
        args.multi_gpu = False
    
    # If running with torchrun/torch.distributed.launch, environment variables will be set
    using_launch_script = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    if args.multi_gpu:
        if using_launch_script:
            # Running with torchrun or similar launcher
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            # Initialize process group
            init_distributed(backend=DistributedBackend(args.backend))
            
            # Run training function
            train_worker(rank, local_rank, world_size, args)
        else:
            # Launch with torch.multiprocessing.spawn
            world_size = torch.cuda.device_count()
            
            if world_size > 1:
                print(f"Launching {world_size} processes with mp.spawn (fallback method)")
                print("For better performance, consider using torchrun instead")
                
                # Set environment variables for the distributed environment
                os.environ["MASTER_ADDR"] = args.master_addr
                os.environ["MASTER_PORT"] = args.master_port
                
                # Using spawn to start multiple processes
                mp.spawn(
                    lambda local_rank: train_worker(
                        rank=local_rank,
                        local_rank=local_rank,
                        world_size=world_size,
                        args=args
                    ),
                    nprocs=world_size,
                    join=True
                )
            else:
                print("Only one GPU found, running in single GPU mode")
                train_worker(rank=0, local_rank=0, world_size=1, args=args)
    else:
        # Single GPU training
        train_worker(rank=0, local_rank=0, world_size=1, args=args)


if __name__ == '__main__':
    main()
