"""
Simple and reliable distributed training utilities
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_process_group(rank, world_size, backend='nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_process_group():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def wrap_model_for_ddp(model, device_id):
    """Wrap a model for DDP training"""
    return DDP(model, device_ids=[device_id])

def run_distributed_training(train_function, world_size, config=None, callbacks=None, commands=None):
    """
    Run a distributed training function across multiple GPUs
    
    Args:
        train_function: The training function to run on each GPU
                       Should accept (rank, world_size, config, callbacks, commands)
        world_size: Number of GPUs to use
        config: Configuration object to pass to each worker
        callbacks: Callbacks object to pass to rank 0 process
        commands: Commands object to pass to rank 0 process
    """
    if world_size < 2:
        raise ValueError(f"Distributed training requires at least 2 GPUs, but only {world_size} specified")
    
    # Use spawn to launch all processes
    mp.spawn(
        _run_worker,
        args=(world_size, train_function, config, callbacks, commands),
        nprocs=world_size,
        join=True
    )

def _run_worker(rank, world_size, train_function, config, callbacks, commands):
    """Worker function for each GPU process"""
    try:
        # Set up the distributed environment
        setup_process_group(rank, world_size)
        
        # Only pass callbacks and commands to rank 0 (main process)
        worker_callbacks = callbacks if rank == 0 else None
        worker_commands = commands if rank == 0 else None
        
        # Run the actual training function
        train_function(rank, world_size, config, worker_callbacks, worker_commands)
    finally:
        # Clean up
        cleanup_process_group()