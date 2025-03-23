"""
Utilities for distributed training with PyTorch.
"""

import datetime
import os
import subprocess
import sys
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedBackend(str, Enum):
    """Backend for distributed communication."""
    NCCL = "nccl"  # GPU training (recommended for multi-GPU)
    GLOO = "gloo"  # CPU training or Windows


class DataDistributionStrategy(str, Enum):
    """Strategy for distributing data in distributed training."""
    DISTRIBUTED = "distributed"  # Each process loads a subset of the data
    CENTRALIZED = "centralized"  # Process 0 loads all data and distributes


def get_rank() -> int:
    """Get the rank of current process in the distributed group."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the world size (total number of processes) in the distributed group."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get the local rank of current process (on the current node)."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    
    # Try to infer local rank from global rank if not set
    if torch.cuda.is_available():
        return get_rank() % torch.cuda.device_count()
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return torch.distributed.is_available() and torch.cuda.is_available() and torch.cuda.device_count() > 1


def configure_nccl_for_nvlink():
    """Configure NCCL to detect and optimize for NVLink."""
    # Set environment variables for better NCCL performance with NVLink
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "WARN"  # Set to INFO for more verbose NCCL output
    
    # Try to detect NVLink using nvidia-smi
    try:
        output = subprocess.check_output(
            "nvidia-smi topo -m", shell=True, universal_newlines=True
        )
        if "NV" in output:
            print("NVLink detected - NCCL configured for optimal performance")
        else:
            print("NVLink not detected - using standard NCCL configuration")
    except:
        print("Could not check for NVLink presence")


def init_distributed(backend: DistributedBackend = DistributedBackend.NCCL, timeout: int = 1800):
    """
    Initialize distributed training.
    
    Args:
        backend: Backend for distributed communication
        timeout: Timeout in seconds for operations
    """
    if not dist.is_available():
        print("Error: Distributed package not available")
        return False

    if not dist.is_initialized():
        # Check for required environment variables
        required_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        for var in required_vars:
            if var not in os.environ:
                print(f"Error: {var} environment variable not set")
                return False

        # Initialize the process group
        try:
            dist.init_process_group(
                backend=backend,
                timeout=datetime.timedelta(seconds=timeout),
            )
            print(f"Initialized process group: rank={get_rank()}, world_size={get_world_size()}, backend={backend}")
            return True
        except Exception as e:
            print(f"Error initializing distributed process group: {e}")
            return False
    return True


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def launch_distributed_from_args(args, fn: Callable, backend: DistributedBackend = DistributedBackend.NCCL):
    """
    Launch distributed training using the provided function.
    
    Args:
        args: Command line arguments
        fn: Function to run in each process
        backend: Backend for distributed communication
    """
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        raise ValueError(f"Cannot launch distributed training with only {world_size} GPU")
    
    os.environ["MASTER_ADDR"] = getattr(args, "master_addr", "localhost")
    os.environ["MASTER_PORT"] = getattr(args, "master_port", "12355")
    
    # Define wrapper to handle process launch
    def _distributed_worker(local_rank, world_size, fn, *args, **kwargs):
        global_rank = local_rank
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Call the actual function
        fn(global_rank, local_rank, world_size, *args, **kwargs)
    
    # Launch processes
    mp.spawn(
        _distributed_worker,
        args=(world_size, fn, *args),
        nprocs=world_size,
        join=True
    )


def wrap_model_for_distributed(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    bucket_cap_mb: int = 25,
) -> DDP:
    """
    Wrap a model for distributed training using DistributedDataParallel.
    
    Args:
        model: The model to wrap
        device_id: The device ID to use
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Whether to use gradient as bucket view
        bucket_cap_mb: Maximum bucket size in MiB
        
    Returns:
        The wrapped model
    """
    if not dist.is_initialized():
        print("Warning: Distributed is not initialized. Using model without DDP wrapper.")
        return model
    
    # Move model to the correct device
    model = model.to(device_id)
    
    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        bucket_cap_mb=bucket_cap_mb,
    )
    
    return ddp_model


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all processes and return them as a list.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors, one from each process
    """
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce a dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average the values
        
    Returns:
        Reduced dictionary
    """
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
    
    # Convert dict to tensor
    names = []
    values = []
    for k, v in sorted(input_dict.items()):
        names.append(k)
        values.append(v)
    
    values = torch.stack(values, dim=0)
    dist.all_reduce(values)
    
    if average:
        values /= world_size
    
    # Convert back to dict
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_model_for_distributed_training(model, device_id, config):
    """
    Set up a model for distributed training.
    
    Args:
        model: The model to set up
        device_id: The device ID
        config: Training configuration
        
    Returns:
        The model set up for distributed training
    """
    if not config.distributed.enabled or not is_distributed_available():
        return model
    
    print(f"Setting up model for distributed training on device {device_id}")
    
    # Wrap model with DDP
    return wrap_model_for_distributed(
        model=model,
        device_id=device_id,
        find_unused_parameters=config.distributed.find_unused_parameters,
        gradient_as_bucket_view=config.distributed.gradient_as_bucket_view,
        bucket_cap_mb=config.distributed.bucket_cap_mb or 25,
    )
