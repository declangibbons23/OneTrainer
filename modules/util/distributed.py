import os
import subprocess
import sys
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


class DistributedBackend(str, Enum):
    NCCL = "nccl"
    GLOO = "gloo"
    

class DataDistributionStrategy(str, Enum):
    DISTRIBUTED = "distributed"
    CENTRALIZED = "centralized"


class NVLinkStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


def is_distributed_available() -> bool:
    """
    Check if distributed training is available on the current system.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def get_world_size() -> int:
    """
    Get the number of processes in the distributed training group.
    Returns 1 if distributed training is not initialized.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get the rank of the current process in the distributed training group.
    Returns 0 if distributed training is not initialized.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """
    Get the local rank of the current process.
    Returns 0 if distributed training is not initialized.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    """
    return get_rank() == 0


def detect_nvlink_status() -> NVLinkStatus:
    """
    Detect if NVLink is available between GPUs.
    """
    if not torch.cuda.is_available():
        return NVLinkStatus.UNAVAILABLE
    
    try:
        # Try to run nvidia-smi topo -m to check for NVLINK connections
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # If NVLink connection is found in the output
        if "NV" in result.stdout:
            return NVLinkStatus.AVAILABLE
        else:
            return NVLinkStatus.UNAVAILABLE
    except (subprocess.SubprocessError, FileNotFoundError):
        return NVLinkStatus.UNKNOWN


def configure_nccl_for_nvlink():
    """
    Configure NCCL to use NVLink if available.
    """
    nvlink_status = detect_nvlink_status()
    
    if nvlink_status == NVLinkStatus.AVAILABLE:
        # Set NCCL environment variables to prioritize NVLink
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        print("NVLink detected, optimizing NCCL for NVLink communication")
    elif nvlink_status == NVLinkStatus.UNAVAILABLE:
        print("NVLink not detected, using PCIe for GPU communication")
    else:
        print("Could not determine NVLink status, using default communication paths")
    
    # Additional NCCL debug info if needed
    # os.environ["NCCL_DEBUG"] = "INFO"


def configure_distributed_env(backend: DistributedBackend = DistributedBackend.NCCL):
    """
    Configure environment variables for distributed training.
    This should be called before init_process_group.
    """
    if backend == DistributedBackend.NCCL and torch.cuda.is_available():
        configure_nccl_for_nvlink()
    elif backend == DistributedBackend.GLOO:
        # No special config needed for GLOO currently
        pass
    
    # Set timeout higher for large models
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


def init_distributed(
    backend: DistributedBackend = DistributedBackend.NCCL,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    init_method: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Initialize the distributed process group.
    
    Args:
        backend: The backend to use (nccl or gloo)
        world_size: The number of processes in the group
        rank: The rank of the current process
        local_rank: The local rank of the current process
        init_method: The initialization method to use
        
    Returns:
        Tuple of (world_size, rank)
    """
    # Check if we're running with torchrun/distributed launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Distributed env detected: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        # If we're not using torchrun, we need all parameters specified
        if world_size is None or rank is None or local_rank is None:
            raise ValueError(
                "For manual initialization, you must specify world_size, rank, and local_rank"
            )
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Configure environment
    configure_distributed_env(backend)
    
    # Initialize the process group
    if init_method is None:
        # Default to env:// if not specified
        init_method = "env://"
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend.value,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    print(f"Initialized process group: rank {rank}/{world_size} on {torch.cuda.get_device_name()}")
    return world_size, rank


def cleanup_distributed():
    """
    Clean up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_ddp(
    model: torch.nn.Module, 
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    broadcast_buffers: bool = True,
) -> DDP:
    """
    Wrap a model for distributed training with DDP.
    
    Args:
        model: The model to wrap
        device_ids: Device IDs to use (defaults to [local_rank])
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Enable memory-efficient gradient views
        broadcast_buffers: Whether to broadcast buffers
        
    Returns:
        The wrapped model
    """
    if device_ids is None:
        device_ids = [get_local_rank()]
    
    return DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=broadcast_buffers,
    )


def prepare_dataloader_for_distributed(
    dataloader: torch.utils.data.DataLoader,
    shuffle: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Prepare a dataloader for distributed training by adding a DistributedSampler.
    
    Args:
        dataloader: The original dataloader
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        A new dataloader with a DistributedSampler
    """
    # Create a distributed sampler
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
    
    # Create a new dataloader with the distributed sampler
    return torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        collate_fn=dataloader.collate_fn,
    )


def launch_distributed_from_args(args, fn, backend=DistributedBackend.NCCL):
    """
    Launch distributed training from command line arguments.
    This is a helper function for scripts.
    
    Args:
        args: Parsed command line arguments containing distributed settings
        fn: The function to run for each process
        backend: The backend to use
    """
    # Check if we're using torchrun already
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Initialize with torchrun environment
        world_size, rank = init_distributed(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        fn(rank=rank, local_rank=local_rank, world_size=world_size)
        cleanup_distributed()
        return
    
    # If not using torchrun, check if we should use mp.spawn
    if hasattr(args, 'multi_gpu') and args.multi_gpu:
        world_size = torch.cuda.device_count()
        if world_size > 1:
            print(f"Launching {world_size} processes with mp.spawn (fallback method)")
            print("Warning: It's recommended to use torchrun for distributed training")
            
            import torch.multiprocessing as mp
            
            # Define a wrapper function for mp.spawn
            def _wrapper(local_rank):
                rank = local_rank
                os.environ["RANK"] = str(rank)
                os.environ["LOCAL_RANK"] = str(local_rank)
                os.environ["WORLD_SIZE"] = str(world_size)
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "12355"
                
                init_distributed(backend=backend, world_size=world_size, rank=rank, local_rank=local_rank)
                fn(rank=rank, local_rank=local_rank, world_size=world_size)
                cleanup_distributed()
            
            mp.spawn(_wrapper, nprocs=world_size, join=True)
        else:
            print("Only one GPU found, running in single GPU mode")
            fn(rank=0, local_rank=0, world_size=1)
    else:
        # No distributed training, just run the function
        fn(rank=0, local_rank=0, world_size=1)
