import os
import socket
import logging
import torch
import torch.distributed as dist
from typing import Optional

logger = logging.getLogger(__name__)

def get_free_port():
    """Get a free port on the local machine"""
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def setup_distributed(rank: int, world_size: int, backend: str = "nccl", port: Optional[int] = None):
    """
    Initialize the distributed environment
    
    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        backend: The backend to use (nccl for GPU, gloo for CPU)
        port: The port to use for communication. If None, a random free port will be used.
    """
    if port is None:
        if rank == 0:
            port = get_free_port()
            os.environ["MASTER_PORT"] = str(port)
        else:
            # Other ranks wait for master to set the port
            if "MASTER_PORT" not in os.environ:
                raise RuntimeError("MASTER_PORT environment variable not set")
            port = int(os.environ["MASTER_PORT"])
    else:
        os.environ["MASTER_PORT"] = str(port)
        
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    logger.info(f"Initialized process group: rank={rank}, world_size={world_size}, backend={backend}")
    
    # Synchronize all processes at this point
    dist.barrier()

def cleanup_distributed():
    """Clean up the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()
        
def get_rank():
    """Get the rank of the current process if distributed is initialized, otherwise 0"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """Get the world size if distributed is initialized, otherwise 1"""
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0

def cleanup_additional_ports():
    """Clean up environment variables used for distributed training"""
    keys_to_clean = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]
    for key in keys_to_clean:
        if key in os.environ:
            del os.environ[key]