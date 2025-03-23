#!/usr/bin/env python3
"""
Simple script to verify multi-GPU training works with PyTorch DDP
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.net(x)

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_worker(rank, world_size):
    print(f"Running DDP worker process on rank {rank}.")
    setup(rank, world_size)

    # Create model and move it to the correct device
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create dummy data
    inputs = torch.randn(20, 10, device=rank)
    labels = torch.randn(20, 1, device=rank)
    
    # Set up optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Run a few training steps
    for step in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = ((outputs - labels) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, step {step}, loss: {loss.item()}")
    
    cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU training requires CUDA support.")
        return 1
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Multi-GPU training requires at least 2 GPUs, but only {world_size} detected.")
        return 1
    
    print(f"Starting test with {world_size} GPUs")
    
    # Use spawn to launch all processes
    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    print("Multi-GPU test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())