import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    """Run training on each GPU"""
    setup(rank, world_size)
    
    # Import here to avoid circular imports
    from modules.ui.TrainUI import TrainUI
    from modules.util.config.TrainConfig import TrainConfig
    
    # Load config and enable FSDP
    config = TrainConfig.default_values()
    config.enable_fsdp = True
    config.fsdp_num_gpus = world_size
    
    # Start training
    ui = TrainUI()
    ui.train(config)
    
    cleanup()

def main():
    """Main entry point for distributed training"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU training requires CUDA.")
        sys.exit(1)
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Found only {world_size} GPU(s). Multi-GPU training requires at least 2 GPUs.")
        sys.exit(1)
    
    print(f"Starting distributed training with {world_size} GPUs...")
    
    # Start processes for each GPU
    try:
        mp.spawn(
            run_training,
            args=(world_size, sys.argv[1:]),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error during distributed training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
