import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    """Run training on each GPU"""
    setup(rank, world_size)
    
    # Set CUDA device for this process
    torch.cuda.set_device(rank)
    
    # Import here to avoid circular imports
    from modules.ui.TrainUI import TrainUI
    from modules.util.config.TrainConfig import TrainConfig
    
    # Start training based on mode
    if '--headless' in args:
        # Headless mode
        from scripts.train_headless import main as train_headless
        # Remove --headless from args
        args = [arg for arg in args if arg != '--headless']
        train_headless(args)
    else:
        # GUI mode
        ui = TrainUI()
        ui.start()
    
    cleanup()

def main():
    """Main entry point for distributed training"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU training requires CUDA.")
        sys.exit(1)
    
    # Get number of GPUs from environment or available devices
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
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
