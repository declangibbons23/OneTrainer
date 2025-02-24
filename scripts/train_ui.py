import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.ui.TrainUI import TrainUI

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training UI')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def setup_distributed(local_rank):
    """Set up distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(local_rank)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.barrier()  # Ensure all processes are ready to clean up
        dist.destroy_process_group()

def main():
    """Main UI function"""
    args = parse_args()
    
    # Set up distributed training if running with multiple GPUs
    if torch.cuda.device_count() > 1:
        setup_distributed(args.local_rank)
    
    try:
        # Only create UI on primary GPU (or in non-distributed mode)
        if not dist.is_initialized() or dist.get_rank() == 0:
            ui = TrainUI()
            ui.mainloop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
