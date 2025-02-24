import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.util.config.TrainConfig import TrainConfig
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed training script')
    parser.add_argument('--config', type=str, help='Path to config file')
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
    """Main training function"""
    args = parse_args()
    
    # Set up distributed training
    setup_distributed(args.local_rank)
    
    try:
        # Load config
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                config = TrainConfig.default_values().from_dict(config_dict)
        else:
            config = TrainConfig.default_values()
        
        # Enable FSDP
        config.enable_fsdp = True
        
        # Get world size
        if dist.is_initialized():
            config.fsdp_num_gpus = dist.get_world_size()
        else:
            config.fsdp_num_gpus = torch.cuda.device_count()
        
        # Create callbacks and commands
        callbacks = TrainCallbacks()
        commands = TrainCommands()
        
        # Create and start trainer
        trainer = GenericTrainer(config, callbacks, commands)
        trainer.start()
        
        while not trainer.should_stop():
            trainer.train_step()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if trainer is not None:
            trainer.end()
        cleanup_distributed()

if __name__ == "__main__":
    main()
