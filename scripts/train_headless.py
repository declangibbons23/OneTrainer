import argparse
import json
import os
import sys
from pathlib import Path

import torch

from modules.util.config.TrainConfig import TrainConfig
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Headless training script')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def main(args=None):
    """Headless training entry point"""
    if args is None:
        args = sys.argv[1:]
    args = parse_args()

    # Initialize process group for distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(args.local_rank)

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = TrainConfig.default_values().from_dict(config_dict)
    else:
        config = TrainConfig.default_values()
    
    # Enable FSDP
    config.enable_fsdp = True
    
    # Get world size if running in distributed mode
    if torch.distributed.is_initialized():
        config.fsdp_num_gpus = torch.distributed.get_world_size()
    else:
        config.fsdp_num_gpus = torch.cuda.device_count()
    
    # Create callbacks and commands
    callbacks = TrainCallbacks()
    commands = TrainCommands()
    
    # Create and start trainer
    trainer = GenericTrainer(config, callbacks, commands)
    trainer.start()
    
    try:
        while not trainer.should_stop():
            trainer.train_step()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        trainer.end()

        # Clean up distributed training
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
