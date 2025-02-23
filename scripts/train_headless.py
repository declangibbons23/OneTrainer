import os
import sys
from pathlib import Path

import torch

from modules.util.config.TrainConfig import TrainConfig
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands

def main():
    """Headless training entry point"""
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

if __name__ == "__main__":
    main()
