from util.import_util import script_imports

script_imports()

import json
import os
import sys
import torch
import torch.multiprocessing as mp

from modules.trainer.GenericTrainer import GenericTrainer
from modules.trainer.DistributedTrainer import DistributedTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed_util import setup_distributed, cleanup_distributed


def train_process(local_rank, world_size, args):
    """
    Function to run on each process for distributed training
    """
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    # Load training configuration
    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))
    
    # Apply distributed training settings from args to config
    train_config.enable_multi_gpu = True
    train_config.distributed_backend = args.distributed_backend
    train_config.use_torchrun = not args.no_torchrun
    train_config.distributed_data_loading = not args.single_gpu_data_loading
    train_config.local_rank = local_rank
    train_config.world_size = world_size

    # Load secrets
    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict = json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    # Setup distributed process group
    setup_distributed(train_config)
    
    # Create distributed trainer
    trainer = DistributedTrainer(train_config, callbacks, commands, local_rank)

    # Start training
    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    # Finalize training
    if not canceled or train_config.backup_before_save:
        trainer.end()
    
    # Cleanup distributed process group
    cleanup_distributed()


def launch_distributed_training(args):
    """
    Launch distributed training using torch.multiprocessing
    """
    # Determine world size (number of GPUs)
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: Less than 2 GPUs detected. Falling back to single GPU training.")
        single_gpu_training(args)
        return
    
    print(f"Launching distributed training on {world_size} GPUs...")
    
    # Use mp.spawn to start world_size worker processes
    mp.spawn(
        train_process,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


def single_gpu_training(args):
    """
    Run training on a single GPU
    """
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict = json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


def main():
    """
    Main entry point
    """
    args = TrainArgs.parse_args()
    
    # Check if running with torchrun (environment variable LOCAL_RANK is set)
    if "LOCAL_RANK" in os.environ:
        # When using torchrun, we initialize the distributed process directly
        args.multi_gpu = True
        args.local_rank = int(os.environ["LOCAL_RANK"])
        train_process(args.local_rank, int(os.environ.get("WORLD_SIZE", "1")), args)
    elif args.multi_gpu and not args.no_torchrun:
        # Print instructions for torchrun
        print("Multi-GPU training with torchrun requires launching via torchrun. Example:")
        print(f"torchrun --nproc_per_node={torch.cuda.device_count()} {' '.join(sys.argv)}")
        print("Falling back to single GPU training. Use --no-torchrun to use torch.multiprocessing instead.")
        single_gpu_training(args)
    elif args.multi_gpu:
        # Launch training with torch.multiprocessing.spawn
        launch_distributed_training(args)
    else:
        # Single GPU training
        single_gpu_training(args)


if __name__ == '__main__':
    # Required for Windows support with multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
