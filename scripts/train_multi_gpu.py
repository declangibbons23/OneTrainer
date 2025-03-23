#!/usr/bin/env python3

# Import the script_imports utility first to set up Python path correctly
try:
    from util.import_util import script_imports
    script_imports()
except ImportError:
    import sys
    import os
    # Add the project root to Python path if needed
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    try:
        from util.import_util import script_imports
        script_imports()
    except ImportError as e:
        print(f"Error importing script_imports: {e}")
        print("Make sure you're running this script from the OneTrainer directory.")
        sys.exit(1)

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install PyTorch with: pip install torch")
    sys.exit(1)

# Check for CUDA support
if not torch.cuda.is_available():
    logger.error("CUDA is not available. Multi-GPU training requires CUDA support.")
    logger.error("Please install CUDA and PyTorch with CUDA support.")
    sys.exit(1)

# Import OneTrainer modules
try:
    from modules.util.args.TrainArgs import TrainArgs
    from modules.util.config.TrainConfig import TrainConfig
    from modules.trainer.DistributedTrainer import DistributedTrainer
    from modules.util.distributed_util import setup_distributed, cleanup_distributed
    from modules.util.callbacks.TrainCallbacks import TrainCallbacks
    from modules.util.commands.TrainCommands import TrainCommands
except ImportError as e:
    logger.error(f"Failed to import OneTrainer modules: {e}")
    logger.error("Make sure you're running this script from the OneTrainer directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU training script")
    parser.add_argument("--config-path", required=True, help="Path to the JSON config file")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use (0 for all available)")
    parser.add_argument("--spawn", action="store_true", help="Use torch.multiprocessing.spawn instead of torchrun")
    parser.add_argument("--port", type=int, default=12355, help="Port for distributed communication")
    parser.add_argument("--distributed-backend", type=str, default="nccl", choices=["nccl", "gloo"], 
                        help="Distributed backend to use")
    
    # Add arguments for distributed data loading
    parser.add_argument("--distributed-data-loading", action="store_true", 
                      help="Enable distributed data loading")
    
    return parser.parse_args()

def get_world_size(args):
    """Get world size (number of GPUs to use)"""
    if args.num_gpus > 0:
        return min(args.num_gpus, torch.cuda.device_count())
    else:
        return torch.cuda.device_count()

def train_worker(rank, world_size, args):
    """Worker function for each GPU process"""
    # Set up the distributed environment
    setup_distributed(rank, world_size, args.distributed_backend, args.port)
    
    # Set CUDA device for this process
    torch.cuda.set_device(rank)
    
    # Load the training configuration
    with open(args.config_path, 'r') as f:
        config_json = json.load(f)
    
    # Set up multi-GPU specific config options
    config_json["enable_multi_gpu"] = True
    config_json["world_size"] = world_size
    config_json["rank"] = rank
    config_json["distributed_backend"] = args.distributed_backend
    config_json["distributed_data_loading"] = args.distributed_data_loading
    
    # Create the train config object
    train_config = TrainConfig.from_dict(config_json)
    
    # Create arguments object
    train_args = TrainArgs()
    train_args.config_path = args.config_path
    
    # Only the master process should save models and show progress
    is_master = rank == 0
    
    # Create callbacks and commands for training
    callbacks = TrainCallbacks()
    commands = TrainCommands()
    
    # Set devices in config
    train_config.train_device = f"cuda:{rank}"
    train_config.temp_device = "cpu"
    train_config.local_rank = rank
    
    # Setup the trainer with proper parameters
    trainer = DistributedTrainer(
        train_config=train_config,
        callbacks=callbacks,
        commands=commands,
        local_rank=rank
    )
    
    # Run the training
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training on rank {rank}: {e}")
        raise e
    finally:
        # Clean up the distributed environment
        cleanup_distributed()

def main():
    """Main entry point for the script"""
    args = parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Cannot run multi-GPU training.")
        return 1
    
    # Determine the number of GPUs to use
    world_size = get_world_size(args)
    
    if world_size < 2:
        logger.error(f"Multi-GPU training requires at least 2 GPUs, but only {world_size} available.")
        return 1
    
    logger.info(f"Starting multi-GPU training with {world_size} GPUs")
    
    if args.spawn:
        # Using torch.multiprocessing.spawn
        logger.info("Using torch.multiprocessing.spawn to launch processes")
        mp.spawn(
            train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Using torchrun (recommended)
        # When running with torchrun, this script will be executed once per GPU
        # with environment variables set by torchrun
        if "LOCAL_RANK" not in os.environ:
            # This is the initial call - prepare torchrun command
            logger.info("Using torchrun to launch processes")
            
            # Construct torchrun command
            script_path = Path(__file__).resolve()
            cmd = [
                sys.executable,
                "-m", "torch.distributed.run",
                f"--nproc_per_node={world_size}",
                f"--master_port={args.port}",
                str(script_path),
                f"--config-path={args.config_path}",
                f"--num-gpus={world_size}",
                f"--port={args.port}",
                f"--distributed-backend={args.distributed_backend}"
            ]
            
            if args.distributed_data_loading:
                cmd.append("--distributed-data-loading")
            
            # Execute torchrun
            logger.info(f"Executing: {' '.join(cmd)}")
            os.execv(sys.executable, cmd)
        else:
            # This is running inside torchrun, get rank from environment
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ.get("RANK", local_rank))
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
            
            # Run the worker function directly
            train_worker(global_rank, world_size, args)
    
    logger.info("Multi-GPU training completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())