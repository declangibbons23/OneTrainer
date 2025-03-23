import os
import time
import logging
import torch
import torch.distributed as dist
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed_util import get_rank, get_world_size, is_main_process

logger = logging.getLogger(__name__)

class DistributedTrainer(BaseTrainer):
    """
    Extended trainer class that supports distributed training across multiple GPUs.
    """
    def __init__(
            self,
            train_config: TrainConfig,
            callbacks=None,
            commands=None,
            local_rank: int = 0,
    ):
        # Get devices from config or use defaults
        train_device = train_config.train_device if hasattr(train_config, 'train_device') else f"cuda:{local_rank}"
        temp_device = train_config.temp_device if hasattr(train_config, 'temp_device') else "cpu"
        
        # Create dummy train_args if not using it
        train_args = TrainArgs()
        
        # Call parent init
        super().__init__(train_config, train_args, torch.device(train_device), torch.device(temp_device))
        
        # Set up callbacks and commands if provided
        self.callbacks = callbacks
        self.commands = commands
        
        # Initialize distributed training properties
        self.rank = local_rank
        self.world_size = train_config.world_size if hasattr(train_config, 'world_size') else 1
        self.is_main_process = local_rank == 0  # Only the first process is the main process
        
        # Configure logging based on rank
        if not self.is_main_process:
            # Reduce logging noise on non-main processes
            logging.getLogger().setLevel(logging.WARNING)
    
    def pre_training_setup(self, model: BaseModel):
        """
        Perform setup before training starts.
        
        Args:
            model: The model to train
        """
        # Call parent method
        super().pre_training_setup(model)
        
        # Print distributed training information
        if self.is_main_process:
            logger.info(f"Starting distributed training with {self.world_size} GPUs")
            logger.info(f"Using {self.train_config.distributed_backend} backend")
            logger.info(f"Distributed data loading: {self.train_config.distributed_data_loading}")
    
    def start(self):
        """
        Start training process, initialize distributed environment if needed.
        """
        # If distributed environment is not yet initialized, do it now
        if not dist.is_initialized():
            from modules.util.distributed_util import setup_distributed
            
            # Get distributed parameters from config
            local_rank = getattr(self.train_config, 'local_rank', 0)
            world_size = getattr(self.train_config, 'world_size', torch.cuda.device_count())
            backend = getattr(self.train_config, 'distributed_backend', 'nccl')
            
            # Use default port or random port
            port = getattr(self.train_config, 'dist_port', 12355)
            
            logger.info(f"Initializing distributed environment: rank={local_rank}, world_size={world_size}")
            setup_distributed(local_rank, world_size, backend, port)
            
            # Update rank and world size based on initialized values
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_main_process = self.rank == 0
            
            # Set device for this process
            torch.cuda.set_device(self.rank)
        
        # Call parent start method
        super().start()

    def post_loading_setup(self, model: BaseModel):
        """
        Perform setup after model loading.
        
        Args:
            model: The loaded model
        """
        # Call parent method
        super().post_loading_setup(model)
        
        # Scale learning rate if needed
        if hasattr(self.train_config, 'learning_rate_scaling') and self.train_config.learning_rate_scaling:
            self.model_setup.scale_learning_rate_for_ddp(self.train_config)
    
    def get_save_checkpoint_sub_paths(self, step_count: int, path: str) -> list[str]:
        """
        Get the sub paths for saving checkpoints.
        
        Args:
            step_count: The current step count
            path: The base path for saving
            
        Returns:
            List of checkpoint sub paths
        """
        # Only the main process should save checkpoints
        if not self.is_main_process:
            return []
        
        return super().get_save_checkpoint_sub_paths(step_count, path)
    
    def sync_model_state(self):
        """
        Synchronize model state across processes.
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Synchronize all processes at this point
            dist.barrier()
    
    def train(self):
        """
        Run the training loop.
        """
        try:
            # Initialize training
            self.train_initialize()
            
            # Main training loop
            while not self.is_finished():
                self.train_step()
                
                # Synchronize at regular intervals
                if self.model.train_progress.global_step % 100 == 0:
                    self.sync_model_state()
            
            # Make sure all processes are done before saving final model
            self.sync_model_state()
            
            # Only the main process saves the final model
            if self.is_main_process:
                # Final model save
                if not self.train_config.save_skip_first and self.model.train_progress.epoch_progress >= 1.0:
                    self.save_model(True)
                
                logger.info("Training finished successfully")
        except Exception as e:
            logger.error(f"Error during training on rank {self.rank}: {e}")
            # Re-raise to ensure proper cleanup
            raise
    
    def save_model(self, is_final: bool = False):
        """
        Save the model.
        
        Args:
            is_final: Whether this is the final save
        """
        # Only the main process should save the model
        if not self.is_main_process:
            return
        
        # Call parent method
        super().save_model(is_final)
    
    def backup_checkpoint(self, step_count: int):
        """
        Backup checkpoint.
        
        Args:
            step_count: The current step count
        """
        # Only the main process should create backups
        if not self.is_main_process:
            return
        
        # Call parent method
        super().backup_checkpoint(step_count)
    
    def sample(self, step_count: int):
        """
        Generate samples.
        
        Args:
            step_count: The current step count
        """
        # Only the main process should generate samples
        if not self.is_main_process:
            return
        
        # Call parent method
        super().sample(step_count)
    
    def train_step(self):
        """
        Run a single training step.
        """
        # Use base class implementation
        return super().train_step()