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
        # Verify multi-GPU settings
        if not train_config.enable_multi_gpu:
            raise ValueError("DistributedTrainer requires enable_multi_gpu=True in config")
            
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for distributed training")
            
        # Set devices
        train_device = f"cuda:{local_rank}"
        temp_device = "cpu"  # Always use CPU for temp storage in distributed mode
        
        # Initialize parent
        super().__init__(train_config, TrainArgs(), torch.device(train_device), torch.device(temp_device))
        
        # Store rank info
        self.rank = local_rank
        self.world_size = train_config.world_size or torch.cuda.device_count()
        self.is_main_process = (local_rank == 0)
        
        # Store callbacks/commands (only main process gets these)
        self.callbacks = callbacks if self.is_main_process else None
        self.commands = commands if self.is_main_process else None
        
        # Initialize process group if needed
        if not dist.is_initialized():
            self._setup_distributed()
            
        # Log initialization
        if self.is_main_process:
            logger.info(f"Initialized DistributedTrainer with {self.world_size} processes")
            logger.info(f"Using {train_config.distributed_backend} backend")
            logger.info(f"Process {local_rank}/{self.world_size-1}")
        
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
    
    def _setup_distributed(self):
        """Initialize the distributed training environment"""
        try:
            # Get distributed parameters from config
            backend = self.train_config.distributed_backend
            port = getattr(self.train_config, 'dist_port', 12355)
            
            # Initialize environment variables
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)
            
            # Initialize process group
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Set device for this process
            torch.cuda.set_device(self.rank)
            
            if self.is_main_process:
                logger.info(f"Initialized distributed process group:")
                logger.info(f"  Backend: {backend}")
                logger.info(f"  World size: {self.world_size}")
                logger.info(f"  Rank: {self.rank}")
                logger.info(f"  Port: {port}")
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {e}")
            raise
            
    def start(self):
        """Start the training process"""
        if self.is_main_process:
            logger.info("Starting distributed training...")
            
        # Call parent start method
        super().start()
        
    def train_initialize(self):
        """Initialize training for distributed mode"""
        if self.is_main_process:
            logger.info("Initializing distributed training...")
            
        # Initialize base trainer
        super().train_initialize()
        
        # Scale learning rate if enabled
        if self.train_config.lr_scaling:
            original_lr = self.train_config.learning_rate
            scaled_lr = original_lr * self.world_size
            if self.is_main_process:
                logger.info(f"Scaling learning rate for {self.world_size} GPUs: {original_lr} -> {scaled_lr}")
            self.train_config.learning_rate = scaled_lr
        
        # Set up distributed sampler if enabled
        if self.train_config.distributed_data_loading:
            if hasattr(self.model, 'train_dataloader'):
                from torch.utils.data.distributed import DistributedSampler
                
                # Create distributed sampler
                sampler = DistributedSampler(
                    self.model.train_dataloader.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True
                )
                
                # Update dataloader with distributed sampler
                self.model.train_dataloader = torch.utils.data.DataLoader(
                    self.model.train_dataloader.dataset,
                    batch_size=self.train_config.batch_size,
                    sampler=sampler,
                    num_workers=self.train_config.dataloader_threads,
                    pin_memory=True
                )
                
                if self.is_main_process:
                    logger.info("Initialized distributed data loading")

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
        """Run the distributed training loop"""
        try:
            # Initialize training
            self.train_initialize()
            
            if self.is_main_process:
                logger.info("Starting distributed training loop...")
            
            # Wrap model in DDP if not already wrapped
            if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.rank],
                    output_device=self.rank
                )
                if self.is_main_process:
                    logger.info("Model wrapped in DistributedDataParallel")
            
            # Main training loop
            while not self.is_finished():
                try:
                    self.train_step()
                    
                    # Synchronize at regular intervals
                    if self.model.train_progress.global_step % 100 == 0:
                        self.sync_model_state()
                        if self.is_main_process:
                            logger.info(f"Step {self.model.train_progress.global_step}: Training progressing normally")
                except Exception as step_error:
                    logger.error(f"Error in training step on rank {self.rank}: {step_error}")
                    raise
            
            # Final synchronization
            self.sync_model_state()
            dist.barrier()  # Make sure all processes are done
            
            # Only main process handles final tasks
            if self.is_main_process:
                if not self.train_config.save_skip_first and self.model.train_progress.epoch_progress >= 1.0:
                    self.save_model(True)
                logger.info("Distributed training completed successfully")
                
        except Exception as e:
            logger.error(f"Error during training on rank {self.rank}: {e}")
            import traceback
            traceback.print_exc()
            # Make sure other processes know about the error
            if dist.is_initialized():
                dist.destroy_process_group()
            raise
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
        """Run a single distributed training step"""
        try:
            # Set train mode
            self.model.train()
            
            # Forward pass and loss calculation
            loss = super().train_step()
            
            # Synchronize gradients across processes
            if dist.is_initialized():
                for param in self.model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= self.world_size
            
            # Log progress on main process
            if self.is_main_process and self.callbacks:
                self.callbacks.on_update_status(f"Training step completed (loss: {loss:.4f})")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in distributed training step on rank {self.rank}: {e}")
            import traceback
            traceback.print_exc()
            raise
        return super().train_step()