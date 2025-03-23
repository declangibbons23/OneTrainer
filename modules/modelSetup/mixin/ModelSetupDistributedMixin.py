"""
Mixin for distributed model setup capabilities.

This mixin provides common functionality for setting up models for
distributed training that can be inherited by specific model setup implementations.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.model.BaseModel import BaseModel
from modules.util import distributed
from modules.util.config.TrainConfig import TrainConfig


class ModelSetupDistributedMixin:
    """
    Mixin providing distributed model setup capabilities.
    
    This mixin can be added to any model setup to provide distributed
    functionality such as wrapping models with DDP, optimizing for
    distributed training, and ensuring proper synchronization.
    """
    
    def is_distributed(self, config: TrainConfig) -> bool:
        """
        Check if distributed training is enabled and available.
        
        Args:
            config: Training configuration
            
        Returns:
            True if distributed training is enabled and available
        """
        return (hasattr(config, 'distributed') and 
                config.distributed and 
                config.distributed.enabled and
                distributed.is_distributed_available())
    
    def wrap_model_for_distributed(
        self,
        model: nn.Module,
        device_id: int,
        config: TrainConfig
    ) -> nn.Module:
        """
        Wrap a model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            device_id: Device ID
            config: Training configuration
            
        Returns:
            The model wrapped with DDP if distributed training is enabled
        """
        if not self.is_distributed(config):
            return model
        
        print(f"[Rank {distributed.get_rank()}] Wrapping model with DDP")
        
        # Get DDP settings from config
        find_unused_parameters = config.distributed.find_unused_parameters
        gradient_as_bucket_view = config.distributed.gradient_as_bucket_view
        bucket_cap_mb = config.distributed.bucket_cap_mb or 25
        
        # Move model to the correct device
        model = model.to(device_id)
        
        # Wrap model with DDP
        wrapped_model = DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            bucket_cap_mb=bucket_cap_mb,
        )
        
        return wrapped_model
    
    def setup_model_for_distributed(
        self,
        model: BaseModel,
        config: TrainConfig
    ) -> BaseModel:
        """
        Set up a model for distributed training.
        
        This method sets up the model parts that need to be wrapped with DDP.
        
        Args:
            model: Model to set up
            config: Training configuration
            
        Returns:
            The model set up for distributed training
        """
        if not self.is_distributed(config):
            return model
        
        # Each subclass should implement this method
        # based on its specific model structure
        raise NotImplementedError("Subclasses must implement this method")
    
    def setup_optimizer_for_distributed(
        self,
        model: BaseModel,
        config: TrainConfig
    ) -> None:
        """
        Set up an optimizer for distributed training.
        
        This method ensures the optimizer is properly configured for
        distributed training, such as adjusting learning rates based
        on the number of GPUs.
        
        Args:
            model: Model to set up optimizer for
            config: Training configuration
        """
        if not self.is_distributed(config):
            return
        
        # If we're scaling learning rate by world size
        if hasattr(config, 'scale_lr_by_world_size') and config.scale_lr_by_world_size:
            world_size = distributed.get_world_size()
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= world_size
                if distributed.is_main_process():
                    print(f"Scaled learning rate to {param_group['lr']} (original * {world_size})")
    
    def synchronize_model(self, model: BaseModel) -> None:
        """
        Synchronize model parameters across all processes.
        
        This is useful to ensure all processes have the same model
        parameters at the start of training.
        
        Args:
            model: Model to synchronize
        """
        if not dist.is_initialized():
            return
        
        # Broadcast model parameters from rank 0
        for param in model.parameters.parameters():
            dist.broadcast(param.data, src=0)
        
        # Synchronize all processes
        dist.barrier()
