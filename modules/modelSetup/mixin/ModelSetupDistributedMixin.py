from typing import Dict, Optional, List, Any, Type

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed import (
    is_distributed_available, 
    is_main_process, 
    get_rank, 
    get_local_rank,
    wrap_model_for_ddp
)


class ModelSetupDistributedMixin:
    """
    A mixin that provides distributed training functionality for model setups.
    """
    
    def apply_ddp_to_model_component(
        self, 
        component: nn.Module,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        broadcast_buffers: bool = True,
        device_ids: Optional[List[int]] = None
    ) -> DDP:
        """
        Wraps a model component with DDP.
        
        Args:
            component: The model component to wrap.
            find_unused_parameters: Whether to find unused parameters in the forward pass.
            gradient_as_bucket_view: Whether to use gradient as bucket view for memory efficiency.
            broadcast_buffers: Whether to broadcast buffers in the beginning of forward pass.
            device_ids: Device IDs to use. Defaults to [local_rank].
            
        Returns:
            The wrapped model component.
        """
        # If distributed training is not enabled or available, return the component as is
        if not is_distributed_available():
            return component
            
        return wrap_model_for_ddp(
            component,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            broadcast_buffers=broadcast_buffers
        )
    
    def setup_for_distributed_training(
        self,
        model: BaseModel,
        config: TrainConfig
    ):
        """
        Sets up a model for distributed training based on the configuration.
        Override this method in your model setup implementation.
        
        Args:
            model: The model to set up for distributed training.
            config: The training configuration.
        """
        # To be implemented by specific model setup classes
        pass
        
    def should_save_in_distributed(self) -> bool:
        """
        Determines if the current process should save model checkpoints.
        In distributed mode, typically only the main process (rank 0) saves.
        
        Returns:
            bool: True if this process should save, False otherwise.
        """
        return is_main_process()
    
    def unwrap_ddp_modules(self, model: BaseModel):
        """
        Unwraps DDP modules for saving or evaluation.
        Override this method in your model setup implementation.
        
        Args:
            model: The model to unwrap DDP modules from.
        """
        # To be implemented by specific model setup classes
        pass
    
    def is_distributed_training_enabled(self, config: TrainConfig) -> bool:
        """
        Checks if distributed training is enabled and available.
        
        Args:
            config: The training configuration.
            
        Returns:
            bool: True if distributed training is enabled and available.
        """
        return config.distributed.enabled and is_distributed_available()
    
    def get_ddp_params(self, config: TrainConfig) -> Dict[str, Any]:
        """
        Gets parameters for DDP based on the configuration.
        
        Args:
            config: The training configuration.
            
        Returns:
            Dict containing DDP parameters.
        """
        return {
            "find_unused_parameters": config.distributed.find_unused_parameters,
            "gradient_as_bucket_view": config.distributed.gradient_as_bucket_view,
            "broadcast_buffers": True,  # Usually best to keep this True
        }
