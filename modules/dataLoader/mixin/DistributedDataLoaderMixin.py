from abc import ABCMeta
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DistributedSampler

from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed import (
    is_distributed_available,
    get_rank,
    get_world_size,
    is_main_process,
    DataDistributionStrategy
)

from mgds.TrainDataLoader import TrainDataLoader
from mgds.MGDS import MGDS


class DistributedDataLoaderMixin(metaclass=ABCMeta):
    """
    A mixin that enhances data loaders with distributed training capabilities.
    It provides methods to convert a regular data loader to a distributed one.
    """
    
    def wrap_data_loader_for_distributed(
        self,
        data_loader: TrainDataLoader,
        config: TrainConfig,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> TrainDataLoader:
        """
        Wrap a data loader with a DistributedSampler for distributed training.
        
        Args:
            data_loader: The original data loader
            config: Training configuration
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            TrainDataLoader with distributed sampling
        """
        if not config.distributed.enabled or not is_distributed_available():
            return data_loader
            
        if config.distributed.data_loading_strategy == DataDistributionStrategy.CENTRALIZED:
            # In centralized mode, only rank 0 loads data
            if not is_main_process():
                # Create an empty or minimal dataset for non-main processes
                # They'll receive data via communication
                return data_loader  # For now, return as is - needs implementation
            else:
                return data_loader
                
        # For distributed strategy - each GPU loads its own data portion
        dataset = data_loader.dataset
        
        # Create a distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        # Apply the sampler to the data loader
        # Assuming MGDS data loader supports setting a sampler
        data_loader.sampler = sampler
        data_loader.shuffle = False  # The sampler will handle shuffling
        
        return data_loader
    
    def prepare_mgds_for_distributed(
        self,
        mgds: MGDS,
        config: TrainConfig
    ) -> MGDS:
        """
        Prepare an MGDS for distributed training.
        
        Args:
            mgds: The MGDS instance
            config: Training configuration
            
        Returns:
            MGDS prepared for distributed training
        """
        if not config.distributed.enabled or not is_distributed_available():
            return mgds
            
        # Modify batch size based on distribution strategy
        if config.distributed.data_loading_strategy == DataDistributionStrategy.DISTRIBUTED:
            # Keep the same global batch size by adjusting per-GPU batch size
            # This is optional - sometimes keeping the same per-GPU batch size is preferred
            # mgds.batch_size = mgds.batch_size // get_world_size()
            pass
            
        # Set appropriate flags or configurations for distributed mode
        # mgds.distributed_rank = get_rank()
        # mgds.distributed_world_size = get_world_size()
        
        # MGDS might need additional configuration for distributed training
        # Adjust based on the actual implementation of MGDS
        
        return mgds
    
    def get_distributed_data_info(self, config: TrainConfig) -> Dict[str, Any]:
        """
        Get information about the distributed training environment.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary with distributed training information
        """
        if not config.distributed.enabled or not is_distributed_available():
            return {
                "distributed": False
            }
            
        return {
            "distributed": True,
            "rank": get_rank(),
            "world_size": get_world_size(),
            "is_main_process": is_main_process(),
            "data_strategy": config.distributed.data_loading_strategy,
        }
        
    def set_epoch_for_distributed_sampler(self, data_loader: TrainDataLoader, epoch: int):
        """
        Set the epoch for the DistributedSampler to ensure proper shuffling.
        Call this at the beginning of each epoch.
        
        Args:
            data_loader: The data loader with distributed sampler
            epoch: Current epoch number
        """
        if hasattr(data_loader, 'sampler') and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
