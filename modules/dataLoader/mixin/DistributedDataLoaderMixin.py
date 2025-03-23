"""
Mixin for distributed data loading capabilities.

This mixin provides common functionality for distributed data loading
that can be inherited by specific data loader implementations.
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from modules.util import distributed
from modules.util.config.TrainConfig import TrainConfig


class DistributedDataLoaderMixin:
    """
    Mixin providing distributed data loading capabilities.
    
    This mixin can be added to any data loader to provide distributed
    functionality such as dataset sharding, distributed sampler creation,
    and distributed caching.
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
    
    def get_distributed_dataloader(
        self,
        dataset: Dataset,
        config: TrainConfig,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn=None,
    ) -> DataLoader:
        """
        Create a data loader suitable for distributed training.
        
        Args:
            dataset: Dataset to load
            config: Training configuration
            batch_size: Batch size (per-GPU or global depending on strategy)
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker threads for loading
            collate_fn: Function to collate samples into a batch
            
        Returns:
            A data loader configured for distributed training
        """
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.get_world_size(),
            rank=distributed.get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        
        # Create the dataloader with the sampler
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle is handled by the sampler
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=collate_fn,
        )
    
    def get_distributed_cache_path(self, config: TrainConfig, base_path: str) -> str:
        """
        Get a distributed-aware cache path.
        
        For distributed training, each rank may have its own cache directory
        to avoid conflicts.
        
        Args:
            config: Training configuration
            base_path: Base path for the cache
            
        Returns:
            Path adjusted for distributed training
        """
        if not self.is_distributed(config):
            return base_path
            
        rank = distributed.get_rank()
        if config.distributed.latent_caching_strategy == distributed.DataDistributionStrategy.DISTRIBUTED:
            # Each rank gets its own cache directory
            return os.path.join(base_path, f"rank_{rank}")
        else:
            # Centralized caching - only rank 0 creates the cache, others use it
            return base_path
    
    def should_skip_caching(self, config: TrainConfig) -> bool:
        """
        Determine if this process should skip caching.
        
        For distributed caching with a centralized strategy, only rank 0
        should perform caching.
        
        Args:
            config: Training configuration
            
        Returns:
            True if this process should skip caching
        """
        if not self.is_distributed(config):
            return False
            
        if config.distributed.latent_caching_strategy == distributed.DataDistributionStrategy.CENTRALIZED:
            # For centralized caching, only rank 0 does the caching
            return distributed.get_rank() != 0
            
        return False
    
    def synchronize_after_epoch(self):
        """Synchronize all processes after an epoch."""
        if dist.is_initialized():
            dist.barrier()
