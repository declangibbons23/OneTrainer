"""
Distributed data loader for Stable Diffusion 3 models.
"""

import os
import torch
from torch.utils.data import DataLoader

from modules.dataLoader.mixin.DistributedDataLoaderMixin import DistributedDataLoaderMixin
from modules.dataLoader.StableDiffusion3BaseDataLoader import StableDiffusion3BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.util import distributed
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusion3DistributedDataLoader(StableDiffusion3BaseDataLoader, DistributedDataLoaderMixin):
    """
    Distributed data loader for Stable Diffusion 3 models.
    
    This class extends the base SD3 data loader with distributed capabilities
    provided by the DistributedDataLoaderMixin.
    """
    
    def __init__(self, model: BaseModel, train_progress: TrainProgress, config: TrainConfig, is_validation: bool = False):
        """
        Initialize the distributed data loader.
        
        Args:
            model: Model to load data for
            train_progress: Current training progress
            config: Training configuration
            is_validation: Whether this is a validation data loader
        """
        super().__init__(model, train_progress, config, is_validation)
        
        # Print distributed info
        if self.is_distributed(config):
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
            print(f"[Rank {rank}] Initializing StableDiffusion3DistributedDataLoader (World Size: {world_size})")
            
            # Log if we're using sharded datasets
            strategy = config.distributed.data_loading_strategy
            print(f"[Rank {rank}] Using {strategy} data loading strategy")
    
    def _get_cache_dir(self) -> str:
        """
        Get the cache directory, adjusted for distributed training.
        
        In distributed training, we may want separate cache directories
        for each process to avoid conflicts.
        
        Returns:
            Path to the cache directory
        """
        base_cache_dir = super()._get_cache_dir()
        
        # Use distributed-aware path if distributed training is enabled
        return self.get_distributed_cache_path(self.config, base_cache_dir)
    
    def _should_skip_caching(self) -> bool:
        """
        Determine if this process should skip caching.
        
        For distributed training with a centralized caching strategy,
        only the main process (rank 0) should perform caching.
        
        Returns:
            True if this process should skip caching
        """
        # Check if we should skip based on distributed settings
        if self.should_skip_caching(self.config):
            print(f"[Rank {distributed.get_rank()}] Skipping caching (using centralized strategy)")
            return True
        
        # Otherwise, use the standard check
        return super()._should_skip_caching()
    
    def _create_data_loader(self, dataset) -> DataLoader:
        """
        Create a data loader for the dataset.
        
        For distributed training, we use a DistributedSampler to ensure
        each process gets a different subset of the data.
        
        Args:
            dataset: Dataset to create a loader for
            
        Returns:
            DataLoader for the dataset
        """
        if self.is_distributed(self.config):
            # Use distributed data loader
            return self.get_distributed_dataloader(
                dataset=dataset,
                config=self.config,
                batch_size=self.batch_size,
                shuffle=not self.is_validation,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            # Use standard data loader
            return super()._create_data_loader(dataset)
    
    def _finish_epoch_processing(self):
        """
        Finish processing for the current epoch.
        
        For distributed training, we synchronize processes after
        each epoch to ensure all processes start the next epoch together.
        """
        # Standard epoch finishing
        super()._finish_epoch_processing()
        
        # Synchronize processes if distributed
        if self.is_distributed(self.config):
            self.synchronize_after_epoch()
