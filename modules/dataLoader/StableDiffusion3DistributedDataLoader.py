import copy

from modules.dataLoader.StableDiffusion3BaseDataLoader import StableDiffusion3BaseDataLoader
from modules.dataLoader.mixin.DistributedDataLoaderMixin import DistributedDataLoaderMixin
from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress
from modules.util.distributed import is_distributed_available, get_rank, is_main_process

from mgds.MGDS import MGDS, TrainDataLoader

import torch


class StableDiffusion3DistributedDataLoader(
    StableDiffusion3BaseDataLoader,
    DistributedDataLoaderMixin,
):
    """
    StableDiffusion3 data loader with distributed training support.
    This extends the base data loader with distributed capabilities.
    """
    
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusion3Model,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        # Initialize the base data loader
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            config=config,
            model=model,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        
        # Store distributed-specific configuration
        self.is_distributed = config.distributed.enabled and is_distributed_available()
        self.rank = get_rank() if self.is_distributed else 0
        self.is_main = is_main_process() if self.is_distributed else True
        
        # Print distributed info on the main process
        if self.is_distributed and self.is_main:
            print(f"Initialized distributed data loader with {self.get_distributed_data_info(config)}")
    
    def get_data_set(self) -> MGDS:
        """
        Get the MGDS data set, prepared for distributed training if enabled.
        """
        dataset = super().get_data_set()
        
        if self.is_distributed:
            # Prepare the dataset for distributed training
            dataset = self.prepare_mgds_for_distributed(dataset, self._get_config())
            
        return dataset
    
    def get_data_loader(self) -> TrainDataLoader:
        """
        Get the data loader, wrapped for distributed training if enabled.
        """
        data_loader = super().get_data_loader()
        
        if self.is_distributed:
            # Wrap the data loader for distributed training
            data_loader = self.wrap_data_loader_for_distributed(
                data_loader=data_loader,
                config=self._get_config(),
                shuffle=True,
                drop_last=True
            )
            
        return data_loader
    
    def _get_config(self) -> TrainConfig:
        """
        Get the config from the data set.
        """
        return self.get_data_set().settings.get('config', None)
    
    def start_next_epoch(self) -> None:
        """
        Start the next epoch, updating distributed sampler if necessary.
        """
        dataset = self.get_data_set()
        data_loader = self.get_data_loader()
        
        # Set epoch for proper shuffling in distributed mode
        if self.is_distributed:
            self.set_epoch_for_distributed_sampler(data_loader, dataset.current_epoch)
        
        # Call the dataset's start_next_epoch method
        dataset.start_next_epoch()
