from abc import ABCMeta, abstractmethod

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.DistributedMGDS import DistributedMGDS, DistributedTrainDataLoader

import torch


class BaseDataLoader(metaclass=ABCMeta):
    """Base class for all data loaders"""

    def __init__(self, train_device: torch.device = None, temp_device: torch.device = None):
        self._ds = None
        self._dl = None
        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def get_data_set(self) -> MGDS | DistributedMGDS:
        pass

    @abstractmethod
    def get_data_loader(self) -> TrainDataLoader | DistributedTrainDataLoader:
        pass

    def _create_mgds(
            self,
            config,
            modules,
            train_progress=None,
            is_validation=False,
    ) -> MGDS | DistributedMGDS:
        """Create MGDS instance with distributed support"""
        # Filter out None modules
        modules = [m for m in modules if m is not None]

        # Flatten list of lists into single list
        flattened_modules = []
        for module in modules:
            if isinstance(module, list):
                flattened_modules.extend(module)
            else:
                flattened_modules.append(module)

        if config.enable_fsdp and torch.distributed.is_initialized():
            # Use existing process group initialized by GenericTrainer
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Create distributed MGDS
            ds = DistributedMGDS(
                device=torch.device(config.train_device),
                concepts=config.concepts,
                settings={},
                definition=flattened_modules,
                batch_size=config.batch_size,
                state=train_progress.state if train_progress else None,
                world_size=world_size,
                rank=rank,
                seed=-1,
                initial_epoch=0,
                initial_epoch_sample=0,
            )

            # Create distributed data loader
            dl = DistributedTrainDataLoader(
                dataset=ds,
                batch_size=config.batch_size,
                num_workers=config.dataloader_threads,
                pin_memory=True,
            )
        else:
            # Create regular MGDS for single GPU
            ds = MGDS(
                device=torch.device(config.train_device),
                concepts=config.concepts,
                settings={},
                definition=flattened_modules,
                batch_size=config.batch_size,
                state=train_progress.state if train_progress else None,
                seed=-1,
                initial_epoch=0,
                initial_epoch_sample=0,
            )

            # Create regular data loader
            dl = TrainDataLoader(ds, config.batch_size)

        self._ds = ds
        self._dl = dl
        return ds
