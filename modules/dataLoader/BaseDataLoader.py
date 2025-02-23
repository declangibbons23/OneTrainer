from abc import ABCMeta, abstractmethod

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.DistributedMGDS import DistributedMGDS, DistributedTrainDataLoader

import torch


class BaseDataLoader(metaclass=ABCMeta):
    """Base class for all data loaders"""

    def __init__(self):
        self._ds = None
        self._dl = None

    @abstractmethod
    def get_data_set(self) -> MGDS | DistributedMGDS:
        pass

    @abstractmethod
    def get_data_loader(self) -> TrainDataLoader | DistributedTrainDataLoader:
        pass

    def _create_mgds(
            self,
            config,
            concepts,
            settings,
            definition,
            state,
            seed=-1,
            initial_epoch=0,
            initial_epoch_sample=0,
    ) -> MGDS | DistributedMGDS:
        """Create MGDS instance with distributed support"""
        if config.enable_fsdp:
            # Initialize process group if not already initialized
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                torch.cuda.set_device(torch.distributed.get_rank())

            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

            # Create distributed MGDS
            ds = DistributedMGDS(
                device=torch.device(config.train_device),
                concepts=concepts,
                settings=settings,
                definition=definition,
                batch_size=config.batch_size,
                state=state,
                world_size=world_size,
                rank=rank,
                seed=seed,
                initial_epoch=initial_epoch,
                initial_epoch_sample=initial_epoch_sample,
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
                concepts=concepts,
                settings=settings,
                definition=definition,
                batch_size=config.batch_size,
                state=state,
                seed=seed,
                initial_epoch=initial_epoch,
                initial_epoch_sample=initial_epoch_sample,
            )

            # Create regular data loader
            dl = TrainDataLoader(ds, config.batch_size)

        self._ds = ds
        self._dl = dl
        return ds
