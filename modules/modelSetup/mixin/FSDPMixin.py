from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.FSDPConfig import FSDPShardingStrategy, FSDPBackwardPrefetch, FSDPStateDict


class FSDPMixin:
    """Mixin class for FSDP setup in model setup classes"""

    def setup_fsdp(self, model: torch.nn.Module, config: TrainConfig) -> torch.nn.Module:
        """Set up FSDP for the model"""
        if not config.enable_fsdp:
            return model

        # Initialize process group if not already done
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(torch.distributed.get_rank())

        # Configure mixed precision
        mixed_precision_policy = self._get_mixed_precision_policy(config)

        # Configure sharding strategy
        sharding_strategy = self._get_sharding_strategy(config)

        # Configure backward prefetch
        backward_prefetch = self._get_backward_prefetch(config)

        # Configure CPU offload
        cpu_offload = CPUOffload(offload_params=config.fsdp_offload_params)

        # Configure auto wrap policy
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer},
            min_num_params=config.fsdp_min_num_params
        )

        # Wrap model with FSDP
        model = FSDP(
            model,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
        )

        return model

    def _get_mixed_precision_policy(self, config: TrainConfig) -> Optional[MixedPrecision]:
        """Get mixed precision policy based on config"""
        if config.train_dtype == DataType.FLOAT_32:
            return None

        param_dtype = torch.float32
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32

        if config.train_dtype == DataType.FLOAT_16:
            param_dtype = torch.float16
            reduce_dtype = torch.float16
            buffer_dtype = torch.float16
        elif config.train_dtype == DataType.BFLOAT_16:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.bfloat16
            buffer_dtype = torch.bfloat16

        return MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    def _get_sharding_strategy(self, config: TrainConfig) -> ShardingStrategy:
        """Get sharding strategy based on config"""
        if config.fsdp_sharding_strategy == FSDPShardingStrategy.FULL_SHARD:
            return ShardingStrategy.FULL_SHARD
        elif config.fsdp_sharding_strategy == FSDPShardingStrategy.SHARD_GRAD_OP:
            return ShardingStrategy.SHARD_GRAD_OP
        else:
            return ShardingStrategy.NO_SHARD

    def _get_backward_prefetch(self, config: TrainConfig) -> Optional[BackwardPrefetch]:
        """Get backward prefetch based on config"""
        if config.fsdp_backward_prefetch == FSDPBackwardPrefetch.BACKWARD_PRE:
            return BackwardPrefetch.BACKWARD_PRE
        elif config.fsdp_backward_prefetch == FSDPBackwardPrefetch.BACKWARD_POST:
            return BackwardPrefetch.BACKWARD_POST
        else:
            return None

    def save_fsdp_model(self, model: torch.nn.Module, save_path: str, config: TrainConfig):
        """Save FSDP model based on config"""
        if not config.enable_fsdp:
            return

        with FSDP.state_dict_type(
            model,
            StateDictType=self._get_state_dict_type(config),
            state_dict_config=None
        ):
            state_dict = model.state_dict()
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, save_path)

    def _get_state_dict_type(self, config: TrainConfig) -> Any:
        """Get state dict type based on config"""
        if config.fsdp_state_dict_type == FSDPStateDict.FULL_STATE_DICT:
            return FSDP.STATE_DICT_TYPE.FULL_STATE_DICT
        elif config.fsdp_state_dict_type == FSDPStateDict.SHARDED_STATE_DICT:
            return FSDP.STATE_DICT_TYPE.SHARDED_STATE_DICT
        else:
            return FSDP.STATE_DICT_TYPE.LOCAL_STATE_DICT

    def load_fsdp_model(self, model: torch.nn.Module, load_path: str, config: TrainConfig):
        """Load FSDP model based on config"""
        if not config.enable_fsdp:
            return

        with FSDP.state_dict_type(
            model,
            StateDictType=self._get_state_dict_type(config),
            state_dict_config=None
        ):
            state_dict = torch.load(load_path)
            model.load_state_dict(state_dict)

    def cleanup_fsdp(self):
        """Clean up FSDP resources"""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Ensure all processes are ready to clean up
            torch.distributed.destroy_process_group()
