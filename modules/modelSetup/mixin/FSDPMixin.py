from typing import List, Optional

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


class FSDPMixin:
    """Mixin class to add FSDP support to model setup classes"""

    def setup_fsdp(
        self,
        model: BaseModel,
        config: TrainConfig,
        modules_to_wrap: Optional[List[torch.nn.Module]] = None
    ):
        """Set up FSDP for the model if enabled in config"""
        if not config.enable_fsdp:
            return

        # Initialize process group
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(torch.distributed.get_rank())

        # Configure mixed precision
        mp_policy = None
        if config.train_dtype != config.weight_dtype:
            mp_policy = MixedPrecision(
                param_dtype=config.weight_dtype.torch_dtype(),
                reduce_dtype=config.train_dtype.torch_dtype(),
                buffer_dtype=config.train_dtype.torch_dtype(),
            )

        # Configure CPU offloading
        cpu_offload = CPUOffload(offload_params=config.fsdp_offload_params)

        # Configure backward prefetch
        if config.fsdp_backward_prefetch == "BACKWARD_PRE":
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        elif config.fsdp_backward_prefetch == "BACKWARD_POST":
            backward_prefetch = BackwardPrefetch.BACKWARD_POST
        else:
            backward_prefetch = None

        # Configure sharding strategy
        if config.fsdp_sharding_strategy == "FULL_SHARD":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD

        # Configure auto wrap policy
        if modules_to_wrap:
            auto_wrap_policy = transformer_auto_wrap_policy(tuple(type(m) for m in modules_to_wrap))
        else:
            auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=config.fsdp_min_num_params)

        # Wrap model with FSDP
        for module in model.get_trainable_modules():
            if not isinstance(module, FSDP):
                wrapped_module = FSDP(
                    module,
                    mixed_precision=mp_policy,
                    sharding_strategy=sharding_strategy,
                    cpu_offload=cpu_offload,
                    backward_prefetch=backward_prefetch,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=torch.cuda.current_device(),
                )
                # Replace original module with wrapped version
                for name, m in model.named_modules():
                    if m is module:
                        setattr(model, name, wrapped_module)

        # Set state dict type
        model.fsdp_state_dict_type = StateDictType[config.fsdp_state_dict_type]

    def cleanup_fsdp(self):
        """Clean up FSDP process group"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
