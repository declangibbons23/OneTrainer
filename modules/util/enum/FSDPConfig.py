from enum import Enum, auto


class FSDPShardingStrategy(Enum):
    FULL_SHARD = auto()  # Shard parameters, gradients, and optimizer states
    SHARD_GRAD_OP = auto()  # Shard gradients and optimizer states
    NO_SHARD = auto()  # Don't shard anything, just wrap in FSDP for mixed precision and other features

    def __str__(self):
        return self.name


class FSDPBackwardPrefetch(Enum):
    BACKWARD_PRE = auto()  # Prefetch next set of parameters before backward pass
    BACKWARD_POST = auto()  # Prefetch next set of parameters after backward pass
    NO_PREFETCH = auto()  # Don't prefetch parameters

    def __str__(self):
        return self.name


class FSDPStateDict(Enum):
    FULL_STATE_DICT = auto()  # Save/load full state dict (unsharded)
    SHARDED_STATE_DICT = auto()  # Save/load sharded state dict
    LOCAL_STATE_DICT = auto()  # Save/load local state dict (for debugging)

    def __str__(self):
        return self.name
