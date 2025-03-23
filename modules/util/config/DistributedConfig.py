from modules.util.config.BaseConfig import BaseConfig
from modules.util.distributed import DistributedBackend, DataDistributionStrategy
from typing import Any

class DistributedConfig(BaseConfig):
    enabled: bool
    backend: DistributedBackend
    data_loading_strategy: DataDistributionStrategy
    latent_caching_strategy: DataDistributionStrategy
    find_unused_parameters: bool
    gradient_as_bucket_view: bool
    bucket_cap_mb: int
    use_torchrun: bool
    detect_nvlink: bool
    master_addr: str
    master_port: int
    timeout_seconds: int

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        data.append(("backend", DistributedBackend.NCCL, DistributedBackend, False))
        data.append(("data_loading_strategy", DataDistributionStrategy.DISTRIBUTED, DataDistributionStrategy, False))
        data.append(("latent_caching_strategy", DataDistributionStrategy.DISTRIBUTED, DataDistributionStrategy, False))
        data.append(("find_unused_parameters", False, bool, False))
        data.append(("gradient_as_bucket_view", True, bool, False))
        data.append(("bucket_cap_mb", 25, int, False))
        data.append(("use_torchrun", True, bool, False))
        data.append(("detect_nvlink", True, bool, False))
        data.append(("master_addr", "localhost", str, False))
        data.append(("master_port", 12355, int, False))
        data.append(("timeout_seconds", 1800, int, False))

        return DistributedConfig(data)
