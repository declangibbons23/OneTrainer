import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class TrainArgs(BaseArgs):
    config_path: str
    secrets_path: str
    multi_gpu: bool
    distributed_backend: str
    single_gpu_data_loading: bool
    no_torchrun: bool
    local_rank: int
    
    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> 'TrainArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        # @formatter:off

        parser.add_argument("--config-path", type=str, required=True, dest="config_path", help="The path to the config file")
        parser.add_argument("--secrets-path", type=str, required=False, dest="secrets_path", help="The path to the secrets file")
        parser.add_argument("--callback-path", type=str, required=False, dest="callback_path", help="The path to the callback pickle file")
        parser.add_argument("--command-path", type=str, required=False, dest="command_path", help="The path to the command pickle file")
        
        # Multi-GPU arguments
        parser.add_argument("--multi-gpu", action="store_true", dest="multi_gpu", help="Enable multi-GPU training")
        parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", dest="distributed_backend", help="Distributed backend")
        parser.add_argument("--single-gpu-data-loading", action="store_true", dest="single_gpu_data_loading", help="Use single-GPU for data loading only")
        parser.add_argument("--no-torchrun", action="store_true", dest="no_torchrun", help="Use mp.spawn instead of torchrun")
        parser.add_argument("--local-rank", type=int, default=0, dest="local_rank", help="Local rank for distributed training (usually set by torchrun)")

        # @formatter:on

        args = TrainArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'TrainArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("config_path", None, str, True))
        data.append(("secrets_path", None, str, True))
        data.append(("callback_path", None, str, True))
        data.append(("command_path", None, str, True))
        
        # Multi-GPU options
        data.append(("multi_gpu", False, bool, False))
        data.append(("distributed_backend", "nccl", str, False))
        data.append(("single_gpu_data_loading", False, bool, False))
        data.append(("no_torchrun", False, bool, False))
        data.append(("local_rank", 0, int, False))

        return TrainArgs(data)
