from abc import ABCMeta, abstractmethod
from typing import Iterator, List
from uuid import uuid4

from modules.module.EMAModule import EMAModuleWrapper
from torch.distributed.fsdp import StateDictType
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress

import torch
from torch.optim import Optimizer


class BaseModelEmbedding:
    def __init__(
            self,
            uuid: str,
            token_count: int,
            placeholder: str,
    ):
        self.uuid = uuid
        self.placeholder = placeholder
        self.text_tokens = [f"<{uuid4()}>" for _ in range(token_count)]


class BaseModel(metaclass=ABCMeta):
    model_type: ModelType
    parameters: NamedParameterGroupCollection | None
    optimizer: Optimizer | None
    optimizer_state_dict: dict | None
    param_group_mapping: list[str] | None
    ema: EMAModuleWrapper
    ema_state_dict: dict | None
    train_progress: TrainProgress
    model_spec: ModelSpec | None
    train_config: TrainConfig | None
    fsdp_state_dict_type: StateDictType | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        self.model_type = model_type
        self._modules = {}  # Required for named_modules support
        self.parameters = None
        self.optimizer = None
        self.optimizer_state_dict = None
        self.param_group_mapping = None
        self.ema_state_dict = None
        self.train_progress = TrainProgress()
        self.model_spec = None
        self.train_config = None
        self.fsdp_state_dict_type = None

    def get_trainable_modules(self) -> List[torch.nn.Module]:
        """Get list of trainable modules for FSDP wrapping"""
        modules = []
        
        # Add UNet if it exists and is trainable
        if hasattr(self, 'unet') and self.unet is not None:
            modules.append(self.unet)
            
        # Add text encoders if they exist and are trainable
        if hasattr(self, 'text_encoder_1') and self.text_encoder_1 is not None:
            modules.append(self.text_encoder_1)
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            modules.append(self.text_encoder_2)
        if hasattr(self, 'text_encoder_3') and self.text_encoder_3 is not None:
            modules.append(self.text_encoder_3)
            
        # Add prior if it exists and is trainable
        if hasattr(self, 'prior') and self.prior is not None:
            modules.append(self.prior)
            
        return modules

    def named_modules(self, *args, **kwargs) -> Iterator[tuple[str, torch.nn.Module]]:
        """Get named modules for FSDP support"""
        for name, module in self._modules.items():
            yield name, module
            for subname, submodule in module.named_modules(*args, **kwargs):
                yield f"{name}.{subname}", submodule

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def eval(self):
        pass

    @staticmethod
    def _add_embeddings_to_prompt(
            additional_embeddings: list[BaseModelEmbedding],
            embedding: BaseModelEmbedding | None,
            prompt: str,
    ) -> str:
        for embedding in additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        if embedding is not None:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        return prompt
