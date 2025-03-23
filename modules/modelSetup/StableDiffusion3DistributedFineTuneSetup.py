from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSetup.StableDiffusion3FineTuneSetup import StableDiffusion3FineTuneSetup
from modules.modelSetup.mixin.ModelSetupDistributedMixin import ModelSetupDistributedMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress
from modules.util.distributed import is_distributed_available, is_main_process

import torch


class StableDiffusion3DistributedFineTuneSetup(
    StableDiffusion3FineTuneSetup,
    ModelSetupDistributedMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )
        self.ddp_modules = {}
        
    def create_parameters(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        if not self.is_distributed_training_enabled(config):
            # Use parent implementation if distributed training is not enabled
            return super().create_parameters(model, config)
        
        parameter_group_collection = NamedParameterGroupCollection()

        # For DDP-wrapped modules, we need to use .module.parameters()
        if config.text_encoder.train:
            text_encoder_1 = self.ddp_modules.get('text_encoder_1', model.text_encoder_1)
            parameters = text_encoder_1.module.parameters() if hasattr(text_encoder_1, 'module') else text_encoder_1.parameters()
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_1",
                parameters=parameters,
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.text_encoder_2.train:
            text_encoder_2 = self.ddp_modules.get('text_encoder_2', model.text_encoder_2)
            parameters = text_encoder_2.module.parameters() if hasattr(text_encoder_2, 'module') else text_encoder_2.parameters()
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_2",
                parameters=parameters,
                learning_rate=config.text_encoder_2.learning_rate,
            ))

        if config.text_encoder_3.train:
            text_encoder_3 = self.ddp_modules.get('text_encoder_3', model.text_encoder_3)
            parameters = text_encoder_3.module.parameters() if hasattr(text_encoder_3, 'module') else text_encoder_3.parameters()
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_3",
                parameters=parameters,
                learning_rate=config.text_encoder_3.learning_rate,
            ))

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder_1 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_1_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding and model.text_encoder_2 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_2_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

            if config.text_encoder_3.train_embedding and model.text_encoder_3 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_3_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_3"
                )

        if config.prior.train:
            transformer = self.ddp_modules.get('transformer', model.transformer)
            parameters = transformer.module.parameters() if hasattr(transformer, 'module') else transformer.parameters()
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer",
                parameters=parameters,
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def setup_for_distributed_training(
        self,
        model: StableDiffusion3Model,
        config: TrainConfig
    ):
        """
        Set up StableDiffusion3 model components for distributed training.
        """
        if not self.is_distributed_training_enabled(config):
            return
            
        ddp_params = self.get_ddp_params(config)
        
        # Wrap model components that are being trained with DDP
        if config.text_encoder.train and model.text_encoder_1 is not None:
            model.text_encoder_1 = self.apply_ddp_to_model_component(
                model.text_encoder_1, **ddp_params
            )
            self.ddp_modules['text_encoder_1'] = model.text_encoder_1
            
        if config.text_encoder_2.train and model.text_encoder_2 is not None:
            model.text_encoder_2 = self.apply_ddp_to_model_component(
                model.text_encoder_2, **ddp_params
            )
            self.ddp_modules['text_encoder_2'] = model.text_encoder_2
            
        if config.text_encoder_3.train and model.text_encoder_3 is not None:
            model.text_encoder_3 = self.apply_ddp_to_model_component(
                model.text_encoder_3, **ddp_params
            )
            self.ddp_modules['text_encoder_3'] = model.text_encoder_3
            
        if config.prior.train and model.transformer is not None:
            model.transformer = self.apply_ddp_to_model_component(
                model.transformer, **ddp_params
            )
            self.ddp_modules['transformer'] = model.transformer
    
    def unwrap_ddp_modules(self, model: StableDiffusion3Model):
        """
        Unwrap DDP modules for saving or evaluation.
        """
        if not self.ddp_modules:
            return
            
        # Unwrap all DDP-wrapped modules
        if 'text_encoder_1' in self.ddp_modules and hasattr(model.text_encoder_1, 'module'):
            model.text_encoder_1 = model.text_encoder_1.module
            
        if 'text_encoder_2' in self.ddp_modules and hasattr(model.text_encoder_2, 'module'):
            model.text_encoder_2 = model.text_encoder_2.module
            
        if 'text_encoder_3' in self.ddp_modules and hasattr(model.text_encoder_3, 'module'):
            model.text_encoder_3 = model.text_encoder_3.module
            
        if 'transformer' in self.ddp_modules and hasattr(model.transformer, 'module'):
            model.transformer = model.transformer.module
            
        self.ddp_modules = {}
    
    def setup_model(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ):
        # First do the standard setup
        super().setup_model(model, config)
        
        # Then apply distributed training setup if enabled
        self.setup_for_distributed_training(model, config)
