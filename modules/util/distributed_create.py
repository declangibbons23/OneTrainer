"""
Factory functions for creating distributed training components.

This module provides factory functions for creating distributed-aware
components such as data loaders and model setups.
"""

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.StableDiffusion3DistributedDataLoader import StableDiffusion3DistributedDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.StableDiffusion3DistributedFineTuneSetup import StableDiffusion3DistributedFineTuneSetup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def create_distributed_data_loader(
    model_type: ModelType,
    model: BaseModel,
    train_progress: TrainProgress,
    config: TrainConfig,
    is_validation: bool = False,
) -> BaseDataLoader:
    """
    Create a distributed data loader appropriate for the given model type.
    
    Args:
        model_type: Type of model
        model: Model to load data for
        train_progress: Current training progress
        config: Training configuration
        is_validation: Whether this is a validation data loader
        
    Returns:
        A distributed data loader
        
    Raises:
        ValueError: If no distributed data loader exists for the given model type
    """
    if model_type == ModelType.STABLE_DIFFUSION_3:
        return StableDiffusion3DistributedDataLoader(model, train_progress, config, is_validation)
    
    # Add other model types here as they're implemented
    # elif model_type == ModelType.PIXART_ALPHA:
    #     return PixArtAlphaDistributedDataLoader(model, train_progress, config, is_validation)
    # elif model_type == ModelType.STABLE_DIFFUSION_XL:
    #     return StableDiffusionXLDistributedDataLoader(model, train_progress, config, is_validation)
    
    # If no distributed data loader exists, raise an error
    raise ValueError(f"No distributed data loader exists for model type {model_type}")


def create_distributed_model_setup(
    model_type: ModelType,
    train_device: str,
    temp_device: str,
    training_method: TrainingMethod,
    debug_mode: bool = False,
) -> BaseModelSetup:
    """
    Create a distributed model setup appropriate for the given model type and training method.
    
    Args:
        model_type: Type of model
        train_device: Device to train on
        temp_device: Device for temporary storage
        training_method: Training method (fine-tune, LoRA, etc.)
        debug_mode: Whether to enable debug mode
        
    Returns:
        A distributed model setup
        
    Raises:
        ValueError: If no distributed model setup exists for the given model type and training method
    """
    # Stable Diffusion 3
    if model_type == ModelType.STABLE_DIFFUSION_3:
        if training_method == TrainingMethod.FINE_TUNE:
            return StableDiffusion3DistributedFineTuneSetup(train_device, temp_device, debug_mode)
        # Add more training methods as they're implemented
        # elif training_method == TrainingMethod.LORA:
        #     return StableDiffusion3DistributedLoRASetup(train_device, temp_device, debug_mode)
    
    # Add other model types here as they're implemented
    # elif model_type == ModelType.PIXART_ALPHA:
    #     if training_method == TrainingMethod.FINE_TUNE:
    #         return PixArtAlphaDistributedFineTuneSetup(train_device, temp_device, debug_mode)
    
    # If no distributed model setup exists, raise an error
    raise ValueError(f"No distributed model setup exists for model type {model_type} and training method {training_method}")
