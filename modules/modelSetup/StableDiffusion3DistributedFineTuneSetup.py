"""
Model setup for distributed fine-tuning of Stable Diffusion 3 models.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSetup.mixin.ModelSetupDistributedMixin import ModelSetupDistributedMixin
from modules.modelSetup.StableDiffusion3FineTuneSetup import StableDiffusion3FineTuneSetup
from modules.util import distributed
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusion3DistributedFineTuneSetup(StableDiffusion3FineTuneSetup, ModelSetupDistributedMixin):
    """
    Model setup for distributed fine-tuning of Stable Diffusion 3 models.
    
    This class extends the base SD3 fine-tune setup with distributed training
    capabilities provided by the ModelSetupDistributedMixin.
    """
    
    def setup_train_device(self, model: BaseModel, config: TrainConfig):
        """
        Set up the model for training on the appropriate device.
        
        For distributed training, this moves different model components
        to the correct device based on the local rank.
        
        Args:
            model: Model to set up
            config: Training configuration
        """
        if not self.is_distributed(config):
            # Fall back to standard device setup if not distributed
            return super().setup_train_device(model, config)
        
        # For distributed training, use local rank as device
        local_rank = distributed.get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
        
        # Move model components to the appropriate device
        sd_model: StableDiffusion3Model = model
        
        # Move components to train device
        sd_model.vae.to(device)
        sd_model.clip_encoder.to(device)
        sd_model.unet.to(device)
        
        if sd_model.noise_scheduler is not None:
            sd_model.noise_scheduler.to(device)
            
        if sd_model.vae_decoder_fix is not None:
            sd_model.vae_decoder_fix.to(device)
            
        return device
    
    def setup_model(self, model: BaseModel, config: TrainConfig):
        """
        Set up the model for distributed training.
        
        This method wraps appropriate model components with DDP.
        
        Args:
            model: Model to set up
            config: Training configuration
        """
        # First do the standard setup
        super().setup_model(model, config)
        
        # Then do distributed-specific setup
        if self.is_distributed(config):
            self.setup_model_for_distributed(model, config)
            self.setup_optimizer_for_distributed(model, config)
    
    def setup_model_for_distributed(self, model: BaseModel, config: TrainConfig) -> BaseModel:
        """
        Set up a model for distributed training.
        
        This wraps the UNet and potentially other trainable components with DDP.
        
        Args:
            model: Model to set up
            config: Training configuration
            
        Returns:
            The model set up for distributed training
        """
        if not self.is_distributed(config):
            return model
        
        sd_model: StableDiffusion3Model = model
        local_rank = distributed.get_local_rank()
        
        # Wrap UNet with DDP (the main trainable component)
        if sd_model.trainable_parameters.unet:
            sd_model.unet = self.wrap_model_for_distributed(sd_model.unet, local_rank, config)
            print(f"[Rank {distributed.get_rank()}] UNet wrapped with DDP")
            
        # Wrap text encoder with DDP if it's being trained
        if sd_model.trainable_parameters.clip_encoder:
            sd_model.clip_encoder = self.wrap_model_for_distributed(sd_model.clip_encoder, local_rank, config)
            print(f"[Rank {distributed.get_rank()}] Text encoder wrapped with DDP")
            
        # Other components that might be trainable
        if sd_model.trainable_parameters.vae:
            sd_model.vae = self.wrap_model_for_distributed(sd_model.vae, local_rank, config)
            print(f"[Rank {distributed.get_rank()}] VAE wrapped with DDP")
        
        # Synchronize model parameters across processes to ensure
        # all processes start with identical model weights
        self.synchronize_model(model)
        
        return model
    
    def predict(
        self,
        model: BaseModel,
        batch,
        config: TrainConfig,
        train_progress,
        deterministic: bool = False,
    ):
        """
        Run prediction on a batch.
        
        For distributed training, ensure proper device placement.
        
        Args:
            model: Model to predict with
            batch: Batch to predict on
            config: Training configuration
            train_progress: Current training progress
            deterministic: Whether to use deterministic settings
            
        Returns:
            Model outputs
        """
        # For distributed, we need to ensure everything is on the right device
        if self.is_distributed(config):
            local_rank = distributed.get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
            
            # Move any additional tensors to the correct device if needed
            # This is usually handled by the data loader, but just in case
            if not batch["latents"].device == device:
                batch["latents"] = batch["latents"].to(device)
                
            if not batch["prompt_embeds"].device == device:
                batch["prompt_embeds"] = batch["prompt_embeds"].to(device)
                
            if "negative_prompt_embeds" in batch and batch["negative_prompt_embeds"] is not None:
                if not batch["negative_prompt_embeds"].device == device:
                    batch["negative_prompt_embeds"] = batch["negative_prompt_embeds"].to(device)
        
        # The standard predict function should work as-is now that everything is on the right device
        return super().predict(model, batch, config, train_progress, deterministic)
    
    def calculate_loss(
        self,
        model: BaseModel,
        batch,
        model_output_data,
        config: TrainConfig,
    ):
        """
        Calculate loss from model outputs.
        
        For distributed training, we want to make sure the loss is properly
        normalized across devices.
        
        Args:
            model: Model
            batch: Input batch
            model_output_data: Model outputs
            config: Training configuration
            
        Returns:
            Loss tensor
        """
        # Calculate the standard loss
        loss = super().calculate_loss(model, batch, model_output_data, config)
        
        # For distributed training, we use the standard PyTorch DDP behavior,
        # which automatically handles the loss reduction across processes.
        # DDP averages the gradients, not the losses, so we don't need to 
        # explicitly modify the loss calculation.
        
        return loss
