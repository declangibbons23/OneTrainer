import contextlib
import copy
import json
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.memory_util import TorchMemoryRecorder
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        # Initialize process group for FSDP if in distributed mode
        if config.enable_fsdp and os.environ.get('RANK') is not None:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                torch.cuda.set_device(torch.distributed.get_rank())

        # Only primary GPU (or non-distributed mode) should create tensorboard
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

    def start(self):
        # Only primary GPU (or non-distributed mode) should save config
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.__save_config_to_workspace()

        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token = self.config.secrets.huggingface_token,
                    new_session = False,
                )

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        # Set up FSDP first if enabled and in distributed mode
        if self.config.enable_fsdp and os.environ.get('RANK') is not None:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                torch.cuda.set_device(torch.distributed.get_rank())
            self.model_setup.setup_fsdp(self.model, self.config)

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        # Create data loader with distributed support
        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )

        # Scale batch size by world size in distributed mode
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            self.config.batch_size = self.config.batch_size // world_size

        self.model_saver = self.create_model_saver()
        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model.train_progress, is_validation=True
            )

    def train(self):
        """Train the model"""
        try:
            while not self.should_stop():
                self.train_step()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.end()

    def end(self):
        try:
            if self.one_step_trained and self.model is not None:
                self.model.to(self.temp_device)

                if self.config.backup_before_save:
                    # Only primary GPU should create backup
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        self.backup(self.model.train_progress)

                # Clean up FSDP if enabled
                if self.config.enable_fsdp:
                    self.model_setup.cleanup_fsdp()

                # Special case for schedule-free optimizers.
                if self.config.optimizer.optimizer.is_schedule_free:
                    torch.clear_autocast_cache()
                    self.model.optimizer.eval()

                # Only primary GPU (or non-distributed mode) should save the model
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    self.callbacks.on_update_status("saving the final model")

                    if hasattr(self.model, 'ema') and self.model.ema is not None:
                        self.model.ema.copy_ema_to(self.parameters, store_temp=False)

                    if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                        save_path = os.path.join(
                            self.config.output_model_destination,
                            f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                        )
                    else:
                        save_path = self.config.output_model_destination
                    print("Saving " + save_path)

                    # Handle FSDP state dict type
                    if self.config.enable_fsdp:
                        with FSDP.state_dict_type(self.model, StateDictType[self.config.fsdp_state_dict_type]):
                            self.model_saver.save(
                                model=self.model,
                                model_type=self.config.model_type,
                                output_model_format=self.config.output_model_format,
                                output_model_destination=save_path,
                                dtype=self.config.output_dtype.torch_dtype()
                            )
                    else:
                        self.model_saver.save(
                            model=self.model,
                            model_type=self.config.model_type,
                            output_model_format=self.config.output_model_format,
                            output_model_destination=save_path,
                            dtype=self.config.output_dtype.torch_dtype()
                        )

            elif self.model is not None:
                self.model.to(self.temp_device)
                
                # Clean up FSDP if enabled
                if self.config.enable_fsdp:
                    self.model_setup.cleanup_fsdp()

            # Close tensorboard only on primary GPU (or non-distributed mode)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if hasattr(self, 'tensorboard'):
                    self.tensorboard.close()

                if self.config.tensorboard:
                    super()._stop_tensorboard()

            for handle in self.grad_hook_handles:
                handle.remove()
        except Exception as e:
            print(f"Error during end: {e}")
            traceback.print_exc()

    def backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        # Skip backup on non-primary GPUs
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        torch_gc()
        self.callbacks.on_update_status("creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            # Handle FSDP state dict type
            if self.config.enable_fsdp:
                with FSDP.state_dict_type(self.model, StateDictType[self.config.fsdp_state_dict_type]):
                    self.model_saver.save(
                        self.model,
                        self.config.model_type,
                        ModelFormat.INTERNAL,
                        backup_path,
                        None,
                    )
            else:
                self.model_saver.save(
                    self.model,
                    self.config.model_type,
                    ModelFormat.INTERNAL,
                    backup_path,
                    None,
                )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        # Skip save on non-primary GPUs
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        torch_gc()
        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if hasattr(self.model, 'ema') and self.model.ema is not None:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
                
            # Handle FSDP state dict type
            if self.config.enable_fsdp:
                with FSDP.state_dict_type(self.model, StateDictType[self.config.fsdp_state_dict_type]):
                    self.model_saver.save(
                        model=self.model,
                        model_type=self.config.model_type,
                        output_model_format=self.config.output_model_format,
                        output_model_destination=save_path,
                        dtype=self.config.output_dtype.torch_dtype()
                    )
            else:
                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )
                
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if hasattr(self.model, 'ema') and self.model.ema is not None:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def should_stop(self) -> bool:
        """Check if training should stop"""
        if self.commands.stop_training:
            return True

        if self.model.train_progress.global_step >= self.config.epochs * len(self.data_loader.get_data_loader()):
            return True

        return False

    def __needs_sample(self, train_progress: TrainProgress) -> bool:
        """Check if sampling is needed"""
        return self.repeating_action_needed(
            "sample",
            self.config.sample_after,
            self.config.sample_after_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_save(self, train_progress: TrainProgress) -> bool:
        """Check if saving is needed"""
        return self.repeating_action_needed(
            "save",
            self.config.save_every,
            self.config.save_every_unit,
            train_progress,
            start_at_zero=False,
        )

    def train_step(self):
        """Execute one training step"""
        self.model_setup.setup_train_device(self.model, self.config)
        self.model.train()

        # Get next batch
        batch = next(iter(self.data_loader.get_data_loader()))

        # Forward pass
        model_output_data = self.model_setup.predict(
            self.model, batch, self.config, self.model.train_progress)

        # Calculate loss
        loss = self.model_setup.calculate_loss(
            self.model, batch, model_output_data, self.config)

        # Backward pass
        self.model_setup.backward_pass(self.model, loss, self.config)

        # Update parameters
        self.model_setup.optimizer_step(self.model, self.config)

        # Update progress
        self.model.train_progress.global_step += 1
        self.model.train_progress.epoch_step += 1

        # Log metrics (only on primary GPU or non-distributed mode)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.tensorboard.add_scalar(
                "loss/train_step",
                loss.item(),
                self.model.train_progress.global_step
            )

        # Sample if needed
        if self.__needs_sample(self.model.train_progress):
            self.sample_queue.append(lambda: self.model_sampler.sample(
                self.model,
                self.config.samples,
                self.config,
                self.model.train_progress,
                self.config.non_ema_sampling,
            ))

        # Save if needed
        if self.__needs_save(self.model.train_progress):
            self.save(self.model.train_progress)

        # Validate if needed
        self.__validate(self.model.train_progress)

        # Set one_step_trained flag
        self.one_step_trained = True

        # Process sample queue
        if len(self.sample_queue) > 0:
            try:
                self.sample_queue[0]()
            except Exception as e:
                print(f"Error during sampling: {e}")
                traceback.print_exc()
            self.sample_queue.pop(0)

    def __needs_gc(self, train_progress: TrainProgress) -> bool:
        """Check if garbage collection is needed"""
        return self.repeating_action_needed(
            "gc",
            self.config.gc_interval,
            self.config.gc_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_validate(self, train_progress: TrainProgress) -> bool:
        """Check if validation is needed"""
        return self.config.validation and self.repeating_action_needed(
            "validation",
            self.config.validation_interval,
            self.config.validation_unit,
            train_progress,
            start_at_zero=False,
        )

    def __save_config_to_workspace(self):
        """Save training configuration to workspace"""
        config_path = os.path.join(self.config.workspace_dir, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)

    def __save_backup_config(self, backup_path: str):
        """Save configuration for backup"""
        config_path = os.path.join(backup_path, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)

    def __prune_backups(self, keep_count: int):
        """Remove old backups, keeping only the specified number"""
        backup_dir = os.path.join(self.config.workspace_dir, "backup")
        if not os.path.exists(backup_dir):
            return

        backups = []
        for entry in os.scandir(backup_dir):
            if entry.is_dir() and entry.name.endswith("-backup"):
                backups.append(entry)

        backups.sort(key=lambda x: os.path.getctime(x.path), reverse=True)

        for backup in backups[keep_count:]:
            try:
                shutil.rmtree(backup.path)
            except Exception:
                print(f"Could not delete backup {backup.path}")

    def __clear_cache(self):
        """Clear cache directory"""
        if hasattr(self.config, "cache_dir") and self.config.cache_dir:
            cache_dir = self.config.cache_dir
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception:
                    print(f"Could not clear cache directory {cache_dir}")

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            # Only log validation metrics on primary GPU or non-distributed mode
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                for concept_seed, total_loss in accumulated_loss_per_concept.items():
                    average_loss = total_loss / concept_counts[concept_seed]
                    self.tensorboard.add_scalar(
                        f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                        average_loss,
                        train_progress.global_step
                    )

                if len(concept_counts) > 1:
                    total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                    total_count = sum(concept_counts[key] for key in concept_counts)
                    total_average_loss = total_loss / total_count
                    self.tensorboard.add_scalar(
                        "loss/validation_step/total_average",
                        total_average_loss,
                        train_progress.global_step
                    )
