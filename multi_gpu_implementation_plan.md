# Multi-GPU Training Implementation Plan for OneTrainer

This document outlines the plan for implementing multi-GPU training support in OneTrainer using PyTorch's DistributedDataParallel (DDP).

## 1. Configuration Changes

Update the `TrainConfig` class to support multi-GPU training parameters:

```python
# Add to TrainConfig in modules/util/config/TrainConfig.py
enable_multi_gpu: bool = False  # Toggle for multi-GPU training
distributed_backend: str = "nccl"  # 'nccl' for NVIDIA GPUs, 'gloo' for CPU or fallback
world_size: int = None  # Number of GPUs to use (auto-detect if None)
use_torchrun: bool = True  # Use torchrun (recommended) over mp.spawn
distributed_data_loading: bool = True  # Use distributed data loading/caching
initial_port: str = "12355"  # Communication port for distributed setup
node_rank: int = 0  # For multi-node: rank of this node
num_nodes: int = 1  # For multi-node: total number of nodes
master_addr: str = "localhost"  # For multi-node: address of master node
```

## 2. Implement a Distributed Trainer Wrapper

Create a new `DistributedTrainer` class that extends `GenericTrainer`:

```python
# New file: modules/trainer/DistributedTrainer.py
class DistributedTrainer(GenericTrainer):
    def __init__(self, config, callbacks, commands, local_rank):
        self.local_rank = local_rank
        config.train_device = torch.device(f"cuda:{local_rank}")
        super().__init__(config, callbacks, commands)
        self.is_main_process = local_rank == 0
```

## 3. Process Group Initialization

Create functions to handle distributed process initialization:

```python
def setup_distributed(config):
    if config.enable_multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but multi-GPU training is enabled")
            
        if config.use_torchrun:
            # Environment variables are set by torchrun
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
        else:
            # For mp.spawn approach, we'll set these manually
            os.environ["MASTER_ADDR"] = config.master_addr
            os.environ["MASTER_PORT"] = str(config.initial_port)
            local_rank = config.local_rank
            world_size = config.world_size or torch.cuda.device_count()
            rank = config.node_rank * torch.cuda.device_count() + local_rank
            
        # Set device for this process
        torch.cuda.set_device(local_rank)
        
        # Check for NVLINK connectivity and warn if not found
        if config.distributed_backend == "nccl":
            try:
                p2p_status = torch.cuda.get_device_properties(local_rank).p2p_status
                has_nvlink = any(p2p_status)
                if not has_nvlink and local_rank == 0:
                    print("NVLINK not detected between GPUs. Using PCIe for NCCL communication.")
            except Exception:
                pass  # Silently continue if we can't check
                
        # Initialize the process group
        init_process_group(
            backend=config.distributed_backend,
            world_size=world_size,
            rank=rank
        )
        
        return local_rank, rank, world_size
    return 0, 0, 1  # Default for single GPU
```

## 4. Modify Data Loader to Support Distributed Training

Update the data loader to optionally use `DistributedSampler`:

```python
# Add to BaseDataLoader.py
def set_sampler(self, sampler):
    # Should be implemented by all data loaders that need DDP support
    pass

# Modify in GenericTrainer.py
def create_data_loader(self, model, train_progress, is_validation=False):
    data_loader = super().create_data_loader(model, train_progress, is_validation)
    
    if self.config.enable_multi_gpu and self.config.distributed_data_loading and dist.is_initialized():
        # Replace existing sampler with DistributedSampler
        dataset = data_loader.get_data_set()
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )
        data_loader.set_sampler(sampler)
    
    return data_loader
```

## 5. Modify Cache Directories for Distributed Data Loading

Create separate cache directories for each process when using distributed data loading:

```python
# In create_dataset method or before creating data loaders
if config.enable_multi_gpu and config.distributed_data_loading and dist.is_initialized():
    # Create process-specific cache directory
    rank = dist.get_rank()
    original_cache_dir = config.cache_dir
    config.cache_dir = os.path.join(original_cache_dir, f"rank-{rank}")
    os.makedirs(config.cache_dir, exist_ok=True)
```

## 6. Modify Model Setup to Wrap Models with DDP

Update the model setup to support DDP:

```python
# Add to BaseModelSetup implementation
def setup_train_device(self, model: BaseModel, config: TrainConfig):
    # Regular setup code
    model.to(self.train_device)
    
    # Wrap with DDP if multi-GPU is enabled
    if config.enable_multi_gpu and dist.is_initialized():
        # Only wrap the trainable models, not embeddings or the entire BaseModel
        # This will be model-specific for each implementation
        if hasattr(model, 'model') and model.model is not None:
            model.model = DDP(
                model.model, 
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=False
            )
```

## 7. Update Training Script for Distributed Launching

Modify the main train.py script to support distributed training:

```python
def run_distributed_training(rank, world_size, config, callbacks, commands):
    # Setup distributed environment
    config.local_rank = rank
    setup_distributed(config)
    
    # Create and run trainer
    trainer = DistributedTrainer(config, callbacks, commands, rank)
    trainer.start()
    trainer.train()
    trainer.end()
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

# In main function
if config.enable_multi_gpu:
    if config.use_torchrun:
        # When using torchrun, we don't need mp.spawn
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        run_distributed_training(local_rank, None, config, callbacks, commands)
    else:
        # Using torch.multiprocessing
        world_size = config.world_size or torch.cuda.device_count()
        mp.spawn(
            run_distributed_training,
            args=(world_size, config, callbacks, commands),
            nprocs=world_size,
            join=True
        )
else:
    # Regular single-GPU training
    trainer = GenericTrainer(config, callbacks, commands)
    trainer.start()
    trainer.train()
    trainer.end()
```

## 8. Modify Training Functions for Coordination

Update necessary training functions to coordinate across processes:

```python
# Modify in DistributedTrainer.py
def backup(self, train_progress, print_msg=True, print_cb=print):
    # Only the main process should create backups
    if not self.config.enable_multi_gpu or dist.get_rank() == 0:
        super().backup(train_progress, print_msg, print_cb)
    
    # Synchronize processes
    if self.config.enable_multi_gpu and dist.is_initialized():
        dist.barrier()

def save(self, train_progress, print_msg=True, print_cb=print):
    # Only the main process should save models
    if not self.config.enable_multi_gpu or dist.get_rank() == 0:
        super().save(train_progress, print_msg, print_cb)
    
    # Synchronize processes
    if self.config.enable_multi_gpu and dist.is_initialized():
        dist.barrier()
        
# For sampling and validation
def __sample_during_training(self, train_progress, train_device, sample_params_list=None):
    if not self.config.enable_multi_gpu or dist.get_rank() == 0:
        super().__sample_during_training(train_progress, train_device, sample_params_list)
        
    if self.config.enable_multi_gpu and dist.is_initialized():
        dist.barrier()
```

## 9. UI Integration

Update the Training tab UI to include multi-GPU options:

```python
# Add to __create_base_frame in modules/ui/TrainingTab.py
def __create_distributed_frame(self, master, row):
    frame = ctk.CTkFrame(master=master, corner_radius=5)
    frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
    frame.grid_columnconfigure(0, weight=1)

    # Enable multi-GPU
    components.label(frame, 0, 0, "Enable Multi-GPU Training",
                     tooltip="Enable distributed data parallel training across multiple GPUs")
    components.switch(frame, 0, 1, self.ui_state, "enable_multi_gpu")

    # Distributed data loading
    components.label(frame, 1, 0, "Distributed Data Loading",
                     tooltip="Enable distributed data loading and caching (disable to use only single-GPU for data loading)")
    components.switch(frame, 1, 1, self.ui_state, "distributed_data_loading")

    # Distributed backend
    components.label(frame, 2, 0, "Distributed Backend",
                     tooltip="Backend for distributed communication (NCCL recommended for NVIDIA GPUs)")
    components.options(frame, 2, 1, ["nccl", "gloo"], self.ui_state, "distributed_backend")

    # Use torchrun
    components.label(frame, 3, 0, "Use Torchrun",
                     tooltip="Use torchrun launcher (recommended) instead of multiprocessing")
    components.switch(frame, 3, 1, self.ui_state, "use_torchrun")

    # Multi-node setup (advanced)
    components.label(frame, 4, 0, "Multi-node Settings", tooltip="Settings for multi-node training")
    components.button(frame, 4, 1, "Configure", command=self.__open_multi_node_settings_window)
```

## 10. Learning Rate Scaling

Add automatic learning rate scaling based on the effective batch size:

```python
# Add to BaseModelSetup.py
def scale_learning_rate_for_ddp(self, config):
    if config.enable_multi_gpu and dist.is_initialized():
        world_size = dist.get_world_size()
        
        # Scale learning rate based on world size
        if hasattr(config, 'learning_rate'):
            config.original_learning_rate = config.learning_rate
            
            # Apply square root scaling rule by default
            config.learning_rate *= math.sqrt(world_size)
            
            if dist.get_rank() == 0:
                print(f"Scaling learning rate by sqrt({world_size}): {config.original_learning_rate} → {config.learning_rate}")
```

## 11. Command-line Interface

Update the CLI to support both torchrun and manual distributed setup:

```python
# In train.py
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
parser.add_argument('--single_gpu_data_loading', action='store_true', help='Use single-GPU for data loading only')
parser.add_argument('--backend', choices=['nccl', 'gloo'], default='nccl', help='Distributed backend')
parser.add_argument('--no_torchrun', action='store_true', help='Use mp.spawn instead of torchrun')
```

## 12. Torchrun Launch Script

Create a dedicated launch script for torchrun (train_distributed.sh):

```bash
#!/bin/bash
# Usage: ./train_distributed.sh [num_gpus] [config_file]

NUM_GPUS=${1:-"all"}  # Use all GPUs by default
CONFIG_FILE=$2

if [ "$NUM_GPUS" = "all" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

echo "Launching distributed training with $NUM_GPUS GPUs"
torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/train.py --config $CONFIG_FILE --multi_gpu
```

## 13. Multi-Node Support

For multi-node training, provide additional configuration in a separate window:

```python
class MultiNodeSettingsWindow:
    def __init__(self, master, ui_state):
        self.top = ctk.CTkToplevel(master)
        self.top.title("Multi-Node Settings")
        self.top.geometry("500x300")
        
        # Number of nodes
        components.label(self.top, 0, 0, "Number of Nodes")
        components.entry(self.top, 0, 1, ui_state, "num_nodes")
        
        # Node rank
        components.label(self.top, 1, 0, "This Node's Rank (0-based)")
        components.entry(self.top, 1, 1, ui_state, "node_rank")
        
        # Master address
        components.label(self.top, 2, 0, "Master Node Address")
        components.entry(self.top, 2, 1, ui_state, "master_addr")
        
        # Initial port
        components.label(self.top, 3, 0, "Communication Port")
        components.entry(self.top, 3, 1, ui_state, "initial_port")
        
        # Generate launch command
        components.button(self.top, 4, 0, "Generate Launch Command", command=self.generate_command)
        self.command_output = ctk.CTkTextbox(self.top, width=480, height=100)
        self.command_output.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
        
    def generate_command(self):
        # Generate torchrun command for the user
        command = "torchrun "
        command += f"--nnodes={ui_state.get_var('num_nodes').get()} "
        command += f"--node_rank={ui_state.get_var('node_rank').get()} "
        command += f"--master_addr={ui_state.get_var('master_addr').get()} "
        command += f"--master_port={ui_state.get_var('initial_port').get()} "
        command += f"--nproc_per_node=<GPUS_PER_NODE> "
        command += "scripts/train.py --config <CONFIG_FILE> --multi_gpu"
        
        self.command_output.delete("0.0", "end")
        self.command_output.insert("0.0", command)
```

## Implementation Phases

### Phase 1 - Basic DDP Integration (Single Node)
1. Update TrainConfig with multi-GPU parameters
2. Create setup_distributed function
3. Modify BaseModelSetup to support DDP wrapping
4. Implement basic DistributedTrainer
5. Modify data loader to use DistributedSampler (with option to disable)
6. Update train.py to support distributed launch

### Phase 2 - UI Integration and Optimization
1. Add multi-GPU options to the Training UI
2. Add distributed data loading option
3. Implement learning rate scaling
4. Add NVLink detection and optimization
5. Create torchrun launch script

### Phase 3 - Multi-Node Support
1. Implement multi-node configuration
2. Add multi-node settings window
3. Update documentation and examples
4. Testing across multiple configurations