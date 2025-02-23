# OneTrainer

A unified trainer for various AI models.

## Multi-GPU Training

OneTrainer now supports distributed training across multiple GPUs using PyTorch's Fully Sharded Data Parallel (FSDP) library. This feature enables efficient training of large models by sharding model parameters, gradients, and optimizer states across multiple GPUs.

### Requirements

- At least 2 NVIDIA GPUs with CUDA support
- PyTorch with CUDA support
- NCCL backend for distributed training

### Using Multi-GPU Training

There are three ways to enable multi-GPU training:

1. **Using the GUI**:
   - In the Training tab, enable "FSDP" under the base settings
   - Configure FSDP settings:
     - Sharding Strategy: Choose how to shard model parameters (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
     - Backward Prefetch: Configure parameter prefetching (BACKWARD_PRE, BACKWARD_POST, NO_PREFETCH)
     - State Dict Type: Choose state dict handling (FULL_STATE_DICT, SHARDED_STATE_DICT, LOCAL_STATE_DICT)
     - Offload Params: Enable CPU offloading to save GPU memory
     - Min Num Params: Minimum number of parameters for a layer to be wrapped in FSDP
     - Number of GPUs: Number of GPUs to use for training

2. **Using the Command Line (with GUI)**:
   - Windows: Run `start-distributed.bat` to automatically launch distributed training
   - Linux with GUI: Run `./start-distributed.sh` to automatically launch distributed training
   - Both scripts will:
     - Check for CUDA availability
     - Verify multiple GPUs are available
     - Launch training processes across all available GPUs
     - Handle proper process initialization and cleanup

   Note for Linux users: Make sure to make the script executable:
   ```bash
   chmod +x start-distributed.sh
   ```

3. **Headless Training (for Servers without GUI)**:
   - Run `./start-distributed-headless.sh` to launch distributed training without GUI
   - Make the script executable first:
   ```bash
   chmod +x start-distributed-headless.sh
   ```
   
   The headless mode:
   - Runs without requiring X11 or display server
   - Automatically configures FSDP settings
   - Uses all available GPUs by default
   - Supports the same distributed features as GUI mode

### Important Notes

1. **Data Handling**:
   - Training data is automatically sharded across GPUs
   - Each GPU processes its own portion of the data
   - Latent caching is GPU-specific and managed automatically

2. **Memory Usage**:
   - Model parameters are sharded across GPUs to reduce memory usage
   - Gradients and optimizer states are also sharded
   - CPU offloading can be enabled to further reduce GPU memory usage

3. **Saving and Loading**:
   - Model checkpoints are automatically handled
   - Only the primary GPU saves checkpoints and logs
   - Different state dict types are supported for flexibility

4. **Performance Tips**:
   - Adjust batch size per GPU (total batch size is divided by number of GPUs)
   - Use gradient checkpointing with FSDP for larger models
   - Enable CPU offloading if needed, but note it may impact training speed

### Troubleshooting

1. **Memory Issues**:
   - Try reducing batch size
   - Enable gradient checkpointing
   - Enable CPU offloading
   - Use a more aggressive sharding strategy

2. **Performance Issues**:
   - Check GPU utilization
   - Adjust batch size and gradient accumulation steps
   - Try different backward prefetch strategies

3. **Common Errors**:
   - "CUDA out of memory": Reduce batch size or enable more aggressive memory optimizations
   - "NCCL error": Check GPU connectivity and NCCL installation
   - "Process group initialization failed": Check if all GPUs are available and CUDA is working

For more details on specific settings and advanced configurations, please refer to the PyTorch FSDP documentation.

### Manual Multi-GPU Launch (Advanced)

For advanced users who want to manually control the distributed training process:

1. **Linux**:
   ```bash
   # Launch with torch.distributed.launch
   python -m torch.distributed.launch --nproc_per_node=NUM_GPUS scripts/launch_distributed.py [args...]
   
   # Or using torchrun (recommended)
   torchrun --nproc_per_node=NUM_GPUS scripts/launch_distributed.py [args...]
   ```

2. **Windows**:
   ```cmd
   # Using torchrun
   torchrun --nproc_per_node=NUM_GPUS scripts/launch_distributed.py [args...]
   ```

Replace `NUM_GPUS` with the number of GPUs you want to use. For headless environments, use `scripts/train_headless.py` instead of `launch_distributed.py`.

The training script will automatically:
- Initialize the distributed process group
- Configure FSDP parameters
- Shard data and model across GPUs
- Handle distributed caching
- Manage checkpoints and logging

### Environment Variables

You can customize the distributed training behavior using these environment variables:

- `MASTER_ADDR`: Address of the master node (default: localhost)
- `MASTER_PORT`: Port for distributed training (default: 12355)
- `WORLD_SIZE`: Total number of processes (set automatically)
- `LOCAL_RANK`: Local rank of the process (set automatically)
- `RANK`: Global rank of the process (set automatically)

These variables are handled automatically by the launch scripts but can be manually set if needed.

### Server Deployment Tips

When running on headless servers:

1. **Display Issues**:
   - If you see "couldn't connect to display" errors:
     * For XRDP users: Make sure you're logged into your desktop session
     * The script will attempt to auto-detect your XRDP display
     * If auto-detection fails, you can manually set: `export DISPLAY=:N` (where N is your display number)
     * If display issues persist, use headless mode with `./start-distributed-headless.sh`
   - The GUI mode requires X11 server:
     * Works with local X11 server
     * Works with XRDP + XFCE/other desktop environments
     * Works with X11 forwarding over SSH (`ssh -X` or `ssh -Y`)
   - Headless mode works without any display requirements

2. **Resource Management**:
   - Monitor GPU memory usage with `nvidia-smi`
   - Use `htop` or `top` to monitor CPU and memory usage
   - Consider using `tmux` or `screen` to keep training running after SSH disconnection

3. **Performance Optimization**:
   - Set appropriate batch sizes based on available GPU memory
   - Monitor network bandwidth between GPUs with `nvidia-smi nvlink`
   - Use `CUDA_VISIBLE_DEVICES` to control which GPUs are used
