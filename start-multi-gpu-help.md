# OneTrainer Multi-GPU Training Guide

This guide explains how to use OneTrainer's multi-GPU training features to accelerate model training across multiple GPUs.

## Requirements

- Multiple CUDA-compatible GPUs (at least 2)
- PyTorch with CUDA support
- NCCL or GLOO backend installed (NCCL is recommended for GPU training)

## Methods to Enable Multi-GPU Training

### Method 1: Using the GUI

1. Launch the OneTrainer UI: `./start-ui.sh` or `start-ui.bat`
2. In the UI, find the "Multi-GPU Training" panel
3. Check the "Enable Multi-GPU" option
4. Configure additional options:
   - Backend: Choose between "nccl" (recommended) or "gloo"
   - Distributed Data: Enable for better performance
   - Use Torchrun: Recommended for most users
   - Scale Learning Rate: Auto-scale learning rate based on GPU count
5. Configure your model and training settings as usual
6. Click "Start Training"

### Method 2: Using the Command Line Script

The `start-multi-gpu.sh` (Linux/macOS) or `start-multi-gpu.bat` (Windows) script provides an interactive way to launch distributed training:

```bash
./start-multi-gpu.sh
```

The script will:
1. Detect available GPUs
2. Ask how many GPUs to use
3. Ask for the path to your training config file
4. Launch distributed training

### Method 3: Running Directly with torchrun (Advanced)

For advanced users, you can run the multi-GPU training script directly with torchrun:

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/train_multi_gpu.py --config-path=YOUR_CONFIG.json
```

## Troubleshooting

If you encounter issues:

1. Run the diagnostic script: `python scripts/check_multi_gpu.py`
2. Verify your GPUs are detected: `nvidia-smi`
3. Make sure your CUDA environment is properly set up
4. Try using the GLOO backend if NCCL has issues on your system

## Performance Tips

1. Enable distributed data loading for optimal performance
2. Use the "Scale Learning Rate" option to automatically adjust learning rates
3. The first epoch may be slower as the distributed environment initializes
4. Monitor GPU utilization with `nvidia-smi` to ensure all GPUs are being used