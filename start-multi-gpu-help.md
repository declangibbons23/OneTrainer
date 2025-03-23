# Multi-GPU Training with OneTrainer

This document provides information about using multiple GPUs for training in OneTrainer.

## Requirements

- At least 2 CUDA-capable GPUs
- PyTorch with CUDA support
- NCCL or GLOO backend (NCCL recommended for better performance)

## Usage

### Using the GUI

1. Launch the OneTrainer UI
2. In the Training tab, you'll see a "Multi-GPU Training" section
3. Check "Enable Multi-GPU" to use distributed training
4. Configure other settings:
   - Backend: NCCL (recommended) or GLOO
   - Distributed Data: Enable for better performance
   - Use Torchrun: Recommended method for launching processes
   - Scale Learning Rate: Automatically adjust LR based on number of GPUs

### Using the Command Line

Run the `start-multi-gpu.sh` script (Linux/macOS) or `start-multi-gpu.bat` (Windows):

```bash
# Linux/macOS
./start-multi-gpu.sh

# Windows
start-multi-gpu.bat
```

Follow the prompts to configure your multi-GPU training.

## How Multi-GPU Training Works

OneTrainer uses PyTorch's Distributed Data Parallel (DDP) for multi-GPU training:

1. The model is replicated across multiple GPUs
2. Each GPU processes a different batch of data
3. Gradients are synchronized across all GPUs
4. The model parameters stay in sync during training

Benefits:
- Linear speedup with the number of GPUs (ideally)
- Ability to train larger models or use larger batch sizes
- Reduced training time

## Troubleshooting

If you're having issues with multi-GPU training:

1. Run the diagnostic script to check your setup:
   ```bash
   python scripts/check_multi_gpu.py
   ```

2. Common issues:
   - CUDA not available: Install CUDA and PyTorch with CUDA support
   - Only one GPU detected: Check if your GPUs are visible to the system
   - NCCL errors: Try using the GLOO backend instead
   - Out of memory errors: Reduce batch size per GPU

## Advanced Configuration

For advanced users, you can directly use the `train_multi_gpu.py` script:

```bash
python scripts/train_multi_gpu.py --config-path=your_config.json --num-gpus=2 --distributed-backend=nccl
```

Options:
- `--config-path`: Path to your training config JSON file
- `--num-gpus`: Number of GPUs to use (0 for all)
- `--spawn`: Use torch.multiprocessing.spawn instead of torchrun
- `--port`: Port for distributed communication
- `--distributed-backend`: "nccl" or "gloo"
- `--distributed-data-loading`: Enable distributed data loading