@echo off
setlocal enabledelayedexpansion

echo Checking for CUDA support...

:: Check if Python and PyTorch are available
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo CUDA is not available. Multi-GPU training requires CUDA support.
    echo Please install CUDA and PyTorch with CUDA support, or use single-GPU training.
    exit /b 1
)

:: Get number of available GPUs
for /f "tokens=*" %%a in ('python -c "import torch; print(torch.cuda.device_count())"') do set NUM_GPUS=%%a
echo CUDA available: Yes
echo Number of GPUs: %NUM_GPUS%

if %NUM_GPUS% LSS 2 (
    echo Only %NUM_GPUS% GPU detected. Multi-GPU training requires at least 2 GPUs.
    echo Falling back to single GPU training.
    start-ui.bat
    exit /b 0
)

echo Found %NUM_GPUS% GPUs available for training.
echo.

:: Check if user wants to use all GPUs
set /p USE_ALL_GPUS="Use all %NUM_GPUS% GPUs? (y/n): "
if /i "%USE_ALL_GPUS%" == "n" (
    set /p NUM_GPUS_TO_USE="Enter number of GPUs to use: "
) else (
    set NUM_GPUS_TO_USE=%NUM_GPUS%
)

:: Check if config file is provided
set /p CONFIG_PATH="Enter path to training config file: "

if "%CONFIG_PATH%" == "" (
    echo Config file path is required.
    exit /b 1
)

:: Check if the user wants to use torchrun or torch.multiprocessing
set /p USE_TORCHRUN="Use torchrun for launching? Recommended. (y/n): "

if /i "%USE_TORCHRUN%" == "n" (
    echo Launching with torch.multiprocessing.spawn
    python scripts/train_multi_gpu.py "--config-path=%CONFIG_PATH%" "--num-gpus=%NUM_GPUS_TO_USE%" "--spawn"
) else (
    echo Launching with torchrun
    python scripts/train_multi_gpu.py "--config-path=%CONFIG_PATH%" "--num-gpus=%NUM_GPUS_TO_USE%"
)

pause