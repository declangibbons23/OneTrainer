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

:: Create a temporary Python script to avoid command-line parsing issues
set TEMP_SCRIPT=%TEMP%\multi_gpu_launcher_%RANDOM%.py
echo import sys > %TEMP_SCRIPT%
echo import os >> %TEMP_SCRIPT%
echo import subprocess >> %TEMP_SCRIPT%
echo sys.path.append(".") >> %TEMP_SCRIPT%

if /i "%USE_TORCHRUN%" == "n" (
    echo Launching with torch.multiprocessing.spawn
    echo from scripts.train_multi_gpu import main >> %TEMP_SCRIPT%
    echo import argparse >> %TEMP_SCRIPT%
    echo. >> %TEMP_SCRIPT%
    echo parser = argparse.ArgumentParser() >> %TEMP_SCRIPT%
    echo parser.add_argument("--config-path", type=str, default=r"%CONFIG_PATH%") >> %TEMP_SCRIPT%
    echo parser.add_argument("--num-gpus", type=int, default=%NUM_GPUS_TO_USE%) >> %TEMP_SCRIPT%
    echo parser.add_argument("--spawn", action="store_true", default=True) >> %TEMP_SCRIPT%
    echo args = parser.parse_args() >> %TEMP_SCRIPT%
    echo. >> %TEMP_SCRIPT%
    echo main(args) >> %TEMP_SCRIPT%
) else (
    echo Launching with torchrun
    echo cmd = [ >> %TEMP_SCRIPT%
    echo     sys.executable, >> %TEMP_SCRIPT%
    echo     "-m", "torch.distributed.run", >> %TEMP_SCRIPT%
    echo     f"--nproc_per_node=%NUM_GPUS_TO_USE%", >> %TEMP_SCRIPT%
    echo     f"--master_port=12355", >> %TEMP_SCRIPT%
    echo     "scripts/train_multi_gpu.py", >> %TEMP_SCRIPT%
    echo     f"--config-path=%CONFIG_PATH%", >> %TEMP_SCRIPT%
    echo     f"--num-gpus=%NUM_GPUS_TO_USE%", >> %TEMP_SCRIPT%
    echo ] >> %TEMP_SCRIPT%
    echo. >> %TEMP_SCRIPT%
    echo print(f"Executing: {' '.join(cmd)}") >> %TEMP_SCRIPT%
    echo process = subprocess.run(cmd) >> %TEMP_SCRIPT%
    echo sys.exit(process.returncode) >> %TEMP_SCRIPT%
)

:: Run the temporary script
python %TEMP_SCRIPT%

:: Clean up
del %TEMP_SCRIPT%

echo.
echo Multi-GPU training completed.
pause