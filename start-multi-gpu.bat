@echo off
setlocal

REM Check if Python is available
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python and try again.
    exit /b 1
)

REM Check if PyTorch is installed
python -c "import torch" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo PyTorch not found. Please install PyTorch and try again.
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; cuda_available = torch.cuda.is_available(); print('CUDA available:', cuda_available); print('Number of GPUs:', torch.cuda.device_count() if cuda_available else 0); exit(0 if cuda_available else 1)"
IF %ERRORLEVEL% NEQ 0 (
    echo CUDA is not available. Multi-GPU training requires CUDA support.
    echo Please install CUDA and PyTorch with CUDA support, or use single-GPU training.
    exit /b 1
)
echo.

REM Get number of available GPUs
for /f "tokens=3" %%a in ('python -c "import torch; print(torch.cuda.device_count())"') do (
    set NUM_GPUS=%%a
)

if %NUM_GPUS% LSS 2 (
    echo Only %NUM_GPUS% GPU detected. Multi-GPU training requires at least 2 GPUs.
    echo Falling back to single GPU training.
    start "" start-ui.bat
    exit /b
)

echo Found %NUM_GPUS% GPUs available for training.
echo.

REM Check if user wants to use all GPUs
set /p USE_ALL_GPUS="Use all %NUM_GPUS% GPUs? (y/n): "
if /i "%USE_ALL_GPUS%"=="n" (
    set /p NUM_GPUS_TO_USE="Enter number of GPUs to use: "
) else (
    set NUM_GPUS_TO_USE=%NUM_GPUS%
)

REM Check if config file is provided
set /p CONFIG_PATH="Enter path to training config file: "

if "%CONFIG_PATH%"=="" (
    echo Config file path is required.
    exit /b 1
)

REM Check if the user wants to use torchrun or torch.multiprocessing
set /p USE_TORCHRUN="Use torchrun for launching? Recommended. (y/n): "

if /i "%USE_TORCHRUN%"=="n" (
    echo Launching with torch.multiprocessing.spawn
    python -m scripts.train_multi_gpu --config-path "%CONFIG_PATH%" --num-gpus %NUM_GPUS_TO_USE% --spawn
) else (
    echo Launching with torchrun
    python -m scripts.train_multi_gpu --config-path "%CONFIG_PATH%" --num-gpus %NUM_GPUS_TO_USE%
)

endlocal