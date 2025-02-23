@echo off
setlocal enabledelayedexpansion

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not found in PATH
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())" | findstr "True" >nul
if %ERRORLEVEL% neq 0 (
    echo CUDA is not available
    exit /b 1
)

REM Get number of available GPUs
for /f %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do set GPU_COUNT=%%i
if %GPU_COUNT% lss 2 (
    echo Found only %GPU_COUNT% GPU^(s^). Multi-GPU training requires at least 2 GPUs.
    exit /b 1
)

echo Starting distributed training with %GPU_COUNT% GPUs...

REM Launch distributed training
python scripts/launch_distributed.py %*

if %ERRORLEVEL% neq 0 (
    echo Error during distributed training
    exit /b 1
)

exit /b 0
