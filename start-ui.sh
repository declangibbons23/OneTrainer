#!/usr/bin/env bash

set -e

# Function to check and fix display settings
check_display() {
    # If DISPLAY is not set and we're in an XRDP session
    if [ -z "$DISPLAY" ] && [ -n "$XRDP_SESSION" ]; then
        # Try to find an available display
        for display in {10..0}; do
            if [ -e "/tmp/.X11-unix/X$display" ]; then
                export DISPLAY=":$display"
                echo "[OneTrainer] Setting DISPLAY to $DISPLAY"
                break
            fi
        done
    fi

    # If still no display, try default
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=":0"
        echo "[OneTrainer] Setting default DISPLAY to :0"
    fi

    # Test display connection
    if ! xset q &>/dev/null; then
        echo "[OneTrainer] WARNING: Could not connect to X display."
        echo "[OneTrainer] If you're using XRDP, make sure you're logged in to your desktop session."
        echo "[OneTrainer] Alternatively, use ./start-distributed-headless.sh for GUI-less training."
        exit 1
    fi
}

# Source library functions
source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Check display before proceeding
check_display

# Prepare environment and run
prepare_runtime_environment
run_python_in_active_env "scripts/train_ui.py" "$@"
