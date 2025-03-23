import json
import tkinter as tk
from tkinter import ttk

# Try to import dependencies, but provide fallbacks if not available
try:
    from modules.ui.TooltipLabel import TooltipLabel
except ImportError:
    # Create a simple fallback if TooltipLabel is not available
    class TooltipLabel(ttk.Label):
        def __init__(self, master=None, text="", tooltip="", **kwargs):
            super().__init__(master, text=text, **kwargs)

try:
    from modules.util.ui.UIState import UIState
except ImportError:
    # Create minimal UIState implementation if unavailable
    class UIState:
        def __init__(self):
            self.train_config = None
        
        def get_var(self, name):
            return None


class MultiGPUFrame(ttk.LabelFrame):
    def __init__(self, master=None, ui_state: UIState = None):
        super().__init__(master, text="Multi-GPU Training")
        self.ui_state = ui_state

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)

        # Variables
        self.enable_multi_gpu = tk.BooleanVar(value=False)
        self.backend = tk.StringVar(value="nccl")
        self.distributed_data_loading = tk.BooleanVar(value=True)
        self.use_torchrun = tk.BooleanVar(value=True)
        self.lr_scaling = tk.BooleanVar(value=True)

        # Device count
        gpu_count = 0
        try:
            import torch
            gpu_count = torch.cuda.device_count()
        except:
            pass
        
        # Create widgets
        row = 0

        # GPU count info label
        if gpu_count > 0:
            gpu_info_text = f"Detected {gpu_count} GPU{'' if gpu_count == 1 else 's'}"
            gpu_info_label = ttk.Label(self, text=gpu_info_text)
            gpu_info_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
            row += 1

            if gpu_count > 1:
                # Enable multi-GPU training checkbox
                enable_label = TooltipLabel(
                    self, 
                    text="Enable Multi-GPU:",
                    tooltip="Enable distributed training across multiple GPUs"
                )
                enable_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                enable_checkbox = ttk.Checkbutton(
                    self, 
                    variable=self.enable_multi_gpu,
                    command=self._update_ui_state
                )
                enable_checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
                
                # Backend dropdown
                backend_label = TooltipLabel(
                    self, 
                    text="Backend:",
                    tooltip="Distributed backend to use (nccl is recommended for GPUs)"
                )
                backend_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                backend_dropdown = ttk.Combobox(
                    self, 
                    textvariable=self.backend,
                    values=["nccl", "gloo"],
                    state="readonly",
                    width=10
                )
                backend_dropdown.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
                
                # Distributed data loading checkbox
                data_loading_label = TooltipLabel(
                    self, 
                    text="Distributed Data:",
                    tooltip="Enable distributed data loading for better performance"
                )
                data_loading_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                data_loading_checkbox = ttk.Checkbutton(
                    self, 
                    variable=self.distributed_data_loading,
                    command=self._update_ui_state
                )
                data_loading_checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
                
                # Use torchrun checkbox
                torchrun_label = TooltipLabel(
                    self, 
                    text="Use Torchrun:",
                    tooltip="Use torchrun launcher (recommended) instead of multiprocessing"
                )
                torchrun_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                torchrun_checkbox = ttk.Checkbutton(
                    self, 
                    variable=self.use_torchrun,
                    command=self._update_ui_state
                )
                torchrun_checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
                
                # Learning rate scaling checkbox
                lr_scaling_label = TooltipLabel(
                    self, 
                    text="Scale Learning Rate:",
                    tooltip="Automatically scale learning rate based on number of GPUs"
                )
                lr_scaling_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                lr_scaling_checkbox = ttk.Checkbutton(
                    self, 
                    variable=self.lr_scaling,
                    command=self._update_ui_state
                )
                lr_scaling_checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
            else:
                # Not enough GPUs warning
                not_enough_label = ttk.Label(
                    self, 
                    text="Multi-GPU training requires at least 2 GPUs",
                    foreground="red"
                )
                not_enough_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
                row += 1
        else:
            # No GPUs detected warning
            no_gpu_label = ttk.Label(
                self, 
                text="No CUDA-capable GPUs detected",
                foreground="red"
            )
            no_gpu_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
            row += 1
    def _update_ui_state(self):
        """Update the UI state with current multi-GPU settings"""
        if not self.ui_state or not hasattr(self.ui_state, 'train_config'):
            return
            
        try:
            # Get current values
            enable_multi_gpu = self.enable_multi_gpu.get()
            backend = self.backend.get()
            distributed_data = self.distributed_data_loading.get()
            use_torchrun = self.use_torchrun.get()
            lr_scaling = self.lr_scaling.get()
            
            # Use the new setup method
            self.ui_state.train_config.setup_multi_gpu(
                enable=enable_multi_gpu,
                backend=backend
            )
            
            # Update additional settings
            self.ui_state.train_config.distributed_data_loading = distributed_data
            self.ui_state.train_config.use_torchrun = use_torchrun
            self.ui_state.train_config.lr_scaling = lr_scaling
            
            print(f"Updated multi-GPU settings: enabled={enable_multi_gpu}, backend={backend}")
            
        except ValueError as e:
            # Handle expected errors (like not enough GPUs)
            print(f"Multi-GPU configuration error: {e}")
            # Reset UI state
            self.enable_multi_gpu.set(False)
            self._show_error(str(e))
        except Exception as e:
            # Handle unexpected errors
            print(f"Error updating multi-GPU settings: {e}")
            import traceback
            traceback.print_exc()
            self._show_error("Unexpected error configuring multi-GPU settings")
            
    def _show_error(self, message: str):
        """Show error message in UI"""
        try:
            from tkinter import messagebox
            
            # Show error dialog
            messagebox.showerror(
                "Multi-GPU Configuration Error",
                message
            )
            
            # Also print to console for logging
            print(f"Multi-GPU Error: {message}")
        except Exception as e:
            # Fallback to console if messagebox fails
            print(f"Multi-GPU Error: {message}")
            print(f"Failed to show error dialog: {e}")

    def update_from_config(self, config):
        """Update UI elements from config"""
        try:
            # Set defaults first
            self.enable_multi_gpu.set(False)
            self.backend.set("nccl")
            self.distributed_data_loading.set(True)
            self.use_torchrun.set(True)
            self.lr_scaling.set(True)
            
            # Update from config if attributes exist
            if hasattr(config, 'enable_multi_gpu'):
                self.enable_multi_gpu.set(config.enable_multi_gpu)
            
            if hasattr(config, 'distributed_backend'):
                self.backend.set(config.distributed_backend)
            
            if hasattr(config, 'distributed_data_loading'):
                self.distributed_data_loading.set(config.distributed_data_loading)
            
            if hasattr(config, 'use_torchrun'):
                self.use_torchrun.set(config.use_torchrun)
                
            if hasattr(config, 'lr_scaling'):
                self.lr_scaling.set(config.lr_scaling)
                
            # Update UI state to ensure all settings are synchronized
            self._update_ui_state()
            
        except Exception as e:
            print(f"Error updating from config: {e}")
            import traceback
            traceback.print_exc()
            self._show_error("Failed to load multi-GPU settings from config")
            
        # Update UI state to ensure all settings are synchronized
        self._update_ui_state()