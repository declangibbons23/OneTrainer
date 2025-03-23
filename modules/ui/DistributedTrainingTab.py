"""
UI Tab for distributed training settings.
"""

import os
import torch
from typing import Dict, List, Tuple, Optional

from PySide6 import QtCore, QtWidgets

from modules.util import distributed
from modules.util.config.DistributedConfig import Backend, DataDistributionStrategy
from modules.util.ui import components


class DistributedTrainingTab(QtWidgets.QWidget):
    """
    Tab for distributed training settings in the GUI.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Initialize with detected GPU count
        self.refresh_gpu_info()
    
    def setup_ui(self):
        """Set up the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # GPU info section
        gpu_group = QtWidgets.QGroupBox("GPU Information")
        gpu_layout = QtWidgets.QVBoxLayout(gpu_group)
        
        self.gpu_info_label = QtWidgets.QLabel("Detecting GPUs...")
        gpu_layout.addWidget(self.gpu_info_label)
        
        self.refresh_button = QtWidgets.QPushButton("Refresh GPU Info")
        self.refresh_button.clicked.connect(self.refresh_gpu_info)
        gpu_layout.addWidget(self.refresh_button)
        
        layout.addWidget(gpu_group)
        
        # Enable distributed training
        self.enable_distributed = components.create_checkbox(
            "Enable distributed training", False,
            "Use multiple GPUs for training with PyTorch DistributedDataParallel"
        )
        layout.addWidget(self.enable_distributed)
        
        # Distributed settings container (only visible when enabled)
        self.distributed_settings = QtWidgets.QWidget()
        distributed_layout = QtWidgets.QVBoxLayout(self.distributed_settings)
        distributed_layout.setContentsMargins(0, 0, 0, 0)
        
        # Connection settings
        connection_group = QtWidgets.QGroupBox("Connection Settings")
        connection_layout = QtWidgets.QFormLayout(connection_group)
        
        # Backend selection
        self.backend = components.create_combo_box(
            ["nccl", "gloo"], 
            "nccl",
            "Backend for distributed communication (nccl recommended for GPU)"
        )
        connection_layout.addRow("Backend:", self.backend)
        
        # Master address
        self.master_addr = components.create_line_edit(
            "localhost", 
            "Address of the master node"
        )
        connection_layout.addRow("Master Address:", self.master_addr)
        
        # Master port
        self.master_port = components.create_line_edit(
            "12355", 
            "Port of the master node"
        )
        connection_layout.addRow("Master Port:", self.master_port)
        
        # Timeout
        self.timeout = components.create_spin_box(
            60, 7200, 1800, 60,
            "Timeout for operations in seconds"
        )
        connection_layout.addRow("Timeout (seconds):", self.timeout)
        
        distributed_layout.addWidget(connection_group)
        
        # Strategy settings
        strategy_group = QtWidgets.QGroupBox("Distribution Strategies")
        strategy_layout = QtWidgets.QFormLayout(strategy_group)
        
        # Data loading strategy
        self.data_strategy = components.create_combo_box(
            ["distributed", "centralized"], 
            "distributed",
            "Strategy for data loading (distributed=each GPU loads a subset, centralized=rank 0 loads all)"
        )
        strategy_layout.addRow("Data Loading Strategy:", self.data_strategy)
        
        # Caching strategy
        self.cache_strategy = components.create_combo_box(
            ["distributed", "centralized"], 
            "distributed",
            "Strategy for latent caching (distributed=each GPU caches a subset, centralized=rank 0 caches all)"
        )
        strategy_layout.addRow("Latent Caching Strategy:", self.cache_strategy)
        
        distributed_layout.addWidget(strategy_group)
        
        # Advanced settings
        advanced_group = QtWidgets.QGroupBox("Advanced Settings")
        advanced_layout = QtWidgets.QVBoxLayout(advanced_group)
        
        self.detect_nvlink = components.create_checkbox(
            "Detect and optimize for NVLink", True,
            "Detect and apply optimizations for NVLink connections between GPUs"
        )
        advanced_layout.addWidget(self.detect_nvlink)
        
        self.find_unused_parameters = components.create_checkbox(
            "Find unused parameters", False,
            "Find unused parameters in forward pass (can be slower but needed for some models)"
        )
        advanced_layout.addWidget(self.find_unused_parameters)
        
        self.gradient_as_bucket_view = components.create_checkbox(
            "Gradient as bucket view", True,
            "Use gradient bucket view to reduce memory usage"
        )
        advanced_layout.addWidget(self.gradient_as_bucket_view)
        
        self.static_graph = components.create_checkbox(
            "Use static graph optimization", False,
            "Enable static graph optimization (for models without changing structure during training)"
        )
        advanced_layout.addWidget(self.static_graph)
        
        bucket_layout = QtWidgets.QHBoxLayout()
        bucket_layout.addWidget(QtWidgets.QLabel("Bucket size (MB):"))
        self.bucket_cap_mb = components.create_spin_box(
            5, 100, 25, 5,
            "Maximum bucket size in MiB for gradient communication"
        )
        bucket_layout.addWidget(self.bucket_cap_mb)
        bucket_layout.addStretch()
        advanced_layout.addLayout(bucket_layout)
        
        distributed_layout.addWidget(advanced_group)
        
        # Info and help section
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        
        info_text = QtWidgets.QLabel(
            "Distributed training allows utilizing multiple GPUs for faster model training. "
            "To use this feature, you need at least 2 GPUs in your system.\n\n"
            "When enabled, training will utilize all available GPUs. You must launch training "
            "with the train_distributed.py script instead of the standard training script.\n\n"
            "Example command:\n"
            "python scripts/train_distributed.py --config your_config.json"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        distributed_layout.addWidget(info_group)
        
        # Add a stretch to push everything up
        distributed_layout.addStretch()
        
        # Add the settings container
        layout.addWidget(self.distributed_settings)
        
        # Connect enable checkbox to show/hide settings
        self.enable_distributed.stateChanged.connect(self.toggle_distributed_settings)
        self.toggle_distributed_settings(self.enable_distributed.checkState())
    
    def refresh_gpu_info(self):
        """Refresh GPU information display."""
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 0:
            self.gpu_info_label.setText("No CUDA-capable GPUs detected")
            self.enable_distributed.setEnabled(False)
            self.enable_distributed.setChecked(False)
            return
            
        if gpu_count == 1:
            self.gpu_info_label.setText(f"1 GPU detected: {torch.cuda.get_device_name(0)}\n"
                             f"Distributed training requires at least 2 GPUs")
            self.enable_distributed.setEnabled(False)
            self.enable_distributed.setChecked(False)
            return
            
        # Multiple GPUs available, gather their info
        gpu_info = f"{gpu_count} GPUs detected:\n"
        for i in range(gpu_count):
            gpu_info += f"GPU {i}: {torch.cuda.get_device_name(i)}\n"
        
        self.gpu_info_label.setText(gpu_info)
        self.enable_distributed.setEnabled(True)
    
    def toggle_distributed_settings(self, state):
        """Show or hide distributed settings based on checkbox state."""
        self.distributed_settings.setVisible(state == QtCore.Qt.CheckState.Checked)
    
    def get_config(self) -> Dict:
        """
        Get the distributed configuration.
        
        Returns:
            Dictionary with distributed configuration
        """
        if not self.enable_distributed.isChecked():
            return {"enabled": False}
        
        return {
            "enabled": True,
            "backend": self.backend.currentText(),
            "master_addr": self.master_addr.text(),
            "master_port": self.master_port.text(),
            "timeout": self.timeout.value(),
            "detect_nvlink": self.detect_nvlink.isChecked(),
            "data_loading_strategy": self.data_strategy.currentText(),
            "latent_caching_strategy": self.cache_strategy.currentText(),
            "find_unused_parameters": self.find_unused_parameters.isChecked(),
            "gradient_as_bucket_view": self.gradient_as_bucket_view.isChecked(),
            "bucket_cap_mb": self.bucket_cap_mb.value(),
            "static_graph": self.static_graph.isChecked(),
        }
    
    def set_config(self, config: Dict):
        """
        Set the distributed configuration.
        
        Args:
            config: Dictionary with distributed configuration
        """
        if not config:
            self.enable_distributed.setChecked(False)
            return
            
        self.enable_distributed.setChecked(config.get("enabled", False))
        
        if "backend" in config:
            self.backend.setCurrentText(config["backend"])
            
        if "master_addr" in config:
            self.master_addr.setText(config["master_addr"])
            
        if "master_port" in config:
            self.master_port.setText(config["master_port"])
            
        if "timeout" in config:
            self.timeout.setValue(config["timeout"])
            
        if "detect_nvlink" in config:
            self.detect_nvlink.setChecked(config["detect_nvlink"])
            
        if "data_loading_strategy" in config:
            self.data_strategy.setCurrentText(config["data_loading_strategy"])
            
        if "latent_caching_strategy" in config:
            self.cache_strategy.setCurrentText(config["latent_caching_strategy"])
            
        if "find_unused_parameters" in config:
            self.find_unused_parameters.setChecked(config["find_unused_parameters"])
            
        if "gradient_as_bucket_view" in config:
            self.gradient_as_bucket_view.setChecked(config["gradient_as_bucket_view"])
            
        if "bucket_cap_mb" in config:
            self.bucket_cap_mb.setValue(config["bucket_cap_mb"])
            
        if "static_graph" in config:
            self.static_graph.setChecked(config["static_graph"])
