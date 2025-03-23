import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.distributed import DistributedBackend, DataDistributionStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.ui.components import PAD


class DistributedTrainingTab:
    """
    Tab for distributed training settings in the UI.
    Provides options for configuring multi-GPU training.
    """
    
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        """
        Initialize the distributed training tab.
        
        Args:
            master: The parent widget
            train_config: The training configuration
            ui_state: The UI state
        """
        self.train_config = train_config
        self.ui_state = ui_state
        
        self.frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, minsize=50)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)
        
        # Enable distributed training
        components.label(self.frame, 0, 0, "Enable Distributed Training",
                          tooltip="Enable distributed training across multiple GPUs")
        components.switch(self.frame, 0, 1, self.ui_state, "distributed.enabled")
        
        # Backend selection
        components.label(self.frame, 1, 0, "Backend",
                          tooltip="Communication backend to use (NCCL for NVIDIA GPUs, Gloo as fallback)")
        components.options_kv(self.frame, 1, 1, [
            ("NCCL", DistributedBackend.NCCL),
            ("Gloo", DistributedBackend.GLOO),
        ], self.ui_state, "distributed.backend")
        
        # Data loading strategy
        components.label(self.frame, 2, 0, "Data Loading Strategy",
                          tooltip="How to distribute data loading across GPUs")
        components.options_kv(self.frame, 2, 1, [
            ("Distributed", DataDistributionStrategy.DISTRIBUTED),
            ("Centralized", DataDistributionStrategy.CENTRALIZED),
        ], self.ui_state, "distributed.data_loading_strategy")
        
        # Latent caching strategy
        components.label(self.frame, 3, 0, "Latent Caching Strategy",
                          tooltip="How to distribute latent caching across GPUs")
        components.options_kv(self.frame, 3, 1, [
            ("Distributed", DataDistributionStrategy.DISTRIBUTED),
            ("Centralized", DataDistributionStrategy.CENTRALIZED),
        ], self.ui_state, "distributed.latent_caching_strategy")
        
        # NVLink detection
        components.label(self.frame, 4, 0, "Detect NVLink",
                          tooltip="Detect and optimize for NVLink connections between GPUs")
        components.switch(self.frame, 4, 1, self.ui_state, "distributed.detect_nvlink")
        
        # Advanced settings section
        section_label = components.label(self.frame, 5, 0, "Advanced Settings", pad=(PAD, PAD*2))
        section_label.configure(font=ctk.CTkFont(weight="bold"))
        
        # Find unused parameters
        components.label(self.frame, 6, 0, "Find Unused Parameters",
                          tooltip="Enable finding unused parameters in the forward pass (slower but more flexible)")
        components.switch(self.frame, 6, 1, self.ui_state, "distributed.find_unused_parameters")
        
        # Gradient as bucket view
        components.label(self.frame, 7, 0, "Gradient as Bucket View",
                          tooltip="Enable memory-efficient gradient views (recommended)")
        components.switch(self.frame, 7, 1, self.ui_state, "distributed.gradient_as_bucket_view")
        
        # Bucket cap MB
        components.label(self.frame, 8, 0, "Bucket Cap MB",
                          tooltip="Maximum size in MB for gradient buckets")
        components.entry(self.frame, 8, 1, self.ui_state, "distributed.bucket_cap_mb")
        
        # Launch method section
        section_label = components.label(self.frame, 9, 0, "Launch Settings", pad=(PAD, PAD*2))
        section_label.configure(font=ctk.CTkFont(weight="bold"))
        
        # Use torchrun
        components.label(self.frame, 10, 0, "Use torchrun",
                          tooltip="Use torchrun for launching distributed training (recommended)")
        components.switch(self.frame, 10, 1, self.ui_state, "distributed.use_torchrun")
        
        # Master address
        components.label(self.frame, 11, 0, "Master Address",
                          tooltip="Address of the master node (for multi-node training)")
        components.entry(self.frame, 11, 1, self.ui_state, "distributed.master_addr")
        
        # Master port
        components.label(self.frame, 11, 3, "Master Port",
                          tooltip="Port of the master node (for multi-node training)")
        components.entry(self.frame, 11, 4, self.ui_state, "distributed.master_port")
        
        # Timeout
        components.label(self.frame, 12, 0, "Timeout (seconds)",
                          tooltip="Timeout for operations in seconds")
        components.entry(self.frame, 12, 1, self.ui_state, "distributed.timeout_seconds")
        
        # Help text
        help_text = (
            "To run distributed training:\n\n"
            "1. Enable distributed training above\n"
            "2. Configure your settings\n"
            "3. Use train_distributed.py instead of train.py\n\n"
            "For single-node multi-GPU: python scripts/train_distributed.py --multi_gpu --config path/to/config.json\n\n"
            "For torchrun: torchrun --nproc_per_node=NUM_GPUS scripts/train_distributed.py --config path/to/config.json"
        )
        
        components.text(self.frame, 13, 0, help_text, colspan=5, height=8)
        
        self.frame.pack(fill="both", expand=1)
        
    def refresh_ui(self):
        """Refresh the UI elements based on current configuration."""
        pass
