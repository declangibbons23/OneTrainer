"""
Configuration for distributed training.
"""
from enum import Enum


class Backend(str, Enum):
    """Backend for distributed communication."""
    NCCL = "nccl"  # GPU training (recommended for multi-GPU)
    GLOO = "gloo"  # CPU training or Windows


class DataDistributionStrategy(str, Enum):
    """Strategy for distributing data in distributed training."""
    DISTRIBUTED = "distributed"  # Each process loads a subset of the data
    CENTRALIZED = "centralized"  # Process 0 loads all data and distributes


class DistributedConfig:
    """Configuration for distributed training."""
    
    def __init__(self):
        # Enable/disable distributed training
        self.enabled = False
        
        # Backend for distributed communication
        self.backend = Backend.NCCL
        
        # Connection parameters
        self.master_addr = "localhost"
        self.master_port = "12355"
        self.timeout = 1800  # 30 minutes
        
        # Optimization settings
        self.detect_nvlink = True  # Automatically detect and optimize for NVLink
        
        # Distribution strategies
        self.data_loading_strategy = DataDistributionStrategy.DISTRIBUTED
        self.latent_caching_strategy = DataDistributionStrategy.DISTRIBUTED
        
        # DDP settings
        self.find_unused_parameters = False  # Finding unused parameters can reduce performance
        self.gradient_as_bucket_view = True  # Gradient as bucket view helps reduce memory usage
        self.bucket_cap_mb = 25  # Maximum bucket size in MiB
        self.static_graph = False  # Static graph optimization (if model doesn't change during training)
        
    def __bool__(self):
        """Return whether distributed training is enabled."""
        return self.enabled
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "timeout": self.timeout,
            "detect_nvlink": self.detect_nvlink,
            "data_loading_strategy": self.data_loading_strategy,
            "latent_caching_strategy": self.latent_caching_strategy,
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "bucket_cap_mb": self.bucket_cap_mb,
            "static_graph": self.static_graph,
        }
        
    def from_dict(self, d):
        """Load from dictionary."""
        if "enabled" in d:
            self.enabled = d["enabled"]
            
        if "backend" in d:
            self.backend = Backend(d["backend"])
            
        if "master_addr" in d:
            self.master_addr = d["master_addr"]
            
        if "master_port" in d:
            self.master_port = d["master_port"]
            
        if "timeout" in d:
            self.timeout = d["timeout"]
            
        if "detect_nvlink" in d:
            self.detect_nvlink = d["detect_nvlink"]
            
        if "data_loading_strategy" in d:
            self.data_loading_strategy = DataDistributionStrategy(d["data_loading_strategy"])
            
        if "latent_caching_strategy" in d:
            self.latent_caching_strategy = DataDistributionStrategy(d["latent_caching_strategy"])
            
        if "find_unused_parameters" in d:
            self.find_unused_parameters = d["find_unused_parameters"]
            
        if "gradient_as_bucket_view" in d:
            self.gradient_as_bucket_view = d["gradient_as_bucket_view"]
            
        if "bucket_cap_mb" in d:
            self.bucket_cap_mb = d["bucket_cap_mb"]
            
        if "static_graph" in d:
            self.static_graph = d["static_graph"]
            
        return self
