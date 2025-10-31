"""
Configuration module for LIMference package
Handles all configuration and parameter management
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import yaml
import warnings


@dataclass
class LIMConfig:
    """Main configuration class for LIM simulations and inference"""
    
    # Required parameters
    root: str  # Root directory for data
    
    # Simulation parameters
    boxsize: float = 80.01  # Box size in Mpc/h
    redshift: float = 3.60  # Redshift
    ngrid: int = 256  # Number of grid points per dimension
    n_signal_sims: int = 10  # Number of signal simulations
    num_noise_simulations: int = 1000  # Number of noise realizations per signal
    
    # Model parameters
    sfr_model: str = "Behroozi19"  # Star formation rate model
    model_name: str = "Alma_scaling"  # Line emission model
    line_name: str = "CII158"  # Emission line
    mmin: float = 1e10  # Minimum halo mass
    
    # Noise parameters (ADD THESE NEW LINES)
    noise_power: float = 1e4  # Base noise power
    noise_reduction_factor: float = 1.0  # Factor to reduce noise by (1.0 = no reduction)
    
    # Analysis parameters
    n_pdf_bins: int = 40  # Number of bins for PDF
    power_spectrum_bins: int = 50  # Number of k-bins for power spectrum
    
    # Inference parameters
    inference_methods: List[str] = field(default_factory=lambda: ["SNPE", "NPE"])
    analysis_methods: List[str] = field(default_factory=lambda: ["power_spectrum", "pdf"])
    
    # Training parameters
    max_num_epochs: int = 100
    stop_after_epochs: int = 10
    training_batch_size: int = 50
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1

    # Random seed
    seed: int = 42
    
    # Device settings
    device: str = "cpu"  # "cpu" or "cuda"
    
    def __post_init__(self):
        """Validate and process configuration after initialization"""
        # Convert root to Path object
        self.root = Path(self.root)
        
        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        self._validate()
        
        # Set device
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Using CPU.")
            self.device = "cpu"
    
    def _validate(self):
        """Validate configuration parameters"""
        if self.ngrid <= 0:
            raise ValueError(f"ngrid must be positive, got {self.ngrid}")
        
        if self.boxsize <= 0:
            raise ValueError(f"boxsize must be positive, got {self.boxsize}")
        
        if self.n_signal_sims <= 0:
            raise ValueError(f"n_signal_sims must be positive, got {self.n_signal_sims}")
        
        if self.num_noise_simulations < 0:
            raise ValueError(f"num_noise_simulations must be non-negative, got {self.num_noise_simulations}")
        
        if self.n_pdf_bins <= 0:
            raise ValueError(f"n_pdf_bins must be positive, got {self.n_pdf_bins}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if not 0 <= self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in [0, 1), got {self.validation_fraction}")
        
        # ADD THIS NEW VALIDATION
        if self.noise_reduction_factor <= 0:
            raise ValueError(f"noise_reduction_factor must be positive, got {self.noise_reduction_factor}")
        
        if self.noise_power <= 0:
            raise ValueError(f"noise_power must be positive, got {self.noise_power}")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        # Convert Path back to string for serialization
        config_dict['root'] = str(config_dict['root'])
        return config_dict
    
    def save(self, path: Union[str, Path], format: str = 'json'):
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        path = Path(path)
        config_dict = self.to_dict()
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format == 'yaml':
            try:
                with open(path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                warnings.warn("PyYAML not installed. Saving as JSON instead.")
                self.save(path.with_suffix('.json'), format='json')
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'yaml'.")
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LIMConfig':
        """
        Load configuration from file
        
        Args:
            path: Path to configuration file
            
        Returns:
            LIMConfig object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                with open(path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required to load YAML files. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        
        return cls(**config_dict)
    
    def get_training_kwargs(self) -> Dict:
        """Get training arguments for SBI"""
        return {
            'max_num_epochs': self.max_num_epochs,
            'stop_after_epochs': self.stop_after_epochs,
            'training_batch_size': self.training_batch_size,
            'learning_rate': self.learning_rate,
            'validation_fraction': self.validation_fraction,
            'show_train_summary': True
        }
    
    def summary(self) -> str:
        """Generate a summary string of the configuration"""
        summary = "LIMference Configuration\n"
        summary += "=" * 50 + "\n"
        summary += f"Root directory: {self.root}\n"
        summary += f"Simulation box: {self.boxsize} Mpc/h at z={self.redshift}\n"
        summary += f"Grid: {self.ngrid}³ cells\n"
        summary += f"Simulations: {self.n_signal_sims} signals × {self.num_noise_simulations} noise\n"
        summary += f"Line: {self.line_name} with {self.model_name} model\n"
        summary += f"Analysis methods: {', '.join(self.analysis_methods)}\n"
        summary += f"Inference methods: {', '.join(self.inference_methods)}\n"
        summary += f"Device: {self.device}\n"
        summary += f"Random seed: {self.seed}\n"
        return summary


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures"""
    
    # U-Net parameters
    unet_in_channels: int = 1
    unet_out_channels: int = 1
    unet_init_features: int = 32
    
    # ResNet parameters
    resnet_in_channels: int = 1
    resnet_num_params: int = 2
    resnet_init_features: int = 64
    
    # CNN embedding parameters
    cnn_input_shape: tuple = (256, 256)
    cnn_embedding_dim: int = 128
    
    # Training parameters
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: str = "relu"  # relu, leaky_relu, elu
    
    # Optimizer settings
    optimizer: str = "adam"  # adam, sgd, rmsprop
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    
    def get_optimizer_kwargs(self) -> Dict:
        """Get optimizer arguments"""
        if self.optimizer == "adam":
            return {"weight_decay": self.weight_decay}
        elif self.optimizer == "sgd":
            return {"momentum": self.momentum, "weight_decay": self.weight_decay}
        elif self.optimizer == "rmsprop":
            return {"weight_decay": self.weight_decay}
        else:
            return {}


@dataclass
class ParameterConfig:
    """Configuration for cosmological and astrophysical parameters"""
    
    # Parameter names
    param_names: List[str] = field(default_factory=lambda: ["sigma8", "omega_m"])
    
    # Parameter ranges (for prior)
    param_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "sigma8": (0.4, 1.3),
        "omega_m": (0.1, 0.6),
        "h": (0.6, 0.8),
        "omh2": (0.12, 0.25),
        "a_off": (4.0, 12.0),
        "b_off": (0.0, 2.0)
    })
    
    # Parameter labels for plotting
    param_labels: Dict[str, str] = field(default_factory=lambda: {
        "sigma8": r"$\sigma_8$",
        "omega_m": r"$\Omega_m$",
        "h": r"$h$",
        "omh2": r"$\Omega_m h^2$",
        "a_off": r"$A_{\mathrm{off}}$",
        "b_off": r"$B_{\mathrm{off}}$"
    })
    
    # True/fiducial values
    true_values: Dict[str, float] = field(default_factory=lambda: {
        "sigma8": 0.8,
        "omega_m": 0.3,
        "h": 0.7,
        "omh2": 0.147
    })
    
    def get_param_bounds(self, param_names: Optional[List[str]] = None) -> tuple:
        """
        Get parameter bounds for specified parameters
        
        Args:
            param_names: List of parameter names. If None, use self.param_names
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        import numpy as np
        
        if param_names is None:
            param_names = self.param_names
        
        lower_bounds = []
        upper_bounds = []
        
        for param in param_names:
            if param not in self.param_ranges:
                raise ValueError(f"Unknown parameter: {param}")
            
            low, high = self.param_ranges[param]
            lower_bounds.append(low)
            upper_bounds.append(high)
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def get_labels(self, param_names: Optional[List[str]] = None) -> List[str]:
        """Get formatted labels for parameters"""
        if param_names is None:
            param_names = self.param_names
        
        return [self.param_labels.get(p, p) for p in param_names]


class ConfigManager:
    """Manager for handling multiple configuration objects"""
    
    def __init__(
        self,
        lim_config: Optional[LIMConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        param_config: Optional[ParameterConfig] = None
    ):
        self.lim = lim_config or LIMConfig(root="./data")
        self.network = network_config or NetworkConfig()
        self.params = param_config or ParameterConfig()
    
    def save_all(self, directory: Union[str, Path]):
        """Save all configurations to a directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save LIM config
        self.lim.save(directory / "lim_config.json")
        
        # Save network config
        with open(directory / "network_config.json", 'w') as f:
            json.dump(asdict(self.network), f, indent=2)
        
        # Save parameter config
        with open(directory / "param_config.json", 'w') as f:
            json.dump(asdict(self.params), f, indent=2)
        
        print(f"All configurations saved to {directory}")
    
    @classmethod
    def load_all(cls, directory: Union[str, Path]) -> 'ConfigManager':
        """Load all configurations from a directory"""
        directory = Path(directory)
        
        # Load LIM config
        lim_config = LIMConfig.load(directory / "lim_config.json")
        
        # Load network config
        with open(directory / "network_config.json", 'r') as f:
            network_config = NetworkConfig(**json.load(f))
        
        # Load parameter config
        with open(directory / "param_config.json", 'r') as f:
            param_config = ParameterConfig(**json.load(f))
        
        return cls(lim_config, network_config, param_config)
    
    def summary(self) -> str:
        """Generate complete configuration summary"""
        summary = self.lim.summary()
        summary += "\nNetwork Configuration:\n"
        summary += f"  U-Net features: {self.network.unet_init_features}\n"
        summary += f"  ResNet features: {self.network.resnet_init_features}\n"
        summary += f"  Optimizer: {self.network.optimizer}\n"
        summary += "\nParameter Configuration:\n"
        summary += f"  Parameters: {', '.join(self.params.param_names)}\n"
        summary += f"  True values: {self.params.true_values}\n"
        return summary


# Utility functions for configuration
def create_default_config(root_dir: str, **kwargs) -> LIMConfig:
    """Create a default LIMConfig with optional overrides"""
    default_params = {"root": root_dir}
    default_params.update(kwargs)
    return LIMConfig(**default_params)


def merge_configs(base_config: LIMConfig, updates: Dict) -> LIMConfig:
    """Merge configuration updates into base configuration"""
    config_dict = base_config.to_dict()
    config_dict.update(updates)
    return LIMConfig(**config_dict)
