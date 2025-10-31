"""
LIMference: Line Intensity Mapping Inference Package
=====================================================

A comprehensive package for comparing inference methods on Line Intensity Mapping data
using power spectrum, PDF, and field-level analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core components
from .core import (
    DataProcessor,
    SimulationLoader,
    LIMSimulator,
    ParameterMatcher  # Add this if you want it available
)

# Import config components
from .config import (
    LIMConfig,
    NetworkConfig,
    ParameterConfig,
    ConfigManager
)

# Import network components
from .networks import (
    UNet,
    ResNetProbabilistic,
    ResNet,
    CNNWithAttention,
    create_network
)

# Import inference components
from .inference import (
    InferenceEngine,
    ComparisonFramework,
    ObservationGenerator,
    create_training_kwargs
)

# Import main interface
from .main import (
    LIMference,
    quick_inference  # This function exists in main.py
)

# Import utilities
from .utils import (
    set_random_seeds,
    create_corner_plot,
    plot_posterior_comparison,
    calculate_statistics,
    save_results,
    load_results
)

# Define what should be imported with "from limference import *"
__all__ = [
    # Main class
    "LIMference",
    
    # Core components
    "LIMConfig",
    "DataProcessor",
    "SimulationLoader",
    "LIMSimulator",
    
    # Neural networks
    "UNet",
    "ResNetProbabilistic",
    
    # Inference
    "InferenceEngine",
    "ComparisonFramework",
    "ObservationGenerator",
    
    # Utilities
    "quick_inference",
    "create_training_kwargs",
    "set_random_seeds",
    
    # Plotting
    "create_corner_plot",
    "plot_posterior_comparison",
    
    # Version info
    "__version__",
]

# Package metadata
__metadata__ = {
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": "Line Intensity Mapping Inference Package",
    "url": "https://github.com/yourusername/limference",
    "license": "MIT",
}

def get_version():
    """Return the package version."""
    return __version__

def show_config():
    """Display package configuration and dependencies."""
    import sys
    import torch
    import numpy as np
    
    print(f"LIMference version: {__version__}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        import sbi
        print(f"SBI version: {sbi.__version__}")
    except ImportError:
        print("SBI: Not installed")
    
    try:
        import limpy
        print("limpy: Available")
    except ImportError:
        print("limpy: Not available (some features may be limited)")

# Display package info when imported
if __name__ == "__main__":
    show_config()
