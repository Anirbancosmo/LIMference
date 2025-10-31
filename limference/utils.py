"""
Utility functions for LIMference package
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # For reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_statistics(
    samples: np.ndarray,
    true_params: Optional[np.ndarray] = None,
    confidence_levels: List[float] = [68, 95]
) -> Dict:
    """
    Calculate comprehensive statistics from posterior samples.
    
    Args:
        samples: Array of samples (n_samples, n_params)
        true_params: True parameter values for comparison
        confidence_levels: Confidence levels for credible intervals
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        "mean": np.mean(samples, axis=0),
        "median": np.median(samples, axis=0),
        "std": np.std(samples, axis=0),
        "mode": _estimate_mode(samples),
        "quantiles": {}
    }
    
    # Calculate quantiles for confidence intervals
    for level in confidence_levels:
        lower = (100 - level) / 2
        upper = 100 - lower
        stats["quantiles"][f"q{lower:.1f}"] = np.percentile(samples, lower, axis=0)
        stats["quantiles"][f"q{upper:.1f}"] = np.percentile(samples, upper, axis=0)
    
    # Special quantiles
    stats["quantiles"]["q16"] = np.percentile(samples, 16, axis=0)
    stats["quantiles"]["q84"] = np.percentile(samples, 84, axis=0)
    
    # If true parameters provided, calculate errors
    if true_params is not None:
        stats["bias"] = stats["mean"] - true_params
        stats["rmse"] = np.sqrt(np.mean((samples - true_params)**2, axis=0))
        stats["mae"] = np.mean(np.abs(samples - true_params), axis=0)
        
        # Coverage check
        stats["coverage"] = _check_coverage(samples, true_params, confidence_levels)
    
    return stats


def _estimate_mode(samples: np.ndarray) -> np.ndarray:
    """Estimate mode using kernel density estimation."""
    try:
        from scipy.stats import gaussian_kde
        modes = []
        for i in range(samples.shape[1]):
            kde = gaussian_kde(samples[:, i])
            x = np.linspace(samples[:, i].min(), samples[:, i].max(), 1000)
            density = kde(x)
            modes.append(x[np.argmax(density)])
        return np.array(modes)
    except ImportError:
        # Fallback to median if scipy not available
        return np.median(samples, axis=0)


def _check_coverage(
    samples: np.ndarray,
    true_params: np.ndarray,
    confidence_levels: List[float]
) -> Dict[str, bool]:
    """Check if true parameters fall within credible intervals."""
    coverage = {}
    for level in confidence_levels:
        lower = (100 - level) / 2
        upper = 100 - lower
        q_lower = np.percentile(samples, lower, axis=0)
        q_upper = np.percentile(samples, upper, axis=0)
        
        in_interval = np.all((true_params >= q_lower) & (true_params <= q_upper))
        coverage[f"{level}%"] = bool(in_interval)
    
    return coverage


def plot_posterior_comparison(
    results_dict: Dict,
    param_names: List[str],
    true_values: Optional[Dict[str, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create comparison plots of posterior distributions.
    
    Args:
        results_dict: Dictionary with method names as keys and results as values
        param_names: List of parameter names
        true_values: Dictionary of true parameter values
        figsize: Figure size (width, height)
        save_path: Path to save figure
        **kwargs: Additional plotting arguments
    
    Returns:
        Matplotlib figure object
    """
    n_methods = len(results_dict)
    n_params = len(param_names)
    
    if figsize is None:
        figsize = (5 * n_methods, 4 * n_params)
    
    fig, axes = plt.subplots(n_params, n_methods, figsize=figsize, squeeze=False)
    
    colors = kwargs.get('colors', plt.cm.tab10(np.linspace(0, 1, n_methods)))
    
    for i, (method_name, stats) in enumerate(results_dict.items()):
        for j, param_name in enumerate(param_names):
            ax = axes[j, i]
            
            if 'samples' in stats:
                samples = stats['samples'][:, j]
                
                # Plot histogram
                n, bins, patches = ax.hist(
                    samples, bins=50, density=True,
                    alpha=0.7, color=colors[i],
                    edgecolor='black', linewidth=0.5
                )
                
                # Add KDE overlay if requested
                if kwargs.get('show_kde', True):
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(samples)
                        x = np.linspace(samples.min(), samples.max(), 200)
                        ax.plot(x, kde(x), color='black', linewidth=1.5, alpha=0.7)
                    except ImportError:
                        pass
                
                # Add statistics
                mean_val = stats.get('mean', [0, 0])[j]
                median_val = stats.get('median', [0, 0])[j]
                
                ax.axvline(mean_val, color='green', linestyle='-', 
                          linewidth=1.5, label='Mean', alpha=0.8)
                ax.axvline(median_val, color='blue', linestyle='--',
                          linewidth=1.5, label='Median', alpha=0.8)
                
                # Add credible intervals
                if 'quantiles' in stats:
                    q16 = stats['quantiles'].get('q16', [0, 0])[j]
                    q84 = stats['quantiles'].get('q84', [0, 0])[j]
                    ax.axvspan(q16, q84, alpha=0.2, color='gray',
                              label='68% CI')
            
            # Add true value if provided
            if true_values and param_name in true_values:
                ax.axvline(true_values[param_name], color='red',
                          linestyle='--', linewidth=2, label='True')
            
            # Formatting
            ax.set_xlabel(param_name, fontsize=12)
            if i == 0:
                ax.set_ylabel('Probability Density', fontsize=12)
            
            ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
            
            if j == 0 and i == 0:
                ax.legend(loc='best', fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_corner_plot(
    samples_list: List[np.ndarray],
    param_names: List[str],
    labels: Optional[List[str]] = None,
    true_values: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """
    Create corner plot for posterior samples.
    
    Args:
        samples_list: List of sample arrays
        param_names: Parameter names for axes
        labels: Labels for different sample sets
        true_values: True parameter values
        save_path: Path to save figure
        **kwargs: Additional arguments for plotting
    
    Returns:
        Figure object or None if pygtc not available
    """
    try:
        import pygtc
        
        # Format parameter names for LaTeX
        formatted_names = []
        for name in param_names:
            if name == "sigma8" or name == "sigma_8":
                formatted_names.append(r"$\sigma_8$")
            elif name == "omega_m" or name == "Omega_m":
                formatted_names.append(r"$\Omega_m$")
            elif name == "omh2":
                formatted_names.append(r"$\Omega_m h^2$")
            elif name == "h":
                formatted_names.append(r"$h$")
            else:
                formatted_names.append(name)
        
        # Create corner plot
        figure_size = kwargs.get('figureSize', 'MNRAS_page')
        
        GTC = pygtc.plotGTC(
            chains=samples_list,
            paramNames=formatted_names,
            chainLabels=labels,
            truths=true_values,
            figureSize=figure_size,
            plotDensity=kwargs.get('plotDensity', True),
            smoothingKernel=kwargs.get('smoothingKernel', 1),
            nContourLevels=kwargs.get('nContourLevels', 3),
            sigmaContourLevels=kwargs.get('sigmaContourLevels', True),
            **{k: v for k, v in kwargs.items() 
               if k not in ['figureSize', 'plotDensity', 'smoothingKernel', 
                           'nContourLevels', 'sigmaContourLevels']}
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Corner plot saved to {save_path}")
        
        return GTC
        
    except ImportError:
        warnings.warn("pygtc not installed. Install with: pip install pygtc")
        
        # Fallback to simple corner plot
        try:
            import corner
            
            # Combine samples if multiple provided
            if len(samples_list) > 1:
                combined_samples = np.vstack(samples_list)
            else:
                combined_samples = samples_list[0]
            
            fig = corner.corner(
                combined_samples,
                labels=formatted_names if 'formatted_names' in locals() else param_names,
                truths=true_values,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                **kwargs
            )
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Corner plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            warnings.warn("Neither pygtc nor corner installed. Cannot create corner plot.")
            return None


def save_results(
    results: Dict,
    save_dir: Union[str, Path],
    prefix: str = "limference"
) -> Dict[str, Path]:
    """
    Save inference results to disk.
    
    Args:
        results: Dictionary of results to save
        save_dir: Directory to save results
        prefix: Prefix for filenames
    
    Returns:
        Dictionary of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save numpy arrays
    np_data = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            np_data[key] = value
        elif isinstance(value, torch.Tensor):
            np_data[key] = value.numpy()
        elif isinstance(value, dict):
            # Flatten nested dictionaries
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                    np_data[f"{key}_{sub_key}"] = (
                        sub_value if isinstance(sub_value, np.ndarray)
                        else sub_value.numpy()
                    )
    
    if np_data:
        np_file = save_dir / f"{prefix}_arrays.npz"
        np.savez_compressed(np_file, **np_data)
        saved_files['arrays'] = np_file
    
    # Save metadata as JSON
    import json
    metadata = {}
    for key, value in results.items():
        if isinstance(value, (str, int, float, bool, list)):
            metadata[key] = value
        elif isinstance(value, dict) and all(
            isinstance(v, (str, int, float, bool, list))
            for v in value.values()
        ):
            metadata[key] = value
    
    if metadata:
        json_file = save_dir / f"{prefix}_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = json_file
    
    print(f"Results saved to {save_dir}")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path.name}")
    
    return saved_files


def load_results(
    save_dir: Union[str, Path],
    prefix: str = "limference"
) -> Dict:
    """
    Load saved inference results.
    
    Args:
        save_dir: Directory containing saved results
        prefix: Prefix used when saving
    
    Returns:
        Dictionary of loaded results
    """
    save_dir = Path(save_dir)
    results = {}
    
    # Load numpy arrays
    np_file = save_dir / f"{prefix}_arrays.npz"
    if np_file.exists():
        with np.load(np_file, allow_pickle=True) as data:
            for key in data.files:
                results[key] = data[key]
    
    # Load metadata
    json_file = save_dir / f"{prefix}_metadata.json"
    if json_file.exists():
        import json
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            results.update(metadata)
    
    return results
