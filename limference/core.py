"""
Core module for LIMference package
Handles data processing, simulation loading, and analysis methods
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from tqdm import tqdm
from scipy.spatial import KDTree

# Import configurations
from .config import LIMConfig, NetworkConfig, ParameterConfig

# Try importing LIM-specific libraries
try:
    import limpy.lines as ll
    import limpy.powerspectra as lp
    LIMPY_AVAILABLE = True
except ImportError:
    warnings.warn("limpy not found. Some functionality will be limited.")
    ll = None
    lp = None
    LIMPY_AVAILABLE = False


class DataProcessor:
    """
    Handle data processing for different analysis methods
    
    Args:
        config: LIMConfig object with simulation parameters
    """
    
    def __init__(self, config: LIMConfig):
        self.config = config
        self._set_random_seed(config.seed)
        
    def _set_random_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def compute_power_spectrum_2d(
        self,
        field: np.ndarray,
        boxsize: Optional[float] = None,
        ngrid: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D power spectrum
        
        Args:
            field: 2D field array
            boxsize: Box size in Mpc/h
            ngrid: Number of grid points
            
        Returns:
            Tuple of (k, P(k)) arrays
        """
        if boxsize is None:
            boxsize = self.config.boxsize
        if ngrid is None:
            ngrid = self.config.ngrid
            
        if LIMPY_AVAILABLE and lp is not None:
            k, pk = lp.get_pk2d(field, boxsize, boxsize, ngrid, ngrid)
        else:
            # Fallback implementation using numpy FFT
            k, pk = self._compute_power_spectrum_numpy(field, boxsize, ngrid)
        
        return k, pk
    
    def _compute_power_spectrum_numpy(
        self,
        field: np.ndarray,
        boxsize: float,
        ngrid: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback power spectrum calculation using numpy"""
        # Compute FFT
        fft_field = np.fft.fft2(field)
        power = np.abs(fft_field) ** 2
        
        # Get k values
        kx = np.fft.fftfreq(ngrid, d=boxsize/ngrid) * 2 * np.pi
        ky = np.fft.fftfreq(ngrid, d=boxsize/ngrid) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k = np.sqrt(kx**2 + ky**2)
        
        # Bin the power spectrum
        k_bins = np.logspace(np.log10(k[k>0].min()), np.log10(k.max()), 
                            self.config.power_spectrum_bins)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        pk = np.zeros(len(k_centers))
        for i, (k_min, k_max) in enumerate(zip(k_bins[:-1], k_bins[1:])):
            mask = (k >= k_min) & (k < k_max)
            if mask.sum() > 0:
                pk[i] = power[mask].mean()
        
        # Normalize
        pk *= (boxsize / ngrid) ** 2 / ngrid ** 2
        
        return k_centers, pk
    
    def compute_pdf(
        self,
        field: np.ndarray,
        scale: str = "log",
        density: bool = True,
        cutoff: float = 0,
        n_bins: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute probability distribution function (voxel intensity distribution)
        
        Args:
            field: Input field array
            scale: 'log' or 'linear' scale for binning
            density: Whether to normalize to get probability density
            cutoff: Minimum value cutoff
            n_bins: Number of bins (uses config value if None)
            
        Returns:
            Tuple of (bin_edges, pdf) arrays
        """
        if n_bins is None:
            n_bins = self.config.n_pdf_bins
            
        flat_field = field.flatten()
        
        if scale == "log":
            valid_field = flat_field[flat_field > cutoff]
            if len(valid_field) == 0:
                return np.array([]), np.array([])
            
            log_field = np.log10(valid_field)
            range_pdf = (log_field.min(), log_field.max())
            
            # Add small buffer to range
            buffer = 0.01 * (range_pdf[1] - range_pdf[0])
            range_pdf = (range_pdf[0] - buffer, range_pdf[1] + buffer)
            
            hist, bin_edges = np.histogram(log_field, bins=n_bins,
                                          range=range_pdf, density=density)
        else:  # linear
            valid_field = flat_field[flat_field > cutoff]
            if len(valid_field) == 0:
                return np.array([]), np.array([])
            
            range_pdf = (valid_field.min(), valid_field.max())
            
            # Add small buffer to range
            buffer = 0.01 * (range_pdf[1] - range_pdf[0])
            range_pdf = (range_pdf[0] - buffer, range_pdf[1] + buffer)
            
            hist, bin_edges = np.histogram(valid_field, bins=n_bins,
                                          range=range_pdf, density=density)
        
        return bin_edges, hist
    
    def generate_noise_map(
    self,
    P_noise: float,
    shape: Optional[Tuple[int, int]] = None,
    noise_reduction_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate noise realization in real space

        Args:
            P_noise: Noise power spectrum amplitude
            shape: Shape of output array (uses config if None)
            noise_reduction_factor: Factor to reduce noise by (uses config if None)

        Returns:
            2D noise map
        """
        if shape is None:
            shape = (self.config.ngrid, self.config.ngrid)

        if noise_reduction_factor is None:
            noise_reduction_factor = self.config.noise_reduction_factor

        delta_x = self.config.boxsize / self.config.ngrid

        # Apply noise reduction factor to the noise amplitude
        effective_P_noise = P_noise / noise_reduction_factor
        sigma_noise = np.sqrt(effective_P_noise / (delta_x ** 2))

        noise_map = np.random.normal(0, sigma_noise, shape)

        return noise_map
    
    def process_field_level(
        self,
        field: np.ndarray,
        method: str = "raw",
        model: Optional[Any] = None
    ) -> np.ndarray:
        """
        Process field for field-level inference
        
        Args:
            field: Input field array
            method: Processing method ('raw', 'unet', 'resnet', etc.)
            model: Neural network model for processing
            
        Returns:
            Processed field as 1D array
        """
        if method == "raw":
            return field.flatten()
        
        elif method in ["unet", "resnet", "cnn"] and model is not None:
            # Convert to tensor
            field_tensor = torch.tensor(field, dtype=torch.float32)
            
            # Add batch and channel dimensions if needed
            if field_tensor.dim() == 2:
                field_tensor = field_tensor.unsqueeze(0).unsqueeze(0)
            elif field_tensor.dim() == 3:
                field_tensor = field_tensor.unsqueeze(1)
            
            # Process through model
            with torch.no_grad():
                if method == "resnet" and hasattr(model, 'sample'):
                    # Probabilistic ResNet
                    mean, var = model(field_tensor)
                    output = torch.cat([mean, var], dim=1)
                else:
                    output = model(field_tensor)
            
            return output.flatten().numpy()
        
        else:
            raise ValueError(f"Unknown field processing method: {method}")
    
    def standardize_data(
        self,
        data: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize data to zero mean and unit variance
        
        Args:
            data: Input data array
            mean: Pre-computed mean (calculated if None)
            std: Pre-computed std (calculated if None)
            
        Returns:
            Tuple of (standardized_data, mean, std)
        """
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
            std[std == 0] = 1  # Avoid division by zero
        
        standardized = (data - mean) / std
        
        return standardized, mean, std


class SimulationLoader:
    """
    Load and manage simulation data
    
    Args:
        config: LIMConfig object with paths and parameters
    """
    
    def __init__(self, config: LIMConfig):
        self.config = config
        self.root = Path(config.root)
        self.cache = {}  # Cache for loaded data
        
    def load_parameters(
        self,
        param_names: List[str],
        nsims: Optional[int] = None,
        file_name: str = "parameters_saved.npz"
    ) -> Dict[str, np.ndarray]:
        """
        Load parameters from saved file
        
        Args:
            param_names: List of parameter names to load
            nsims: Number of simulations to load (None for all)
            file_name: Name of the parameters file
            
        Returns:
            Dictionary with parameter arrays
        """
        params_file = self.root / file_name
        
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        # Check cache
        cache_key = f"params_{file_name}_{nsims}"
        if cache_key in self.cache:
            return self._filter_params(self.cache[cache_key], param_names)
        
        with np.load(params_file, allow_pickle=True) as data:
            params_dict = {}
            
            for param in param_names:
                if param in data:
                    values = data[param]
                    if nsims is not None:
                        values = values[:nsims]
                    params_dict[param] = values
                else:
                    warnings.warn(f"Parameter '{param}' not found in {params_file}")
            
        # Cache the result
        self.cache[cache_key] = params_dict
        
        return params_dict
    
    def _filter_params(
        self,
        params_dict: Dict[str, np.ndarray],
        param_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Filter parameters dictionary to requested names"""
        return {k: v for k, v in params_dict.items() if k in param_names}
    
    def load_halo_catalog(self, index: int) -> Dict:
        """
        Load halo catalog for given index
        
        Args:
            index: Simulation index
            
        Returns:
            Dictionary with halo catalog data
        """
        halo_file = self.root / f"halocat_80_256_{index}.npz"
        
        if not halo_file.exists():
            raise FileNotFoundError(f"Halo catalog not found: {halo_file}")
        
        # Check cache
        cache_key = f"halo_{index}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        with np.load(halo_file, allow_pickle=True) as data:
            catalog = dict(data)
        
        # Cache the result
        self.cache[cache_key] = catalog
        
        return catalog
    
    def load_noise_realization(
    self,
    index: int,
    noise_dir: Optional[str] = None,
    noise_reduction_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Load pre-computed noise realization with optional reduction

        Args:
            index: Noise realization index
            noise_dir: Directory containing noise files (uses root if None)
            noise_reduction_factor: Factor to reduce noise by

        Returns:
            2D noise array (possibly scaled)
        """
        if noise_dir:
            noise_file = Path(noise_dir) / f"noise_2d_{index}.npz"
        else:
            noise_file = self.root / f"noise_2d_{index}.npz"

        if not noise_file.exists():
            raise FileNotFoundError(f"Noise file not found: {noise_file}")

        with np.load(noise_file) as data:
            noise = data["grid"]

            # Apply noise reduction if specified
            if noise_reduction_factor is None:
                noise_reduction_factor = self.config.noise_reduction_factor

            if noise_reduction_factor != 1.0:
                # Scale down the noise amplitude
                noise = noise / np.sqrt(noise_reduction_factor)

        return noise
    
    def save_noise_realizations(
        self,
        n_realizations: int,
        sigma_noise_pix: float = 1e3,
        noise_dir: Optional[str] = None
    ):
        """
        Generate and save noise realizations
        
        Args:
            n_realizations: Number of realizations to generate
            sigma_noise_pix: Noise standard deviation per pixel
            noise_dir: Directory to save noise files (uses root if None)
        """
        if noise_dir:
            save_dir = Path(noise_dir)
        else:
            save_dir = self.root
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(n_realizations), desc="Generating noise"):
            noise = np.random.normal(0, sigma_noise_pix,
                                    (self.config.ngrid, self.config.ngrid))
            save_path = save_dir / f"noise_2d_{i}.npz"
            np.savez_compressed(save_path, grid=noise, sigma_pix=sigma_noise_pix)
        
        print(f"Saved {n_realizations} noise realizations to {save_dir}")
    
    def load_simulation_batch(
        self,
        indices: List[int],
        data_type: str = "intensity"
    ) -> np.ndarray:
        """
        Load a batch of simulations
        
        Args:
            indices: List of simulation indices
            data_type: Type of data to load ('intensity', 'halo', etc.)
            
        Returns:
            Array of simulation data
        """
        data_list = []
        
        for idx in tqdm(indices, desc=f"Loading {data_type}"):
            if data_type == "intensity":
                # This would load actual intensity maps
                # Placeholder for now
                data = np.random.randn(self.config.ngrid, self.config.ngrid)
            elif data_type == "halo":
                data = self.load_halo_catalog(idx)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            data_list.append(data)
        
        return np.array(data_list) if data_type == "intensity" else data_list
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        print("Cache cleared")


class LIMSimulator:
    """
    Interface to LIM simulations using limpy
    
    Args:
        config: LIMConfig object
        param_config: ParameterConfig object
    """
    
    def __init__(self, config: LIMConfig, param_config: ParameterConfig):
        self.config = config
        self.param_config = param_config
        
        if not LIMPY_AVAILABLE:
            warnings.warn("limpy not available. LIM simulations will use placeholders.")
    
    def generate_intensity_map(
        self,
        index: int,
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """
        Generate intensity map for given parameters
        
        Args:
            index: Simulation index
            parameters: Dictionary of cosmological/astrophysical parameters
            
        Returns:
            3D intensity map
        """
        if not LIMPY_AVAILABLE or ll is None:
            # Return placeholder
            return self._generate_placeholder_map()
        
        # Load halo catalog
        halo_file = f"{self.config.root}/halocat_80_256_{index}.npz"
        
        # Create LIM simulation
        lim_sim = ll.lim_sims(
            halo_file,
            self.config.redshift,
            model_name=self.config.model_name,
            line_name=self.config.line_name,
            halo_cutoff_mass=self.config.mmin,
            halocat_type="input_cat",
            parameters=parameters,
            ngrid_x=self.config.ngrid,
            ngrid_y=self.config.ngrid,
            ngrid_z=self.config.ngrid,
            boxsize_x=self.config.boxsize,
            boxsize_y=self.config.boxsize,
            boxsize_z=self.config.boxsize,
            nu_obs=220,
            theta_fwhm=1,
            dnu_obs=2.2
        )
        
        # Generate intensity grid
        intensity_map = lim_sim.make_intensity_grid()
        
        return intensity_map
    
    def _generate_placeholder_map(self) -> np.ndarray:
        """Generate placeholder intensity map for testing"""
        # Generate correlated Gaussian field as placeholder
        ngrid = self.config.ngrid
        
        # Create random field in Fourier space
        k = np.fft.fftfreq(ngrid, d=1.0).reshape(-1, 1, 1)
        k2 = k**2 + k.T**2 + np.rollaxis(k, 0, 3)**2
        
        # Power spectrum (simple power law)
        pk = np.where(k2 > 0, k2**(-1.5), 0)
        
        # Random phases
        phases = np.random.randn(ngrid, ngrid, ngrid) + 1j * np.random.randn(ngrid, ngrid, ngrid)
        
        # Apply power spectrum
        field_k = phases * np.sqrt(pk)
        
        # Transform to real space
        field = np.real(np.fft.ifftn(field_k))
        
        # Make positive and add noise
        field = np.abs(field) + np.random.lognormal(0, 0.5, field.shape)
        
        return field
    
    def process_to_2d(self, intensity_3d: np.ndarray) -> np.ndarray:
        """
        Process 3D intensity map to 2D
        
        Args:
            intensity_3d: 3D intensity map
            
        Returns:
            2D projected map
        """
        # Simple mean along line of sight
        return np.mean(intensity_3d, axis=2)


class ParameterMatcher:
    """
    Match parameters using KDTree for efficient nearest neighbor search
    
    Args:
        params_array: Array of parameters (n_samples, n_params)
    """
    
    def __init__(self, params_array: np.ndarray):
        self.params_array = params_array
        self.kdtree = KDTree(params_array)
        
    def find_nearest(
        self,
        query_params: Union[np.ndarray, Dict[str, float]],
        k: int = 1
    ) -> Union[int, List[int]]:
        """
        Find nearest parameter sets
        
        Args:
            query_params: Query parameters (array or dict)
            k: Number of nearest neighbors
            
        Returns:
            Index or list of indices of nearest parameter sets
        """
        if isinstance(query_params, dict):
            query_params = np.array(list(query_params.values()))
        
        query_params = query_params.reshape(1, -1)
        
        dist, ind = self.kdtree.query(query_params, k=k)
        
        if k == 1:
            return ind[0]
        else:
            return ind.tolist()
    
    def find_within_radius(
        self,
        query_params: Union[np.ndarray, Dict[str, float]],
        radius: float
    ) -> List[int]:
        """
        Find all parameter sets within given radius
        
        Args:
            query_params: Query parameters
            radius: Search radius
            
        Returns:
            List of indices within radius
        """
        if isinstance(query_params, dict):
            query_params = np.array(list(query_params.values()))
        
        query_params = query_params.reshape(1, -1)
        
        indices = self.kdtree.query_ball_point(query_params[0], radius)
        
        return indices
