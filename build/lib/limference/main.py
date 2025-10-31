"""
Main interface module for LIMference package
Provides high-level API for complete inference pipeline
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import json
from tqdm import tqdm
from datetime import datetime

# Import package components
from .config import LIMConfig, NetworkConfig, ParameterConfig, ConfigManager
from .core import DataProcessor, SimulationLoader, LIMSimulator, ParameterMatcher
from .networks import create_network
from .inference import InferenceEngine, ComparisonFramework, ObservationGenerator, create_training_kwargs
from .utils import set_random_seeds, calculate_statistics, plot_posterior_comparison, create_corner_plot, save_results, load_results


class LIMference:
    """
    Main interface for the LIMference package
    
    This class orchestrates the complete inference pipeline, managing:
    - Data loading and processing
    - Multiple analysis methods (power spectrum, PDF, field-level)
    - Multiple inference methods (SNPE, NPE, etc.)
    - Results comparison and visualization
    
    Args:
        config: LIMConfig object, dictionary, or path to config file
        param_config: Optional ParameterConfig object
        network_config: Optional NetworkConfig object
    """
    
    def __init__(
        self,
        config: Union[LIMConfig, Dict, str],
        param_config: Optional[ParameterConfig] = None,
        network_config: Optional[NetworkConfig] = None
    ):
        """Initialize LIMference with configuration"""
        
        # Handle different config input types
        if isinstance(config, str):
            self.config = LIMConfig.load(config)
        elif isinstance(config, dict):
            self.config = LIMConfig(**config)
        else:
            self.config = config
        
        # Set random seeds
        set_random_seeds(self.config.seed)
        
        # Initialize configurations
        self.param_config = param_config or ParameterConfig()
        self.network_config = network_config or NetworkConfig()
        
        # Create config manager
        self.config_manager = ConfigManager(self.config, self.network_config, self.param_config)
        
        # Initialize components
        self.processor = DataProcessor(self.config)
        self.loader = SimulationLoader(self.config)
        self.simulator = LIMSimulator(self.config, self.param_config)
        
        # Initialize inference components
        self.engine = InferenceEngine()
        self.comparison = ComparisonFramework(self.engine)
        self.obs_generator = ObservationGenerator(self.processor, self.loader)
        
        # Storage for data and results
        self.parameters = None
        self.simulations = {}
        self.observations = {}
        self.results = {}
        self.posteriors = {}
        
        # Neural networks for field-level analysis
        self.networks = {}
        
        # Status tracking
        self.is_initialized = False
        self.analysis_completed = False
        
        print(f"LIMference initialized with root: {self.config.root}")
        print(f"Device: {self.config.device}")
    
    def initialize_networks(self, force_reinit: bool = False):
        """
        Initialize neural networks for field-level analysis
        
        Args:
            force_reinit: Force reinitialization even if already initialized
        """
        if self.is_initialized and not force_reinit:
            print("Networks already initialized. Use force_reinit=True to reinitialize.")
            return
        
        print("Initializing neural networks...")
        
        # Create networks based on configuration
        self.networks = {
            'unet': create_network(
                'unet',
                in_channels=self.network_config.unet_in_channels,
                out_channels=self.network_config.unet_out_channels,
                init_features=self.network_config.unet_init_features
            ),
            'resnet': create_network(
                'resnet',
                in_channels=self.network_config.resnet_in_channels,
                num_classes=self.network_config.resnet_num_params,
                init_features=self.network_config.resnet_init_features
            ),
            'resnet_prob': create_network(
                'resnet_prob',
                in_channels=self.network_config.resnet_in_channels,
                num_params=self.network_config.resnet_num_params,
                init_features=self.network_config.resnet_init_features
            ),
            'cnn_attention': create_network(
                'cnn_attention',
                input_shape=self.network_config.cnn_input_shape,
                embedding_dim=self.network_config.cnn_embedding_dim
            ),
            'field_embedding': create_network(
                'field_embedding',
                input_shape=self.network_config.cnn_input_shape,
                embedding_dim=self.network_config.cnn_embedding_dim,
                architecture='cnn'
            )
        }
        
        # Move networks to device
        device = torch.device(self.config.device)
        for name, net in self.networks.items():
            self.networks[name] = net.to(device)
        
        self.is_initialized = True
        print(f"Networks initialized on {self.config.device}")
    
    def load_precomputed_data(
        self,
        params_file: Optional[str] = None,
        ps_file: Optional[str] = None,
        pdf_file: Optional[str] = None,
        field_file: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Load precomputed simulation data
        
        Args:
            params_file: Path to parameters file
            ps_file: Path to power spectrum file
            pdf_file: Path to PDF file
            field_file: Path to field-level data file
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of loaded data
        """
        loaded_data = {}
        
        print("Loading precomputed data...")
        
        # Load parameters
        if params_file:
            print(f"  Loading parameters from {params_file}")
            data = np.load(params_file, allow_pickle=True)
            
            # Handle different parameter formats
            if "parameters" in data:
                params_raw = data["parameters"]
            elif "params_array" in data:
                params_raw = data["params_array"]
            else:
                params_raw = data
            
            # Convert to standard format
            if isinstance(params_raw, np.ndarray) and len(params_raw.shape) == 2:
                self.parameters = params_raw
            elif isinstance(params_raw[0], dict):
                # Extract from dictionary format
                param_list = []
                for d in params_raw:
                    param_values = []
                    for param_name in self.param_config.param_names:
                        if param_name in d:
                            param_values.append(d[param_name])
                        elif param_name == "sigma8" and "sigma_8" in d:
                            param_values.append(d["sigma_8"])
                        elif param_name == "omega_m" and "Omega_m" in d:
                            param_values.append(d["Omega_m"])
                        else:
                            param_values.append(self.param_config.true_values.get(param_name, 0))
                    param_list.append(param_values)
                self.parameters = np.array(param_list)
            
            loaded_data["parameters"] = self.parameters
            print(f"    Loaded {len(self.parameters)} parameter sets")
        
        # Load power spectrum data
        if ps_file:
            print(f"  Loading power spectrum from {ps_file}")
            data = np.load(ps_file, allow_pickle=True)
            self.simulations["power_spectrum"] = {
                "k": data.get("k_2d", data.get("k", None)),
                "pk": data.get("pk_2d", data.get("pk", None))
            }
            loaded_data["power_spectrum"] = self.simulations["power_spectrum"]
            
            if self.simulations["power_spectrum"]["pk"] is not None:
                shape = self.simulations["power_spectrum"]["pk"].shape
                print(f"    Loaded power spectra with shape {shape}")
        
        # Load PDF data
        if pdf_file:
            print(f"  Loading PDF from {pdf_file}")
            data = np.load(pdf_file, allow_pickle=True)
            self.simulations["pdf"] = {
                "bins_log": data.get("lim_bin_edges_log_2d", None),
                "pdf_log": data.get("lim_pdf_log_2d", None),
                "bins_linear": data.get("lim_bin_edges_linear_2d", None),
                "pdf_linear": data.get("lim_pdf_linear_2d", None)
            }
            loaded_data["pdf"] = self.simulations["pdf"]
            
            if self.simulations["pdf"]["pdf_linear"] is not None:
                shape = self.simulations["pdf"]["pdf_linear"].shape
                print(f"    Loaded PDFs with shape {shape}")
        
        # Load field-level data
        if field_file:
            print(f"  Loading field-level data from {field_file}")
            data = np.load(field_file, allow_pickle=True)
            self.simulations["field_level"] = data.get("fields", data.get("maps", None))
            loaded_data["field_level"] = self.simulations["field_level"]
            
            if self.simulations["field_level"] is not None:
                shape = self.simulations["field_level"].shape
                print(f"    Loaded fields with shape {shape}")
        
        print(f"Successfully loaded: {list(loaded_data.keys())}")
        return loaded_data
    
    def generate_simulations(
        self,
        n_sims: Optional[int] = None,
        n_noise: Optional[int] = None,
        analysis_methods: Optional[List[str]] = None,
        save_output: bool = True,
        output_dir: Optional[str] = None,
        noise_reduction_factor: Optional[float] = None 
    ) -> Dict:
        """
        Generate new simulations for inference
        
        Args:
            n_sims: Number of signal simulations
            n_noise: Number of noise realizations per signal
            analysis_methods: List of analysis methods to generate
            save_output: Whether to save generated data
            output_dir: Directory to save output
            
        Returns:
            Dictionary of generated simulations
        """
        if n_sims is None:
            n_sims = self.config.n_signal_sims
        if n_noise is None:
            n_noise = self.config.num_noise_simulations
        if analysis_methods is None:
            analysis_methods = self.config.analysis_methods
        if noise_reduction_factor is None:
            noise_reduction_factor = self.config.noise_reduction_factor
        
        print(f"\nGenerating simulations:")
        print(f"  Signals: {n_sims}")
        print(f"  Noise realizations: {n_noise}")
        
        print(f"  Noise reduction factor: {noise_reduction_factor}")
        print(f"  Total: {n_sims * n_noise}")
        print(f"  Analysis methods: {analysis_methods}")
        
        # Load or generate parameters
        if self.parameters is None:
            params_dict = self.loader.load_parameters(
                self.param_config.param_names,
                nsims=n_sims
            )
            self.parameters = np.column_stack([params_dict[p] for p in self.param_config.param_names])
        
        # Generate noise realizations if needed
        noise_maps = []
        for i in range(n_noise):
            # Apply noise reduction when generating noise
            noise = self.processor.generate_noise_map(
                P_noise=self.config.noise_power,
                noise_reduction_factor=noise_reduction_factor
            )
            noise_maps.append(noise)
        
        # Initialize storage
        results = {
            "parameters": np.repeat(self.parameters[:n_sims], n_noise, axis=0),
            "simulations": {}
        }
        
        # Generate simulations for each analysis method
        for method in analysis_methods:
            print(f"\n  Processing {method}...")
            
            if method == "power_spectrum":
                all_pk = []
                k_values = None
                
                for i in tqdm(range(n_sims), desc="    Generating PS"):
                    # Generate signal (placeholder)
                    signal = self.simulator._generate_placeholder_map()
                    signal_2d = np.mean(signal, axis=2)
                    
                    for j in range(n_noise):
                        # Add noise and compute power spectrum
                        noisy_signal = signal_2d + noise_maps[j]
                        k, pk = self.processor.compute_power_spectrum_2d(noisy_signal)
                        all_pk.append(pk)
                        
                        if k_values is None:
                            k_values = k
                
                results["simulations"][method] = {
                    "k": k_values,
                    "pk": np.array(all_pk)
                }
            
            elif method == "pdf":
                all_pdf = []
                bins = None
                
                for i in tqdm(range(n_sims), desc="    Generating PDF"):
                    # Generate signal (placeholder)
                    signal = self.simulator._generate_placeholder_map()
                    signal_2d = np.mean(signal, axis=2)
                    
                    for j in range(n_noise):
                        # Add noise and compute PDF
                        noisy_signal = signal_2d + noise_maps[j]
                        bin_edges, pdf = self.processor.compute_pdf(noisy_signal, scale="linear")
                        all_pdf.append(pdf)
                        
                        if bins is None:
                            bins = bin_edges
                
                results["simulations"][method] = {
                    "bins": bins,
                    "pdf": np.array(all_pdf)
                }
            
            elif method in ["field_level", "field_level_unet", "field_level_resnet"]:
                all_fields = []
                
                # Initialize network if needed
                if not self.is_initialized:
                    self.initialize_networks()
                
                for i in tqdm(range(n_sims), desc="    Generating fields"):
                    # Generate signal (placeholder)
                    signal = self.simulator._generate_placeholder_map()
                    signal_2d = np.mean(signal, axis=2)
                    
                    for j in range(n_noise):
                        # Add noise
                        noisy_signal = signal_2d + noise_maps[j]
                        
                        # Process through network if specified
                        if "unet" in method:
                            processed = self.processor.process_field_level(
                                noisy_signal, method="unet", model=self.networks["unet"]
                            )
                        elif "resnet" in method:
                            processed = self.processor.process_field_level(
                                noisy_signal, method="resnet", model=self.networks["resnet_prob"]
                            )
                        else:
                            processed = noisy_signal.flatten()
                        
                        all_fields.append(processed)
                
                results["simulations"][method] = np.array(all_fields)
        
        # Store results
        self.simulations.update(results["simulations"])
        
        # Save if requested
        if save_output:
            if output_dir is None:
                output_dir = self.config.root / "generated_simulations"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = output_dir / f"simulations_{timestamp}.npz"
            
            np.savez_compressed(save_path, **results)
            print(f"\nSimulations saved to {save_path}")
        
        return results
    
    def create_observation(
        self,
        true_params: Dict[str, float],
        analysis_method: str = "power_spectrum",
        add_noise: bool = True,
        noise_level: float = 1e4,
        return_field: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Create an observation for inference
        
        Args:
            true_params: True parameter values
            analysis_method: Analysis method to use
            add_noise: Whether to add noise
            noise_level: Noise power level
            return_field: Whether to return the raw field
            
        Returns:
            Tuple of (observation tensor, true parameters dict)
        """
        # Find closest parameters in our grid
        if self.parameters is None:
            raise ValueError("No parameters loaded. Load data or generate simulations first.")
        
        # Create parameter matcher
        matcher = ParameterMatcher(self.parameters)
        true_param_array = np.array([true_params[p] for p in self.param_config.param_names])
        closest_idx = matcher.find_nearest(true_param_array)
        
        # Generate observation (using placeholder)
        signal = self.simulator._generate_placeholder_map()
        signal_2d = np.mean(signal, axis=2)
        
        if add_noise:
            noise = self.processor.generate_noise_map(noise_level)
            observation = signal_2d + noise
        else:
            observation = signal_2d
        
        # Process according to analysis method
        if analysis_method == "power_spectrum":
            _, pk = self.processor.compute_power_spectrum_2d(observation)
            result = pk
        elif analysis_method == "pdf":
            _, pdf = self.processor.compute_pdf(observation, scale="linear")
            result = pdf
        elif analysis_method.startswith("field_level"):
            if not self.is_initialized:
                self.initialize_networks()
            
            if "unet" in analysis_method:
                result = self.processor.process_field_level(
                    observation, method="unet", model=self.networks["unet"]
                )
            elif "resnet" in analysis_method:
                result = self.processor.process_field_level(
                    observation, method="resnet", model=self.networks["resnet_prob"]
                )
            else:
                result = observation.flatten()
        else:
            raise ValueError(f"Unknown analysis method: {analysis_method}")
        
        # Convert to tensor
        obs_tensor = torch.tensor(result, dtype=torch.float32)
        
        # Store observation
        self.observations[analysis_method] = {
            "tensor": obs_tensor,
            "true_params": true_params,
            "field": observation if return_field else None
        }
        
        return obs_tensor, true_params
    
    def run_inference_comparison(
        self,
        true_params: Optional[Dict[str, float]] = None,
        observation_index: Optional[int] = None,
        analysis_methods: Optional[List[str]] = None,
        inference_methods: Optional[List[str]] = None,
        training_kwargs: Optional[Dict] = None,
        n_posterior_samples: int = 10000,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete inference comparison across methods
        
        Args:
            true_params: True parameter values for observation
            observation_index: Index of observation to use (alternative to true_params)
            analysis_methods: List of analysis methods to compare
            inference_methods: List of inference methods to use
            training_kwargs: Training arguments for SBI
            n_posterior_samples: Number of posterior samples to draw
            verbose: Whether to print progress
            
        Returns:
            Dictionary of inference results
        """
        # Set defaults
        if analysis_methods is None:
            analysis_methods = self.config.analysis_methods
        if inference_methods is None:
            inference_methods = self.config.inference_methods
        if training_kwargs is None:
            training_kwargs = self.config.get_training_kwargs()
        
        # Validate data availability
        for method in analysis_methods:
            if method not in self.simulations and "field" not in method and method != "combined_ps_pdf":
                raise ValueError(f"No simulation data for {method}. Load or generate data first.")
        
        if verbose:
            print("\n" + "="*70)
            print("LIMFERENCE: Running Inference Comparison")
            print("="*70)
            print(f"Analysis methods: {analysis_methods}")
            print(f"Inference methods: {inference_methods}")
        
        # Create or get observation
        if true_params is not None:
            obs_tensors = {}
            for method in analysis_methods:
                if method == "combined_ps_pdf":
                    # Create combined observation from components
                    _, pk = self.create_observation(true_params, "power_spectrum")
                    _, pdf = self.create_observation(true_params, "pdf")
                    combined = torch.cat([pk, pdf])
                    obs_tensors[method] = combined
                else:
                    obs_tensor, _ = self.create_observation(true_params, method)
                    obs_tensors[method] = obs_tensor
        elif observation_index is not None:
            obs_tensors = {}
            true_params = {
                param: self.parameters[observation_index, i]
                for i, param in enumerate(self.param_config.param_names)
            }
            for method in analysis_methods:
                if method == "power_spectrum":
                    obs_tensors[method] = torch.tensor(
                        self.simulations["power_spectrum"]["pk"][observation_index],
                        dtype=torch.float32
                    )
                elif method == "pdf":
                    obs_tensors[method] = torch.tensor(
                        self.simulations["pdf"]["pdf_linear"][observation_index],
                        dtype=torch.float32
                    )
                elif method == "combined_ps_pdf":
                    # Handle combined observation
                    obs_tensors[method] = torch.tensor(
                        self.simulations["combined_ps_pdf"][observation_index],
                        dtype=torch.float32
                    )
                elif "field" in method:
                    obs_tensors[method] = torch.tensor(
                        self.simulations["field_level"][observation_index],
                        dtype=torch.float32
                    )
        else:
            raise ValueError("Must provide either true_params or observation_index")
        
        # Create prior
        if self.engine.prior is None:
            self.engine.create_prior(self.parameters)
        
        # Run inference for each combination
        all_results = {}
        
        for analysis_method in analysis_methods:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Analysis Method: {analysis_method}")
                print(f"{'='*50}")
            
            # Get simulation data
            if analysis_method == "power_spectrum":
                sim_data = self.simulations["power_spectrum"]["pk"]
            elif analysis_method == "pdf":
                sim_data = self.simulations["pdf"]["pdf_linear"]
            elif analysis_method == "combined_ps_pdf":
                # Handle combined PS+PDF case
                if "combined_ps_pdf" in self.simulations:
                    sim_data = self.simulations["combined_ps_pdf"]
                else:
                    print(f"  Skipping {analysis_method} - combined data not found")
                    continue
            elif "field" in analysis_method:
                sim_data = self.simulations.get("field_level", None)
                if sim_data is None:
                    print(f"  Skipping {analysis_method} - no field data available")
                    continue
            else:
                print(f"  Skipping {analysis_method} - unknown method")
                continue
            
            # Prepare tensors
            param_tensor = torch.tensor(self.parameters, dtype=torch.float32)
            
            # Handle multiple noise realizations
            if len(sim_data.shape) == 3:  # (n_sims, n_noise, n_features)
                n_sims, n_noise, n_features = sim_data.shape
                sim_data = sim_data.reshape(n_sims * n_noise, n_features)
                param_tensor = param_tensor.repeat_interleave(n_noise, dim=0)
            
            sim_tensor = torch.tensor(sim_data, dtype=torch.float32)
            
            # Get observation for this method
            x_obs = obs_tensors[analysis_method]
            
            # Run each inference method
            for inf_method in inference_methods:
                if verbose:
                    print(f"\n  Running {inf_method}...")
                
                key = f"{analysis_method}_{inf_method}"
                
                try:
                    # Determine if we need embedding network
                    embedding_net = None
                    if "field" in analysis_method:
                        if not self.is_initialized:
                            self.initialize_networks()
                        embedding_net = self.networks["field_embedding"]
                    
                    # Run inference
                    posterior, loss_history = self.engine.perform_single_round_inference(
                        param_tensor,
                        sim_tensor,
                        method=inf_method,
                        analysis_method=analysis_method,
                        embedding_net=embedding_net,
                        training_kwargs=training_kwargs
                    )
                    
                    # Sample from posterior
                    samples = posterior.sample((n_posterior_samples,), x=x_obs)
                    
                    # Calculate statistics
                    true_param_array = np.array([true_params[p] for p in self.param_config.param_names])
                    stats = calculate_statistics(samples.numpy(), true_param_array)
                    
                    # Store results
                    all_results[key] = {
                        "posterior": posterior,
                        "samples": samples,
                        "loss_history": loss_history,
                        "statistics": stats,
                        "true_params": true_params,
                        "observation": x_obs
                    }
                    
                    if verbose:
                        print(f"    Success! Mean: {stats['mean']}")
                        print(f"             Std:  {stats['std']}")
                        if 'rmse' in stats:
                            print(f"             RMSE: {stats['rmse']}")
                    
                except Exception as e:
                    if verbose:
                        print(f"    Failed: {str(e)}")
                    all_results[key] = {"error": str(e)}
        
        # Store results
        self.results = all_results
        self.analysis_completed = True
        
        # Print summary
        if verbose:
            self._print_summary(all_results)
        
        return all_results
    
    def visualize_results(
        self,
        param_names: Optional[List[str]] = None,
        methods_to_plot: Optional[List[str]] = None,
        true_values: Optional[Dict[str, float]] = None,
        save_plots: bool = False,
        plot_dir: Optional[str] = None,
        show_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualization of results
        
        Args:
            param_names: Parameters to visualize
            methods_to_plot: Specific method combinations to plot
            true_values: True parameter values
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of figure objects
        """
        if not self.analysis_completed:
            warnings.warn("No analysis results to visualize. Run inference first.")
            return {}
        
        if param_names is None:
            param_names = self.param_config.param_names
        
        if methods_to_plot is None:
            methods_to_plot = [k for k in self.results.keys() if "error" not in self.results[k]]
        
        if plot_dir is None:
            plot_dir = self.config.root / "plots"
        
        if save_plots:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # 1. Posterior comparison plot
        print("\nCreating posterior comparison plots...")
        
        # Prepare results for plotting
        plot_results = {}
        for method in methods_to_plot:
            if method in self.results and "error" not in self.results[method]:
                plot_results[method] = self.results[method]
        
        if plot_results:
            fig = plot_posterior_comparison(
                plot_results,
                param_names,
                true_values=true_values or self.results[methods_to_plot[0]].get("true_params"),
                save_path=str(plot_dir / "posterior_comparison.pdf") if save_plots else None
            )
            figures["posterior_comparison"] = fig
        
        # 2. Corner plots for each method
        print("Creating corner plots...")
        
        for method in methods_to_plot:
            if method in self.results and "samples" in self.results[method]:
                samples = self.results[method]["samples"].numpy()
                true_vals = self.results[method].get("true_params")
                
                if true_vals:
                    true_array = np.array([true_vals[p] for p in param_names])
                else:
                    true_array = None
                
                fig = create_corner_plot(
                    [samples],
                    param_names=param_names,
                    labels=[method],
                    true_values=true_array,
                    save_path=str(plot_dir / f"corner_{method}.pdf") if save_plots else None
                )
                
                if fig is not None:
                    figures[f"corner_{method}"] = fig
        
        # 3. Combined corner plot
        if len(methods_to_plot) > 1:
            print("Creating combined corner plot...")
            
            samples_list = []
            labels_list = []
            
            for method in methods_to_plot[:4]:  # Limit to 4 for clarity
                if method in self.results and "samples" in self.results[method]:
                    samples_list.append(self.results[method]["samples"].numpy())
                    labels_list.append(method)
            
            if samples_list:
                true_vals = self.results[methods_to_plot[0]].get("true_params")
                if true_vals:
                    true_array = np.array([true_vals[p] for p in param_names])
                else:
                    true_array = None
                
                fig = create_corner_plot(
                    samples_list,
                    param_names=param_names,
                    labels=labels_list,
                    true_values=true_array,
                    save_path=str(plot_dir / "corner_combined.pdf") if save_plots else None
                )
                
                if fig is not None:
                    figures["corner_combined"] = fig
        
        # 4. Summary statistics plot
        print("Creating summary statistics plot...")
        fig = self._plot_summary_statistics(param_names, save_plots, plot_dir)
        if fig is not None:
            figures["summary_stats"] = fig
        
        if show_plots:
            plt.show()
        
        print(f"\nCreated {len(figures)} figures")
        if save_plots:
            print(f"Plots saved to {plot_dir}")
        
        return figures
    
    def save_results(
        self,
        save_dir: Optional[str] = None,
        prefix: str = "limference_results"
    ) -> Dict[str, Path]:
        """
        Save all results to disk
        
        Args:
            save_dir: Directory to save results
            prefix: Prefix for filenames
            
        Returns:
            Dictionary of saved file paths
        """
        if save_dir is None:
            save_dir = self.config.root / "results"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_{timestamp}"
        
        # Save configuration
        self.config_manager.save_all(save_dir / f"{prefix}_configs")
        
        # Save results
        saved_files = save_results(self.results, save_dir, prefix)
        
        print(f"\nResults saved to {save_dir}")
        
        return saved_files
    
    def _print_summary(self, results: Dict):
        """Print formatted summary of results"""
        print("\n" + "="*70)
        print("INFERENCE RESULTS SUMMARY")
        print("="*70)
        
        # Organize by analysis method
        by_analysis = {}
        for key, result in results.items():
            if "error" not in result:
                analysis, inference = key.rsplit("_", 1)
                if analysis not in by_analysis:
                    by_analysis[analysis] = {}
                by_analysis[analysis][inference] = result
        
        for analysis_method, inf_results in by_analysis.items():
            print(f"\n{analysis_method.upper()}")
            print("-"*50)
            
            for inf_method, result in inf_results.items():
                stats = result.get("statistics", {})
                print(f"  {inf_method}:")
                
                if "mean" in stats:
                    mean_str = ", ".join([f"{m:.3f}" for m in stats["mean"]])
                    print(f"    Mean: [{mean_str}]")
                
                if "std" in stats:
                    std_str = ", ".join([f"{s:.3f}" for s in stats["std"]])
                    print(f"    Std:  [{std_str}]")
                
                if "rmse" in stats:
                    rmse_str = ", ".join([f"{r:.3f}" for r in stats["rmse"]])
                    print(f"    RMSE: [{rmse_str}]")
                
                if "coverage" in stats:
                    cov_str = ", ".join([f"{k}:{v}" for k, v in stats["coverage"].items()])
                    print(f"    Coverage: {cov_str}")
    
    def _plot_summary_statistics(
        self,
        param_names: List[str],
        save_plots: bool,
        plot_dir: Path
    ) -> Optional[plt.Figure]:
        """Create summary statistics plot"""
        
        # Collect statistics
        methods = []
        rmse_values = []
        bias_values = []
        
        for key, result in self.results.items():
            if "error" not in result and "statistics" in result:
                methods.append(key)
                stats = result["statistics"]
                
                if "rmse" in stats:
                    rmse_values.append(stats["rmse"])
                if "bias" in stats:
                    bias_values.append(stats["bias"])
        
        if not rmse_values:
            return None
        
        # Create plot
        n_params = len(param_names)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4 + 0.3 * len(methods)))
        
        # RMSE plot
        ax = axes[0]
        rmse_array = np.array(rmse_values)
        
        for i, param in enumerate(param_names):
            ax.barh(np.arange(len(methods)) + i * 0.3 / n_params,
                   rmse_array[:, i],
                   height=0.3 / n_params,
                   label=self.param_config.get_labels([param])[0])
        
        ax.set_yticks(np.arange(len(methods)) + 0.15)
        ax.set_yticklabels(methods)
        ax.set_xlabel("RMSE")
        ax.set_title("Root Mean Square Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bias plot
        ax = axes[1]
        bias_array = np.array(bias_values)
        
        for i, param in enumerate(param_names):
            ax.barh(np.arange(len(methods)) + i * 0.3 / n_params,
                   bias_array[:, i],
                   height=0.3 / n_params,
                   label=self.param_config.get_labels([param])[0])
        
        ax.set_yticks(np.arange(len(methods)) + 0.15)
        ax.set_yticklabels(methods)
        ax.set_xlabel("Bias")
        ax.set_title("Parameter Bias")
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            fig.savefig(plot_dir / "summary_statistics.pdf", dpi=150, bbox_inches='tight')
        
        return fig


# Convenience functions
def quick_inference(
    data_dir: str,
    true_params: Dict[str, float],
    ps_file: Optional[str] = None,
    pdf_file: Optional[str] = None,
    params_file: Optional[str] = None,
    analysis_methods: List[str] = ["power_spectrum", "pdf"],
    inference_methods: List[str] = ["SNPE"],
    **kwargs
) -> Tuple[LIMference, Dict]:
    """
    Quick inference run with minimal setup
    
    Args:
        data_dir: Directory containing data
        true_params: True parameter values
        ps_file: Power spectrum file
        pdf_file: PDF file
        params_file: Parameters file
        analysis_methods: Analysis methods to use
        inference_methods: Inference methods to use
        **kwargs: Additional arguments for LIMConfig
        
    Returns:
        Tuple of (LIMference object, results dictionary)
    """
    # Create configuration
    config = LIMConfig(root=data_dir, **kwargs)
    
    # Initialize LIMference
    lim = LIMference(config)
    
    # Load data
    lim.load_precomputed_data(
        params_file=params_file,
        ps_file=ps_file,
        pdf_file=pdf_file
    )
    
    # Run inference
    results = lim.run_inference_comparison(
        true_params=true_params,
        analysis_methods=analysis_methods,
        inference_methods=inference_methods
    )
    
    # Visualize
    lim.visualize_results()
    
    return lim, results
