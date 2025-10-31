"""
LIMference Inference Module
Handles SBI inference for different analysis methods
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial import KDTree
from tqdm import tqdm
import warnings

from sbi.inference import SNPE, SNRE, SNLE, NPE, NRE, SNL
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding


class InferenceEngine:
    """Main inference engine for LIMference package"""
    
    SUPPORTED_METHODS = ["SNPE", "SNRE", "SNLE", "NPE", "NRE", "SNL"]
    ANALYSIS_METHODS = ["power_spectrum", "pdf", "field_level", "field_level_unet", "field_level_resnet"]
    
    def __init__(self, prior: Optional[BoxUniform] = None):
        self.prior = prior
        self.posteriors = {}
        self.loss_histories = {}
        
    def create_prior(self, params_array: np.ndarray) -> BoxUniform:
        """Create BoxUniform prior from parameter array"""
        min_vals = params_array.min(axis=0)
        max_vals = params_array.max(axis=0)
        
        prior = BoxUniform(
            low=torch.tensor(min_vals, dtype=torch.float32),
            high=torch.tensor(max_vals, dtype=torch.float32)
        )
        self.prior = prior
        return prior
    
    def get_inference_method(self, method: str, embedding_net: Optional[Any] = None):
        """Get the appropriate inference method"""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported inference method: {method}. Choose from {self.SUPPORTED_METHODS}")
        
        method_map = {
            "SNPE": SNPE,
            "SNRE": SNRE,
            "SNLE": SNLE,
            "NPE": NPE,
            "NRE": NRE,
            "SNL": SNL
        }
        
        if embedding_net is not None and method == "NPE":
            neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)
            return NPE(prior=self.prior, density_estimator=neural_posterior)
        else:
            return method_map[method](prior=self.prior)
    
    def perform_single_round_inference(
        self,
        params: torch.Tensor,
        simulations: torch.Tensor,
        method: str = "SNPE",
        analysis_method: str = "power_spectrum",
        embedding_net: Optional[Any] = None,
        training_kwargs: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """Perform single round of inference"""
        
        print(f"Performing single-round {method} inference with {analysis_method} analysis...")
        
        # Get inference object
        inference = self.get_inference_method(method, embedding_net)
        
        # Append simulations
        inference = inference.append_simulations(params, simulations)
        
        # Train with custom kwargs if provided
        if training_kwargs is None:
            training_kwargs = {}
        
        density_estimator = inference.train(**training_kwargs)
        posterior = inference.build_posterior(density_estimator)
        
        # Store results
        key = f"{method}_{analysis_method}"
        self.posteriors[key] = posterior
        self.loss_histories[key] = inference._summary
        
        return posterior, inference._summary
    
    def perform_multi_round_inference(
        self,
        params_array: np.ndarray,
        simulations_array: np.ndarray,
        x_observation: torch.Tensor,
        method: str = "SNPE",
        analysis_method: str = "power_spectrum",
        num_rounds: int = 3,
        num_simulations_per_round: int = 100,
        embedding_net: Optional[Any] = None,
        training_kwargs: Optional[Dict] = None
    ) -> Tuple[Any, List[Dict]]:
        """Perform multi-round active learning inference"""
        
        print(f"Performing {num_rounds}-round {method} inference with {analysis_method} analysis...")
        
        if self.prior is None:
            self.create_prior(params_array)
        
        posteriors = []
        loss_histories = []
        proposal = self.prior
        
        # Build KDTree for efficient parameter matching
        kdtree = KDTree(params_array)
        
        for round_idx in range(num_rounds):
            print(f"Round {round_idx + 1}/{num_rounds}...")
            
            # Sample from current proposal
            theta_samples = proposal.sample((num_simulations_per_round,))
            theta_samples_np = theta_samples.numpy()
            
            # Find closest precomputed simulations
            _, indices = kdtree.query(theta_samples_np, k=1)
            indices = indices.flatten()
            
            matched_params = params_array[indices]
            matched_sims = simulations_array[indices]
            
            # Convert to tensors
            params_tensor = torch.tensor(matched_params, dtype=torch.float32)
            sims_tensor = torch.tensor(matched_sims, dtype=torch.float32)
            
            # Get inference object
            inference = self.get_inference_method(method, embedding_net)
            
            # Append simulations
            inference = inference.append_simulations(params_tensor, sims_tensor, proposal=proposal)
            
            # Train
            if training_kwargs is None:
                training_kwargs = {}
            
            density_estimator = inference.train(**training_kwargs)
            posterior = inference.build_posterior(density_estimator)
            
            posteriors.append(posterior)
            loss_histories.append(inference._summary)
            
            # Update proposal for next round
            proposal = posterior.set_default_x(x_observation)
        
        # Store final results
        key = f"{method}_{analysis_method}_multiround"
        self.posteriors[key] = posteriors[-1]
        self.loss_histories[key] = loss_histories
        
        return posteriors[-1], loss_histories


class ComparisonFramework:
    """Framework for comparing different inference methods"""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
        self.results = {}
        
    def run_comparison(
        self,
        params_array: np.ndarray,
        simulations_dict: Dict[str, np.ndarray],
        x_observation: torch.Tensor,
        methods: List[str] = ["SNPE", "NPE"],
        inference_kwargs: Optional[Dict] = None
    ) -> Dict:
        """Run comparison across different analysis methods and inference techniques"""
        
        results = {}
        
        for analysis_method, simulations in simulations_dict.items():
            print(f"\n{'='*50}")
            print(f"Analysis Method: {analysis_method}")
            print(f"{'='*50}")
            
            results[analysis_method] = {}
            
            # Convert to tensors
            params_tensor = torch.tensor(params_array, dtype=torch.float32)
            sims_tensor = torch.tensor(simulations, dtype=torch.float32)
            
            # Create embedding net if needed
            embedding_net = None
            if "field_level" in analysis_method:
                # Determine input shape from simulations
                if len(simulations.shape) == 3:  # Assuming (n_samples, height, width)
                    input_shape = simulations.shape[1:]
                else:
                    input_shape = (256, 256)  # Default
                embedding_net = CNNEmbedding(input_shape=input_shape)
            
            for method in methods:
                try:
                    print(f"\nRunning {method}...")
                    
                    # Single round inference
                    posterior, loss_history = self.engine.perform_single_round_inference(
                        params_tensor,
                        sims_tensor,
                        method=method,
                        analysis_method=analysis_method,
                        embedding_net=embedding_net,
                        training_kwargs=inference_kwargs
                    )
                    
                    # Sample from posterior
                    samples = posterior.sample((1000,), x=x_observation)
                    
                    # Calculate statistics
                    stats = self.calculate_statistics(samples.numpy(), params_array)
                    
                    results[analysis_method][method] = {
                        "posterior": posterior,
                        "samples": samples,
                        "loss_history": loss_history,
                        "statistics": stats
                    }
                    
                except Exception as e:
                    warnings.warn(f"Failed {method} for {analysis_method}: {str(e)}")
                    results[analysis_method][method] = {"error": str(e)}
        
        self.results = results
        return results
    
    def calculate_statistics(self, samples: np.ndarray, true_params: np.ndarray) -> Dict:
        """Calculate statistics for posterior samples"""
        stats = {
            "mean": np.mean(samples, axis=0),
            "std": np.std(samples, axis=0),
            "median": np.median(samples, axis=0),
            "quantiles": {
                "16": np.percentile(samples, 16, axis=0),
                "84": np.percentile(samples, 84, axis=0),
                "2.5": np.percentile(samples, 2.5, axis=0),
                "97.5": np.percentile(samples, 97.5, axis=0)
            }
        }
        
        # If we have true parameters for comparison
        if true_params is not None and len(true_params.shape) == 1:
            stats["bias"] = stats["mean"] - true_params
            stats["rmse"] = np.sqrt(np.mean((samples - true_params)**2, axis=0))
        
        return stats
    
    def summarize_results(self) -> Dict:
        """Create summary of comparison results"""
        summary = {}
        
        for analysis_method, method_results in self.results.items():
            summary[analysis_method] = {}
            
            for inference_method, results in method_results.items():
                if "error" in results:
                    summary[analysis_method][inference_method] = "Failed"
                else:
                    stats = results["statistics"]
                    summary[analysis_method][inference_method] = {
                        "mean": stats["mean"].tolist() if isinstance(stats["mean"], np.ndarray) else stats["mean"],
                        "std": stats["std"].tolist() if isinstance(stats["std"], np.ndarray) else stats["std"],
                        "coverage_68": (stats["quantiles"]["84"] - stats["quantiles"]["16"]).tolist() 
                                      if isinstance(stats["quantiles"]["84"], np.ndarray) else 
                                      stats["quantiles"]["84"] - stats["quantiles"]["16"]
                    }
                    
                    if "rmse" in stats:
                        summary[analysis_method][inference_method]["rmse"] = stats["rmse"].tolist() \
                            if isinstance(stats["rmse"], np.ndarray) else stats["rmse"]
        
        return summary


class ObservationGenerator:
    """Generate mock observations for testing"""
    
    def __init__(self, data_processor, simulation_loader):
        self.processor = data_processor
        self.loader = simulation_loader
        
    def create_observation(
        self,
        true_params: Dict[str, float],
        params_array: Dict[str, np.ndarray],
        analysis_method: str = "power_spectrum",
        signal_index: int = 0,
        noise_index: int = 0,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Create an observation for inference"""
        
        # Find closest parameters in array
        param_values = np.array([params_array[key][signal_index] for key in true_params.keys()])
        
        # Load signal (assuming you have a method to generate this)
        # This is simplified - you'd need the actual LIM simulation here
        signal = self._generate_signal(signal_index)
        
        if add_noise:
            noise = self.loader.load_noise_realization(noise_index)
            observation = signal + noise
        else:
            observation = signal
        
        # Process according to analysis method
        if analysis_method == "power_spectrum":
            k, pk = self.processor.compute_power_spectrum_2d(observation)
            result = pk
        elif analysis_method == "pdf":
            _, pdf = self.processor.compute_pdf(observation)
            result = pdf
        elif analysis_method.startswith("field_level"):
            result = observation.flatten()
        else:
            raise ValueError(f"Unknown analysis method: {analysis_method}")
        
        return torch.tensor(result, dtype=torch.float32), true_params
    
    def _generate_signal(self, index: int) -> np.ndarray:
        """Generate signal from simulation (placeholder)"""
        # This would call your actual LIM simulation code
        # For now, returning a placeholder
        warnings.warn("Using placeholder signal generation")
        return np.random.randn(self.processor.config.ngrid, self.processor.config.ngrid)


def create_training_kwargs(
    max_num_epochs: int = 50,
    stop_after_epochs: int = 5,
    training_batch_size: int = 50,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1
) -> Dict:
    """Create training kwargs for SBI"""
    return {
        "max_num_epochs": max_num_epochs,
        "stop_after_epochs": stop_after_epochs,
        "training_batch_size": training_batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "show_train_summary": True
    }
