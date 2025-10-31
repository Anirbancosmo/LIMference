"""
Convergence and Validation Module for LIMference
Tests SBI training convergence, network learning, and inference quality
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
from scipy import stats
from tqdm import tqdm
import json

import pandas as pd

from .config import LIMConfig, ParameterConfig
from .inference import InferenceEngine, create_training_kwargs
from .utils import calculate_statistics, create_corner_plot


class ConvergenceAnalyzer:
    """
    Analyze convergence and performance of SBI inference
    """
    
    def __init__(self, config: LIMConfig, param_config: ParameterConfig):
        self.config = config
        self.param_config = param_config
        self.results = {}
        
    def analyze_loss_convergence(
        self,
        loss_history: Dict,
        method_name: str = "SNPE",
        plot: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze training loss convergence
        
        Returns:
            Dictionary with convergence metrics
        """
        metrics = {}
        
        if not loss_history or 'train_log_probs' not in loss_history:
            warnings.warn(f"No training history available for {method_name}")
            return metrics
        
        train_losses = loss_history['train_log_probs']
        val_losses = loss_history.get('validation_log_probs', [])
        
        # Calculate convergence metrics
        metrics['final_train_loss'] = train_losses[-1] if train_losses else None
        metrics['final_val_loss'] = val_losses[-1] if val_losses else None
        metrics['n_epochs'] = len(train_losses)
        
        # Check for convergence
        if len(train_losses) > 10:
            # Look at last 10% of training
            window = max(1, len(train_losses) // 10)
            recent_losses = train_losses[-window:]
            
            # Calculate stability (std of recent losses)
            metrics['loss_stability'] = np.std(recent_losses)
            
            # Calculate improvement rate
            if len(train_losses) > 1:
                improvement = (train_losses[0] - train_losses[-1]) / abs(train_losses[0])
                metrics['total_improvement'] = improvement
                
                # Check if still improving
                recent_improvement = (recent_losses[0] - recent_losses[-1]) / abs(recent_losses[0])
                metrics['recent_improvement'] = recent_improvement
                metrics['converged'] = abs(recent_improvement) < 0.01  # Less than 1% improvement
            
        # Check for overfitting
        if val_losses and train_losses:
            # Calculate gap between train and validation
            gap = val_losses[-1] - train_losses[-1]
            metrics['train_val_gap'] = gap
            metrics['potential_overfitting'] = gap > 0.1 * abs(train_losses[-1])
        
        # Plot if requested
        if plot:
            self._plot_loss_curves(train_losses, val_losses, method_name, save_path)
        
        return metrics
    
    def _plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        method_name: str,
        save_path: Optional[str] = None
    ):
        """Plot training and validation loss curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Log Probability', fontsize=12)
        ax.set_title(f'{method_name} Training Convergence', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add convergence indicators
        if len(train_losses) > 10:
            # Mark potential convergence point
            window = max(1, len(train_losses) // 10)
            recent_losses = train_losses[-window:]
            if np.std(recent_losses) < 0.01 * abs(np.mean(recent_losses)):
                ax.axvline(len(train_losses) - window, color='green', 
                          linestyle='--', alpha=0.5, label='Convergence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        
    def analyze_pdf_sensitivity(self):
        """Check how sensitive the PDF is to parameter changes"""

        # First ensure we have data
        if not self.data_loaded:
            print("Please load data first!")
            return

        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: PDF variation with omega_m at fixed sigma8
        ax = axes[0, 0]

        # Find simulations with sigma8 in narrow range (around median)
        median_sigma8 = np.median(self.params_array[:, 0])
        tolerance = 0.02
        mask = np.abs(self.params_array[:500, 0] - median_sigma8) < tolerance

        if np.sum(mask) > 10:  # Need at least 10 samples
            subset_params = self.params_array[:500][mask]
            subset_pdf = self.pdf_sim_linear[:500][mask]

            # Sort by omega_m for clearer visualization
            sort_idx = np.argsort(subset_params[:, 1])
            subset_params = subset_params[sort_idx]
            subset_pdf = subset_pdf[sort_idx]

            # Plot several PDFs with different omega_m
            n_samples = min(5, len(subset_params))
            colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

            for i in range(n_samples):
                ax.plot(subset_pdf[i], alpha=0.7, color=colors[i],
                       label=f'Ωₘ={subset_params[i, 1]:.3f}')

            ax.set_xlabel('PDF bin')
            ax.set_ylabel('PDF value')
            ax.set_title(f'PDFs at σ₈≈{median_sigma8:.3f}, varying Ωₘ')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Plot 2: Mean PDF for different omega_m ranges
        ax = axes[0, 1]

        # Split data into low and high omega_m
        omega_m_percentiles = np.percentile(self.params_array[:, 1], [25, 75])
        omega_m_low = self.params_array[:500, 1] < omega_m_percentiles[0]
        omega_m_high = self.params_array[:500, 1] > omega_m_percentiles[1]

        mean_pdf_low = np.mean(self.pdf_sim_linear[:500][omega_m_low], axis=0)
        std_pdf_low = np.std(self.pdf_sim_linear[:500][omega_m_low], axis=0)
        mean_pdf_high = np.mean(self.pdf_sim_linear[:500][omega_m_high], axis=0)
        std_pdf_high = np.std(self.pdf_sim_linear[:500][omega_m_high], axis=0)

        bins = np.arange(len(mean_pdf_low))

        # Plot with error bands
        ax.plot(bins, mean_pdf_low, 'b-', label=f'Ωₘ < {omega_m_percentiles[0]:.3f} (n={omega_m_low.sum()})')
        ax.fill_between(bins, mean_pdf_low - std_pdf_low, mean_pdf_low + std_pdf_low, 
                        alpha=0.2, color='blue')

        ax.plot(bins, mean_pdf_high, 'r-', label=f'Ωₘ > {omega_m_percentiles[1]:.3f} (n={omega_m_high.sum()})')
        ax.fill_between(bins, mean_pdf_high - std_pdf_high, mean_pdf_high + std_pdf_high, 
                        alpha=0.2, color='red')

        # Calculate and display relative difference
        rel_diff = (mean_pdf_high - mean_pdf_low) / (mean_pdf_low + 1e-10)
        ax2 = ax.twinx()
        ax2.plot(bins, rel_diff * 100, 'g--', alpha=0.5, linewidth=1)
        ax2.set_ylabel('Relative diff (%)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        ax.set_xlabel('PDF bin')
        ax.set_ylabel('Mean PDF')
        ax.set_title('Mean PDF: Low vs High Ωₘ')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: PCA analysis
        ax = axes[1, 0]

        # Standardize PDFs for PCA
        scaler = StandardScaler()
        pdf_scaled = scaler.fit_transform(self.pdf_sim_linear[:500])

        # Apply PCA
        pca = PCA(n_components=2)
        pdf_pca = pca.fit_transform(pdf_scaled)

        # Create scatter plot colored by omega_m
        scatter = ax.scatter(pdf_pca[:, 0], pdf_pca[:, 1], 
                            c=self.params_array[:500, 1], 
                            cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Ωₘ')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA of PDFs colored by Ωₘ')
        ax.grid(True, alpha=0.3)

        # Plot 4: Correlation analysis
        ax = axes[1, 1]

        correlations_omega = []
        correlations_sigma = []

        # Calculate correlation for each PDF bin
        for i in range(self.pdf_sim_linear.shape[1]):
            # Skip if variance is too small (constant bin)
            if np.std(self.pdf_sim_linear[:500, i]) > 1e-10:
                corr_omega = np.corrcoef(self.pdf_sim_linear[:500, i], 
                                         self.params_array[:500, 1])[0, 1]
                corr_sigma = np.corrcoef(self.pdf_sim_linear[:500, i], 
                                         self.params_array[:500, 0])[0, 1]
            else:
                corr_omega = 0
                corr_sigma = 0

            correlations_omega.append(corr_omega)
            correlations_sigma.append(corr_sigma)

        x = range(len(correlations_omega))
        ax.plot(x, correlations_omega, 'b-', label='PDF-Ωₘ correlation', linewidth=2)
        ax.plot(x, correlations_sigma, 'r-', label='PDF-σ₈ correlation', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.1, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(-0.1, color='gray', linestyle=':', alpha=0.3)

        ax.set_xlabel('PDF bin index')
        ax.set_ylabel('Correlation coefficient')
        ax.set_title('Parameter-PDF Correlations by Bin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)

        plt.suptitle('PDF Sensitivity to Parameters Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print("\n" + "="*60)
        print("PDF INFORMATION CONTENT ANALYSIS")
        print("="*60)

        print(f"\nParameter ranges in data:")
        print(f"  σ₈:  [{self.params_array[:, 0].min():.3f}, {self.params_array[:, 0].max():.3f}]")
        print(f"  Ωₘ:  [{self.params_array[:, 1].min():.3f}, {self.params_array[:, 1].max():.3f}]")

        print(f"\nCorrelation with PDF bins:")
        print(f"  Maximum |correlation| with Ωₘ:  {np.max(np.abs(correlations_omega)):.3f}")
        print(f"  Mean |correlation| with Ωₘ:     {np.mean(np.abs(correlations_omega)):.3f}")
        print(f"  Maximum |correlation| with σ₈:  {np.max(np.abs(correlations_sigma)):.3f}")
        print(f"  Mean |correlation| with σ₈:     {np.mean(np.abs(correlations_sigma)):.3f}")

        info_ratio = np.max(np.abs(correlations_omega)) / (np.max(np.abs(correlations_sigma)) + 1e-10)
        print(f"  Information ratio (Ωₘ/σ₈):      {info_ratio:.3f}")

        print(f"\nPCA variance explained:")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total (first 2): {sum(pca.explained_variance_ratio_[:2]):.1%}")

        # Check if PCs correlate with parameters
        pc1_omega_corr = np.corrcoef(pdf_pca[:, 0], self.params_array[:500, 1])[0, 1]
        pc1_sigma_corr = np.corrcoef(pdf_pca[:, 0], self.params_array[:500, 0])[0, 1]
        pc2_omega_corr = np.corrcoef(pdf_pca[:, 1], self.params_array[:500, 1])[0, 1]
        pc2_sigma_corr = np.corrcoef(pdf_pca[:, 1], self.params_array[:500, 0])[0, 1]

        print(f"\nPrincipal component correlations:")
        print(f"  PC1 with Ωₘ: {pc1_omega_corr:.3f}")
        print(f"  PC1 with σ₈: {pc1_sigma_corr:.3f}")
        print(f"  PC2 with Ωₘ: {pc2_omega_corr:.3f}")
        print(f"  PC2 with σ₈: {pc2_sigma_corr:.3f}")

        # Diagnosis
        print("\n" + "="*60)
        print("DIAGNOSIS:")

        if info_ratio < 0.3:
            print("❌ PDF contains very little information about Ωₘ")
            print("   The PDF is ~{:.0f}x less sensitive to Ωₘ than to σ₈".format(1/info_ratio))
        elif info_ratio < 0.6:
            print("⚠️  PDF contains limited information about Ωₘ")
        else:
            print("✓  PDF contains reasonable information about Ωₘ")

        if np.max(np.abs(correlations_omega)) < 0.1:
            print("❌ No PDF bins show meaningful correlation with Ωₘ")
            print("   This suggests the PDF is fundamentally insensitive to Ωₘ")

        if abs(pc1_omega_corr) < 0.2 and abs(pc2_omega_corr) < 0.2:
            print("❌ Principal components don't capture Ωₘ variation")
    
        print("\nRECOMMENDATIONS:")
        if info_ratio < 0.3:
            print("1. Switch to power spectrum - it contains spatial information")
            print("2. Use combined analysis (PS + PDF) for complementary constraints")
            print("3. Consider alternative statistics (Minkowski functionals, peaks)")
            print("4. Check if your simulations properly vary Ωₘ effects")

    def test_posterior_recovery(
        self,
        posterior: Any,
        true_params: np.ndarray,
        test_data: np.ndarray,
        n_test_samples: int = 100,
        n_posterior_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Test how well the posterior recovers known parameters
        
        Args:
            posterior: Trained posterior
            true_params: True parameters for test samples
            test_data: Test observations
            n_test_samples: Number of test samples to use
            n_posterior_samples: Samples to draw from posterior
            
        Returns:
            Dictionary with recovery metrics
        """
        metrics = {
            'coverage_68': [],
            'coverage_95': [],
            'rmse': [],
            'bias': [],
            'uncertainty_calibration': []
        }
        
        n_test = min(n_test_samples, len(test_data))
        
        for i in tqdm(range(n_test), desc="Testing posterior recovery"):
            # Get observation
            x_obs = torch.tensor(test_data[i], dtype=torch.float32)
            true_param = true_params[i]
            
            # Sample from posterior
            samples = posterior.sample((n_posterior_samples,), x=x_obs).numpy()
            
            # Calculate statistics
            stats = calculate_statistics(samples, true_param, confidence_levels=[68, 95])
            
            # Check coverage
            metrics['coverage_68'].append(stats['coverage']['68%'])
            metrics['coverage_95'].append(stats['coverage']['95%'])
            
            # Calculate errors
            metrics['rmse'].append(np.mean(stats['rmse']))
            metrics['bias'].append(np.mean(np.abs(stats['bias'])))
            
            # Check uncertainty calibration
            # (std should correlate with actual error)
            actual_error = np.abs(stats['mean'] - true_param)
            predicted_std = stats['std']
            z_score = actual_error / (predicted_std + 1e-10)
            metrics['uncertainty_calibration'].append(np.mean(z_score))
        
        # Aggregate metrics
        final_metrics = {
            'coverage_68_rate': np.mean(metrics['coverage_68']),
            'coverage_95_rate': np.mean(metrics['coverage_95']),
            'mean_rmse': np.mean(metrics['rmse']),
            'mean_bias': np.mean(metrics['bias']),
            'calibration_score': np.mean(metrics['uncertainty_calibration']),
            'calibration_std': np.std(metrics['uncertainty_calibration'])
        }
        
        # Check if well-calibrated
        final_metrics['well_calibrated_68'] = abs(final_metrics['coverage_68_rate'] - 0.68) < 0.05
        final_metrics['well_calibrated_95'] = abs(final_metrics['coverage_95_rate'] - 0.95) < 0.05
        
        return final_metrics
    
    def compare_methods(
        self,
        results_dict: Dict,
        test_params: np.ndarray,
        test_observations: Dict[str, np.ndarray],
        n_test: int = 50
    ) -> pd.DataFrame:
        """
        Compare different inference methods
        
        Returns:
            DataFrame with comparison metrics
        """
        try:
            import pandas as pd
        except ImportError:
            warnings.warn("pandas not installed. Returning dict instead of DataFrame")
            return self._compare_methods_dict(results_dict, test_params, test_observations, n_test)
        
        comparison_data = []
        
        for method_key, result in results_dict.items():
            if "error" in result:
                continue
                
            analysis_method, inference_method = method_key.rsplit('_', 1)
            
            # Get test data for this analysis method
            if analysis_method not in test_observations:
                continue
            
            test_obs = test_observations[analysis_method][:n_test]
            test_params_subset = test_params[:n_test]
            
            # Test recovery
            recovery_metrics = self.test_posterior_recovery(
                result['posterior'],
                test_params_subset,
                test_obs,
                n_test_samples=min(n_test, 50)
            )
            
            # Analyze convergence
            if 'loss_history' in result:
                convergence_metrics = self.analyze_loss_convergence(
                    result['loss_history'],
                    method_key,
                    plot=False
                )
            else:
                convergence_metrics = {}
            
            # Combine metrics
            row = {
                'analysis_method': analysis_method,
                'inference_method': inference_method,
                **recovery_metrics,
                **convergence_metrics
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
        
        
   
  
    
    def run_simulation_based_calibration(
        self,
        inference_engine: InferenceEngine,
        params_array: np.ndarray,
        simulations: np.ndarray,
        n_sbc_runs: int = 100,
        n_posterior_samples: int = 1000
    ) -> Dict:
        """
        Run Simulation-Based Calibration (SBC) test
        
        This tests if the posterior is properly calibrated by checking
        if the rank statistics are uniform
        """
        ranks = []
        
        for i in tqdm(range(n_sbc_runs), desc="Running SBC"):
            # Sample true parameters from prior
            idx = np.random.randint(0, len(params_array))
            true_param = params_array[idx]
            observation = simulations[idx]
            
            # Get posterior samples
            x_obs = torch.tensor(observation, dtype=torch.float32)
            
            # This assumes posterior is already trained
            if not inference_engine.posteriors:
                raise ValueError("No trained posteriors available")
            
            posterior = list(inference_engine.posteriors.values())[0]
            samples = posterior.sample((n_posterior_samples,), x=x_obs).numpy()
            
            # Calculate rank of true parameter
            rank = np.sum(samples < true_param[None, :], axis=0) / n_posterior_samples
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Test uniformity using Kolmogorov-Smirnov test
        sbc_results = {}
        for i, param_name in enumerate(self.param_config.param_names):
            ks_stat, p_value = stats.kstest(ranks[:, i], 'uniform')
            sbc_results[param_name] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'uniform': p_value > 0.05  # Not rejecting uniformity
            }
        
        # Plot SBC results
        self._plot_sbc_results(ranks, sbc_results)
        
        return sbc_results
    
    def _plot_sbc_results(self, ranks: np.ndarray, sbc_results: Dict):
        """Plot SBC rank statistics"""
        n_params = ranks.shape[1]
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, results) in enumerate(sbc_results.items()):
            ax = axes[i]
            
            # Plot histogram
            ax.hist(ranks[:, i], bins=20, density=True, alpha=0.7, 
                   edgecolor='black', label='Rank statistics')
            
            # Plot uniform reference
            ax.axhline(1.0, color='red', linestyle='--', label='Uniform')
            
            # Add test results
            ax.set_title(f'{param_name}\nKS p-value: {results["p_value"]:.3f}')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Simulation-Based Calibration Test', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def test_network_embedding(
        self,
        network: torch.nn.Module,
        test_inputs: np.ndarray,
        n_samples: int = 100
    ) -> Dict:
        """
        Test if neural network embeddings are meaningful
        """
        network.eval()
        
        # Get embeddings
        embeddings = []
        with torch.no_grad():
            for i in range(min(n_samples, len(test_inputs))):
                input_tensor = torch.tensor(test_inputs[i], dtype=torch.float32)
                if input_tensor.dim() == 2:
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                embedding = network(input_tensor)
                embeddings.append(embedding.numpy().flatten())
        
        embeddings = np.array(embeddings)
        
        # Analyze embeddings
        metrics = {
            'embedding_dim': embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
            'effective_dim': self._calculate_effective_dimension(embeddings)
        }
        
        # Check if embeddings are collapsing
        pairwise_distances = np.linalg.norm(
            embeddings[:, None] - embeddings[None, :], axis=2
        )
        np.fill_diagonal(pairwise_distances, np.inf)
        
        metrics['min_distance'] = np.min(pairwise_distances)
        metrics['mean_distance'] = np.mean(pairwise_distances[pairwise_distances != np.inf])
        metrics['collapsed'] = metrics['min_distance'] < 1e-6
        
        return metrics
    
    def _calculate_effective_dimension(self, embeddings: np.ndarray) -> float:
        """Calculate effective dimensionality using PCA"""
        # Center the data
        centered = embeddings - np.mean(embeddings, axis=0)
        
        # Calculate covariance
        cov = np.cov(centered.T)
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        # Calculate effective dimension (participation ratio)
        if len(eigenvalues) > 0:
            normalized = eigenvalues / np.sum(eigenvalues)
            effective_dim = 1 / np.sum(normalized**2)
        else:
            effective_dim = 0
        
        return effective_dim
    
    def generate_diagnostic_report(
        self,
        all_results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive diagnostic report
        """
        report = []
        report.append("="*70)
        report.append("LIMFERENCE CONVERGENCE AND VALIDATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Add timestamp
        from datetime import datetime
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Convergence summary
        report.append("CONVERGENCE SUMMARY")
        report.append("-"*40)
        
        for method, metrics in all_results.get('convergence', {}).items():
            report.append(f"\n{method}:")
            if metrics:
                report.append(f"  Final loss: {metrics.get('final_train_loss', 'N/A'):.4f}")
                report.append(f"  Converged: {metrics.get('converged', 'Unknown')}")
                report.append(f"  Epochs: {metrics.get('n_epochs', 'N/A')}")
                if 'potential_overfitting' in metrics:
                    report.append(f"  Overfitting warning: {metrics['potential_overfitting']}")
        
        # Recovery performance
        report.append("\n\nPOSTERIOR RECOVERY PERFORMANCE")
        report.append("-"*40)
        
        for method, metrics in all_results.get('recovery', {}).items():
            report.append(f"\n{method}:")
            if metrics:
                report.append(f"  68% Coverage: {metrics.get('coverage_68_rate', 0):.2%} (target: 68%)")
                report.append(f"  95% Coverage: {metrics.get('coverage_95_rate', 0):.2%} (target: 95%)")
                report.append(f"  Mean RMSE: {metrics.get('mean_rmse', 'N/A'):.4f}")
                report.append(f"  Well calibrated: {metrics.get('well_calibrated_68', False)}")
        
        # SBC results if available
        if 'sbc' in all_results:
            report.append("\n\nSIMULATION-BASED CALIBRATION")
            report.append("-"*40)
            for param, results in all_results['sbc'].items():
                report.append(f"\n{param}:")
                report.append(f"  KS statistic: {results['ks_statistic']:.4f}")
                report.append(f"  p-value: {results['p_value']:.4f}")
                report.append(f"  Passes uniformity test: {results['uniform']}")
        
        # Network diagnostics if available
        if 'network' in all_results:
            report.append("\n\nNEURAL NETWORK DIAGNOSTICS")
            report.append("-"*40)
            for name, metrics in all_results['network'].items():
                report.append(f"\n{name}:")
                report.append(f"  Embedding dimension: {metrics.get('embedding_dim', 'N/A')}")
                report.append(f"  Effective dimension: {metrics.get('effective_dim', 'N/A'):.2f}")
                report.append(f"  Collapsed: {metrics.get('collapsed', 'Unknown')}")
        
        # Recommendations
        report.append("\n\nRECOMMENDATIONS")
        report.append("-"*40)
        recommendations = self._generate_recommendations(all_results)
        for rec in recommendations:
            report.append(f"• {rec}")
        
        report_text = "\n".join(report)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check convergence
        for method, metrics in results.get('convergence', {}).items():
            if not metrics.get('converged', True):
                recommendations.append(f"Consider training {method} for more epochs")
            if metrics.get('potential_overfitting', False):
                recommendations.append(f"Consider regularization for {method}")
        
        # Check calibration
        for method, metrics in results.get('recovery', {}).items():
            if not metrics.get('well_calibrated_68', True):
                recommendations.append(f"Posterior calibration issues in {method}")
        
        # Check SBC
        if 'sbc' in results:
            failed_params = [p for p, r in results['sbc'].items() if not r['uniform']]
            if failed_params:
                recommendations.append(f"SBC test failed for: {', '.join(failed_params)}")
        
        if not recommendations:
            recommendations.append("All tests passed - inference appears well-calibrated")
        
        return recommendations


def run_full_diagnostic(
    lim_instance,
    test_params: np.ndarray,
    test_observations: Dict[str, np.ndarray],
    save_dir: Optional[str] = None
) -> Dict:
    """
    Run complete diagnostic suite on LIMference results
    
    Args:
        lim_instance: Trained LIMference instance
        test_params: Test parameter set
        test_observations: Test observations for each analysis method
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all diagnostic results
    """
    analyzer = ConvergenceAnalyzer(lim_instance.config, lim_instance.param_config)
    
    all_results = {
        'convergence': {},
        'recovery': {},
        'network': {}
    }
    
    # Analyze each method
    for method_key, result in lim_instance.results.items():
        if "error" not in result:
            # Convergence analysis
            if 'loss_history' in result:
                all_results['convergence'][method_key] = analyzer.analyze_loss_convergence(
                    result['loss_history'],
                    method_key,
                    plot=True,
                    save_path=f"{save_dir}/{method_key}_convergence.pdf" if save_dir else None
                )
            
            # Recovery analysis
            analysis_method = method_key.rsplit('_', 1)[0]
            if analysis_method in test_observations:
                all_results['recovery'][method_key] = analyzer.test_posterior_recovery(
                    result['posterior'],
                    test_params[:50],
                    test_observations[analysis_method][:50],
                    n_test_samples=50
                )
    
    # Test networks if available
    if lim_instance.is_initialized:
        for name, network in lim_instance.networks.items():
            if "field_level" in lim_instance.simulations:
                all_results['network'][name] = analyzer.test_network_embedding(
                    network,
                    lim_instance.simulations["field_level"][:100]
                )
    
    # Generate report
    report = analyzer.generate_diagnostic_report(
        all_results,
        save_path=f"{save_dir}/diagnostic_report.txt" if save_dir else None
    )
    
    print(report)
    
    return all_results
