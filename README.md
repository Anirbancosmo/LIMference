# LIMference

**Simulation-Based Inference for Line Intensity Mapping**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/Anirbancosmo/LIMference)](https://github.com/Anirbancosmo/LIMference/issues)

LIMference is a comprehensive Python package for performing **Simulation-Based Inference (SBI)** on **Line Intensity Mapping (LIM)** data. It enables robust cosmological and astrophysical parameter inference by comparing multiple analysis methods (power spectrum, PDF, field-level) and inference techniques (NPE, SNPE, NLE, NRE).

---

## Key Features

### Inference Methods
-  **Neural Posterior Estimation (NPE)**: Direct posterior learning
-  **Sequential NPE (SNPE)**: Active learning with proposal refinement
-  **Neural Likelihood Estimation (NLE)**: Likelihood-based inference
-  **Neural Ratio Estimation (NRE)**: Density ratio estimation

### Analysis Methods
-  **Power Spectrum**: Fourier-space summary statistics
-  **PDF**: Probability distribution function of intensity
- ️**Field-Level**: Direct inference from 3D intensity maps using CNNs

### Advanced Diagnostics
-  **Coverage Calibration**: Ensures posterior uncertainties are reliable
-  **Simulation-Based Calibration (SBC)**: Tests for correct posterior coverage
-  **LC2ST**: Local Classifier Two-Sample Test for posterior quality
-  **MCMC Diagnostics**: Convergence checks with ArviZ
-  **Posterior Predictive Checks**: Validates model performance
-  **Active Subspace Analysis**: Identifies dominant parameter directions
-  **Conditional Distribution Analysis**: Parameter correlations and dependencies

### Optimization & Performance
-  **GPU Acceleration**: Full PyTorch GPU support (10-30× speedup)
-  **Hyperparameter Optimization**: Automated tuning with Optuna
- ️**HPC Support**: Ready for clusters like NYU Greene
-  **Batch Processing**: Efficient handling of large simulation suites

---

## Installation

### Basic Installation
```bash
pip install git+https://github.com/Anirbancosmo/LIMference.git
```

### Development Installation
```bash
git clone https://github.com/Anirbancosmo/LIMference.git
cd LIMference
pip install -e .
```

### Dependencies

**Core:**
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- sbi ≥ 0.21.0
- NumPy, SciPy, Matplotlib

**Optional:**
- optuna ≥ 3.0.0 (hyperparameter optimization)
- arviz ≥ 0.12.0 (MCMC diagnostics)
- corner, pygtc (corner plots)
- plotly, seaborn (interactive visualizations)

---

##  Quick Start

### 1. Basic Inference Pipeline
```python
from limference import SBIConvergenceTester

# Initialize convergence tester
tester = SBIConvergenceTester(
    param_names=["sigma8", "omega_m", "a_off", "b_off"],
    inference_methods=["NPE"],
    analysis_methods=["pdf"],
    n_train=2500,
    n_test=500,
    n_calibration=50,
    test_obs_idx=0
)

# Load simulation data
SEED = 51
SIM_ROOT = "/path/to/simulation/data"
OUTPUT_ROOT = "/path/to/output"

tester.load_data(SIM_ROOT, SEED)

# Configure training
training_kwargs = {
    "max_num_epochs": 1000,
    "stop_after_epochs": 20,
    "training_batch_size": 50,
    "learning_rate": 5e-4,
    "validation_fraction": 0.2,
}

# Setup and train
tester.setup_limference(OUTPUT_ROOT, training_kwargs=training_kwargs)
tester.train_models()

# Run diagnostics
tester.calibrate_posteriors()
tester.test_recovery()
tester.plot_results()
tester.generate_report()
```

### 2. Comprehensive Diagnostic Suite
```python
# Coverage calibration with temperature scaling
tester.calibrate_posteriors()
tester.plot_conditional_analysis()

# Simulation-Based Calibration
tester.plot_sbc_ranks(n_sbc_runs=100, n_posterior_samples=1000)

# Posterior Predictive Check
ppc_results = tester.run_posterior_predictive_check(n_posterior_samples=5000)
tester.plot_ppc()

# LC2ST Diagnostic
lc2st_results = tester.run_lc2st_diagnostic(
    n_cal=250,
    n_eval=10000,
    n_trials=100,
    alpha=0.05
)
tester.plot_lc2st_diagnostics()

# MCMC Diagnostics
inference_data = tester.run_mcmc_diagnostics(
    n_samples=5000,
    n_chains=4,
    warmup_steps=1000
)

# Active Subspace Analysis
active_results = tester.run_active_subspace_analysis(n_samples=5000)
tester.plot_active_subspace()

# Conditional Distribution Analysis
cond_results = tester.analyze_conditional_distributions(n_conditions=5)
```

### 3. Parameter Effects Visualization
```python
# Visualize how observables change with each parameter
for param in ["sigma8", "omega_m", "a_off", "b_off"]:
    tester.plot_parameter_effects(param_to_vary=param, n_samples=5)
```

---

##  Hyperparameter Optimization

LIMference includes integrated **Optuna** support for automatic hyperparameter tuning:
```python
import optuna
from limference.optimization import optimize_hyperparameters

# Run optimization to minimize calibration error
study = optimize_hyperparameters(
    sim_root="/path/to/simulations",
    output_root="/path/to/output",
    n_trials=30,
    n_train=2500,
    n_test=500,
    n_calibration=50,
    param_names=["sigma8", "omega_m", "a_off", "b_off"],
    timeout=7200  # 2 hours
)

# Get best hyperparameters
print("Best calibration error:", study.best_value)
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Visualize optimization
import optuna.visualization as vis
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
```

### Hyperparameters Optimized:
- Learning rate
- Batch size
- Number of epochs
- Early stopping patience
- Validation fraction


##  Related Projects

- **[LIMpy](https://github.com/Anirbancosmo/LIMpy)** - Line Intensity Mapping simulations

**Related Publications:**
- Roy et al. (2025) - "Cosmological Parameter Constraints from Line Intensity Mapping with Simulation-Based Inference" *(in prep)*

##  Acknowledgments

- Built on the excellent [`sbi`](https://github.com/mackelab/sbi) package by Mackelab
- [`torch`](https://pytorch.org/) for neural network implementations
- [`optuna`](https://optuna.org/) for hyperparameter optimization
- [`arviz`](https://arviz-devs.github.io/arviz/) for MCMC diagnostics


## Contact

**Anirban Roy**
-  Email: anirbanroy.personal@gmail.com
-  Affiliation: New York University 
-  GitHub: [@Anirbancosmo](https://github.com/Anirbancosmo)
-  Website: https://anirbanroy.in
