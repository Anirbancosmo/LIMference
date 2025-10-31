"""
Setup script for LIMference package
This file configures how the package is built, distributed, and installed
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = """
    LIMference: Line Intensity Mapping Inference Package
    
    A comprehensive package for comparing power spectrum, PDF, and field-level 
    inference methods on Line Intensity Mapping data using simulation-based inference.
    """

# Read version from package
version = {}
with open("limference/__init__.py") as fp:
    for line in fp:
        if line.startswith("__version__"):
            exec(line, version)
            break
    else:
        version["__version__"] = "0.1.0"

setup(
    # Basic package information
    name="limference",
    version=version.get("__version__", "0.1.0"),
    author="Anirban Roy",
    author_email="anirbanroy.personal@gmail.com",
    url="https://github.com/anirbancosmo/limference",
    license="MIT",
    
    # Description
    description="Line Intensity Mapping Inference Package - Comparing power spectrum, PDF, and field-level methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    
    # Include non-Python files specified in MANIFEST.in
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classifiers (for PyPI)
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Required dependencies (installed automatically)
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "torch>=1.9.0",
        "sbi>=0.20.0",
        "tqdm>=4.62.0",
    ],
    
    # Optional dependencies (installed with pip install limference[extra])
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "jupyter>=1.0",
            "ipython>=7.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        
        # Visualization extras
        "viz": [
            "pygtc",  # For corner plots
            "seaborn>=0.11.0",  # Enhanced plotting
            "corner>=2.0",  # Alternative corner plots
            "plotly>=5.0",  # Interactive plots
        ],
        
        # LIM-specific dependencies
        "lim": [
            # "limpy",  # Uncomment when available
        ],
        
        # All extras
        "all": [
            "pygtc",
            "seaborn>=0.11.0",
            "corner>=2.0",
            "plotly>=5.0",
            "pytest>=6.0",
            "jupyter>=1.0",
        ],
    },
    
    # Entry points (command-line scripts)
    entry_points={
        "console_scripts": [
            # Creates a command-line tool called 'limference'
            "limference=limference.cli:main",
            "limference-quick=limference.main:quick_inference_cli",
        ],
    },
    
    # Package data (data files to include)
    package_data={
        "limference": [
            "configs/*.json",
            "configs/*.yaml",
            "data/*.npz",
        ],
    },
    
    # Minimum versions for key dependencies
    dependency_links=[],
    
    # Keywords for package discovery
    keywords=[
        "cosmology",
        "astrophysics",
        "line intensity mapping",
        "simulation-based inference",
        "machine learning",
        "bayesian inference",
        "power spectrum",
        "neural networks",
    ],
    
    # Project URLs (appear on PyPI)
    project_urls={
        "Documentation": "https://limference.readthedocs.io",
        "Source": "https://github.com/anirbancosmo/limference",
        "Issues": "https://github.com/anirbancosmo/limference/issues",
        "Paper": "https://arxiv.org/abs/XXXXXX",
    },
)
