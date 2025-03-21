# Bayesian NPZ Model Parameter Recovery

This repository presents a feasibility study using a Bayesian framework to recover the parameters of a Nitrogen–Phytoplankton–Zooplankton (NPZ) model from synthetic data. Unlike traditional methods—where coefficients are established in the lab or field and then fixed—this approach uses informative priors and MCMC sampling to yield full posterior distributions, quantifying uncertainty in parameter estimates.

## Overview

- **NPZ Model:**  
  The model describes nutrient uptake (using Monod kinetics), phytoplankton growth, and zooplankton grazing (using a Hill function).  
- **Synthetic Data Generation:**  
  Synthetic observations are generated by solving the NPZ model using literature-informed parameter values set as true parameter values, with added noise to mimic observational uncertainty.
- **Bayesian Inference:**  
  I use PyMC (v5+) and ArviZ to perform MCMC sampling to recover the true parameter values from synthetic data and quantify uncertainty.  


## Repository Structure

- **notebooks/**  
  Jupyter Notebooks demonstrating synthetic data generation, Bayesian inference, and figure creation.


## Requirements

- Python 3.12
- PyMC (v5+)
- ArviZ
- Plotly
- Matplotlib
- SciPy
- NumPy

You can install the required packages with:

```bash
pip install pymc arviz matplotlib scipy numpy
```
or 
```bash
conda install -c conda-forge pymc arviz marplotlib scipy numpy
