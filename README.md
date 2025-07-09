# [proximal](/book)

[![PyPI version](https://badge.fury.io/py/proximal.svg)](https://badge.fury.io/py/proximal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Created with Cradle](https://img.shields.io/badge/Created%20with-Cradle-blue?style=flat-square)](https://github.com/tschm/package)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/proximal)

## Overview

Proximal is a Python library for solving constrained linear least squares
problems of the form:

```text
minimize 0.5 ||mat @ x - vec||^2
subject to x >= 0, sum(x) = 1
```

This type of optimization problem appears in many applications including

- Portfolio optimization
- Signal processing
- Machine learning
- Statistics

The library implements an efficient proximal gradient descent algorithm
that projects onto the probability simplex.

## Features

- Fast implementation of projection onto the probability simplex
- Proximal gradient descent solver with configurable convergence criteria
- Pure NumPy implementation for high performance
- Simple API with minimal dependencies

## Usage

```python
import numpy as np
from proximal import prox_gradient

# Create a matrix and vector for the optimization problem
mat = np.array([[1.0, 0.5], [0.5, 1.0]])  # Example covariance matrix
vec = np.ones(2)  # Target vector

# Solve the optimization problem
# Find x that minimizes 0.5 ||mat @ x - vec||^2 subject to x >= 0, sum(x) = 1
result = prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000)

print(result)  # Optimal weights that satisfy the constraints
```

For portfolio optimization, you might use a covariance matrix of asset returns:

```python
# Load a covariance matrix of asset returns
covar = np.genfromtxt("returns_covariance.csv", delimiter=",")

# Find the optimal portfolio weights
weights = prox_gradient(covar, np.ones(covar.shape[0]))

# weights will be non-negative and sum to 1
```

## Getting Started

### **Set Up Environment**

```bash
make install
```

This installs/updates [uv](https://github.com/astral-sh/uv),
creates your virtual environment and installs dependencies.

For adding or removing packages:

```bash
uv add/remove requests  # for main dependencies
uv add/remove requests --dev  # for dev dependencies
```

### **Configure Pre-commit Hooks**

```bash
make fmt
```

Installs hooks to maintain code quality and formatting.

### **Update Project Info**

- Edit `pyproject.toml` to update authors and email addresses
- Configure GitHub Pages (branch: gh-pages) in repository settings

## Development Commands

```bash
make tests   # Run test suite
make marimo  # Start Marimo notebooks
```

## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
