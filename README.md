# [proximal](/book)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

Proximal is a Python library for solving constrained linear least squares
problems of the form:

```
minimize    0.5 ||Ax - b||^2
subject to  x >= 0, sum(x) = 1
```

This type of optimization problem appears in many applications including:

- **Portfolio optimization**: Finding optimal asset allocations
- **Signal processing**: Non-negative sparse coding
- **Machine learning**: Constrained regression, mixture models
- **Statistics**: Probability distribution estimation

## Mathematical Background

The library implements an efficient **proximal gradient descent** algorithm
that projects onto the probability simplex at each iteration.

### Key Functions

**`proj_simplex(vec, rad=1)`**: Projects a vector onto the simplex using
the algorithm from Duchi et al. (2008):

```
proj(v) = argmin_{x in Delta} ||x - v||^2
```

where Delta = {x : x >= 0, sum(x) = rad}

**`prox_gradient(mat, vec)`**: Solves the full optimization problem using
iterative projection with convergence guarantees.

## Installation

```bash
pip install proximal-lq
```

## Usage

```python
import numpy as np
from proximal_lq import prox_gradient, proj_simplex

# Create a matrix and vector for the optimization problem
mat = np.array([[1.0, 0.5], [0.5, 1.0]])  # Example covariance matrix
vec = np.ones(2)  # Target vector

# Solve the optimization problem
# Find x that minimizes 0.5 ||mat @ x - vec||^2 subject to x >= 0, sum(x) = 1
result = prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000)

print(np.round(result, 4))  # Optimal weights that satisfy the constraints
```

```result
[0.5 0.5]
```

## Features

- **Fast simplex projection** using the algorithm from
  [Duchi et al. (2008)](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf)
- **Proximal gradient descent** solver with configurable convergence criteria
- **Pure NumPy implementation** for high performance
- **Type annotations** for better IDE support
- **Simple API** with minimal dependencies

## Getting Started

### Set Up Environment

```bash
make install
```

## Development Commands

```bash
make test    # Run test suite
make fmt     # Format and lint code
make marimo  # Start Marimo notebooks
make book    # Build documentation
```

## Documentation

- [API Documentation](/book) - Generated API reference
- [Development Guide](docs/DEVELOPMENT.md) - Algorithm details and implementation notes
- [Examples](examples/) - Usage examples including portfolio optimization

## Contributing

Contributions are most welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## References

Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
"Efficient Projections onto the l1-Ball for Learning in High Dimensions."
*Proceedings of ICML*.

## License

MIT License - see [LICENSE](LICENSE) for details.
