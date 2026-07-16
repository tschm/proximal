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

## API Reference

The public API is exported directly from `proximal_lq`:

```python +RHIZA_SKIP
from proximal_lq import proj_simplex, prox_gradient
```

### `proj_simplex(vec, rad=1.0)`

```python +RHIZA_SKIP
proj_simplex(vec: NDArray[np.floating], rad: float = 1.0) -> NDArray[np.floating]
```

Euclidean projection of `vec` onto the probability simplex
`{x : x >= 0, sum(x) = rad}`, using the algorithm of Duchi et al. (2008).

- **`vec`** – input vector to project (must be non-empty).
- **`rad`** – radius of the simplex; the result sums to this value (default `1.0`).
- **Raises** `ValueError` if `vec` is empty.

```python
import numpy as np
from proximal_lq import proj_simplex

weights = proj_simplex(np.array([0.3, 0.9, -0.2, 0.5]))
print(np.round(weights, 4))
```

```result
[0.0667 0.6667 0.     0.2667]
```

### `prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000, seed=None)`

```python +RHIZA_SKIP
prox_gradient(
    mat: NDArray[np.floating],
    vec: NDArray[np.floating],
    eps_rel: float = 1e-6,
    max_iter: int = 1000,
    seed: int | None = None,
) -> NDArray[np.floating]
```

Solve `minimize 0.5 ||mat @ x - vec||^2` subject to `x >= 0, sum(x) = 1` via
proximal gradient descent with simplex projection.

- **`mat`** – matrix of shape `(n_samples, n_features)`.
- **`vec`** – vector of shape `(n_samples,)`; its length must equal `mat.shape[0]`.
- **`eps_rel`** – relative stopping tolerance (default `1e-6`).
- **`max_iter`** – maximum number of iterations (default `1000`).
- **`seed`** – optional seed for the random initialisation; pass an integer for
  reproducible results (the problem is convex, so the optimum is independent of
  the seed).
- **Returns** the solution vector of shape `(n_features,)`.
- **Raises** `ValueError` if `mat` is not 2-D, `vec` is not 1-D, either input is
  empty, or `vec.shape[0]` does not match `mat.shape[0]`.

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
