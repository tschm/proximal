# Development Guide

This document explains the mathematical background and implementation details of the proximal-lq library.

## Problem Formulation

We solve the constrained quadratic optimization problem:

```
minimize    0.5 ||A x - b||_2^2
subject to  x >= 0
            sum(x) = 1
```

where:

- `A` is an (m x n) matrix
- `b` is an m-dimensional vector
- `x` is the n-dimensional decision variable
- The constraint set is the probability simplex: `Delta = {x : x >= 0, sum(x) = 1}`

## Algorithm: Proximal Gradient Descent

### Overview

We use the proximal gradient method (also known as forward-backward splitting):

```
x^{k+1} = prox_{tau * g}(x^k - tau * grad f(x^k))
```

where:

- `f(x) = 0.5 ||Ax - b||^2` is the smooth objective (gradient Lipschitz)
- `g(x) = indicator_Delta(x)` is the indicator function of the simplex
- `prox_g(v) = proj_Delta(v)` is the projection onto the simplex
- `tau = 1 / L` is the step size, where `L = ||A^T A||_2` is the Lipschitz constant

### Gradient Computation

The gradient of f is:

```
grad f(x) = A^T (Ax - b) = A^T A x - A^T b
```

We precompute:

- `sym_mat = A^T A` (the Gram matrix)
- `out_prod = A^T b`
- `L = ||sym_mat||_2` (spectral norm)

### Convergence

The algorithm converges at rate O(1/k) for the objective value:

```
f(x^k) - f(x*) <= ||x^0 - x*||^2 / (2 * tau * k)
```

We use a relative error criterion:

```
||x^{k+1} - x^k||_2 < eps_rel
```

## Simplex Projection

### Problem

```
minimize    0.5 ||x - v||_2^2
subject to  x >= 0
            sum(x) = s
```

### Algorithm (Duchi et al., 2008)

1. Sort v in descending order: mu_1 >= mu_2 >= ... >= mu_n
2. Compute cumulative means: theta_j = (sum_{i=1}^j mu_i - s) / j
3. Find rho = max{j : mu_j > theta_j}
4. Set theta = theta_rho
5. Return x_i = max(v_i - theta, 0)

### Complexity

- Time: O(n log n) due to sorting
- Space: O(n) for the sorted array

### Reference

Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
"Efficient Projections onto the l1-Ball for Learning in High Dimensions."
Proceedings of the 25th International Conference on Machine Learning (ICML).

## Implementation Notes

### Numerical Stability

1. **Near-zero Lipschitz constant**: When `||A^T A|| ~ 0`, we use step size 1.0 to avoid division issues.

2. **Random initialization**: The algorithm uses random starting points, which means results may vary slightly between runs but converge to similar objective values.

### Performance Considerations

- The algorithm is well-suited for problems where the simplex projection is cheap (O(n log n)) compared to gradient computation (O(mn) for dense matrices).
- For very large problems, consider using sparse matrix representations if applicable.

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=proximal_lq tests/

# Run specific test class
pytest tests/test_proximal/test_proximal.py::TestProjSimplex -v
```

## Code Style

We follow:

- Google-style docstrings
- ruff for linting and formatting
- Type annotations (numpy.typing for arrays)

```bash
# Format code
make fmt

# Check linting
ruff check src/
```
