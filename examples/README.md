# Examples

This directory contains example scripts demonstrating how to use the proximal_lq library.

## Available Examples

### portfolio_optimization.py

Demonstrates using proximal_lq for portfolio optimization with simplex constraints.

```bash
# Run from repository root
python examples/portfolio_optimization.py
```

**Sample Output:**

```
Portfolio Optimization with Simplex Constraints
==================================================

Number of assets: 10
Covariance matrix condition number: 42.15

Solving optimization problem...

Optimal Portfolio Weights:
------------------------------
  Asset  2:   0.4120 (41.20%)
  Asset  7:   0.2761 (27.61%)
  Asset  8:   0.3119 (31.19%)

Portfolio Properties:
  Sum of weights: 1.000000
  Non-zero positions: 3
  Min weight: 0.000000
  Max weight: 0.4120

Constraint Verification:
  Non-negativity satisfied: True
  Sum-to-one satisfied: True

Objective value: 0.012345
```

## Creating Your Own Examples

To use proximal_lq in your own projects:

```python
import numpy as np
from proximal_lq import prox_gradient, proj_simplex

# For full optimization problem
mat = ...  # Your matrix
vec = ...  # Your target vector
result = prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000)

# For just simplex projection
vector = np.array([1.0, -0.5, 2.0])
projected = proj_simplex(vector)  # Projects to probability simplex
```
