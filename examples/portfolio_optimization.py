#!/usr/bin/env python
"""Portfolio optimization example using proximal_lq.

This example demonstrates how to use the proximal_lq library to solve
a classic Markowitz-style portfolio optimization problem where we want
to find portfolio weights that minimize tracking error to a target portfolio.

The problem formulation:
    minimize    0.5 ||Sigma @ w - target||^2
    subject to  w >= 0
                sum(w) = 1

where:
    - Sigma is the covariance matrix of asset returns
    - w is the vector of portfolio weights
    - target is a target portfolio characteristic vector

Run with: python examples/portfolio_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path for running example directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proximal_lq import prox_gradient


def generate_sample_covariance(n_assets: int, seed: int = 42) -> np.ndarray:
    """Generate a random positive definite covariance matrix.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        Covariance matrix of shape (n_assets, n_assets).

    """
    rng = np.random.default_rng(seed)
    # Generate a random matrix and create a covariance matrix
    a = rng.standard_normal((n_assets, n_assets))
    return a @ a.T / n_assets


def main() -> None:
    """Run portfolio optimization example."""
    print("Portfolio Optimization with Simplex Constraints")
    print("=" * 50)

    # Problem setup
    n_assets = 10
    print(f"\nNumber of assets: {n_assets}")

    # Generate sample covariance matrix
    covariance = generate_sample_covariance(n_assets)
    print(f"Covariance matrix condition number: {np.linalg.cond(covariance):.2f}")

    # Target: equal-weighted portfolio characteristics
    target = np.ones(n_assets) / n_assets

    # Solve the optimization problem
    print("\nSolving optimization problem...")
    weights = prox_gradient(covariance, target, eps_rel=1e-8, max_iter=2000)

    # Display results
    print("\nOptimal Portfolio Weights:")
    print("-" * 30)
    for i, w in enumerate(weights):
        if w > 1e-4:  # Only show non-zero weights
            print(f"  Asset {i + 1:2d}: {w:8.4f} ({w * 100:5.2f}%)")

    print("\nPortfolio Properties:")
    print(f"  Sum of weights: {weights.sum():.6f}")
    print(f"  Non-zero positions: {np.sum(weights > 1e-4)}")
    print(f"  Min weight: {weights.min():.6f}")
    print(f"  Max weight: {weights.max():.4f}")

    # Verify constraints
    print("\nConstraint Verification:")
    print(f"  Non-negativity satisfied: {np.all(weights >= -1e-10)}")
    print(f"  Sum-to-one satisfied: {abs(weights.sum() - 1.0) < 1e-6}")

    # Compute objective value
    residual = covariance @ weights - target
    objective = 0.5 * np.linalg.norm(residual) ** 2
    print(f"\nObjective value: {objective:.6f}")


if __name__ == "__main__":
    main()
