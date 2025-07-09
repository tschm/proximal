# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo==0.13.15",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "pandas==2.3.1",
# ]
# ///
"""Demonstration of the Proximal Gradient Method for Simplex-Constrained Optimization."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from proximal import proj_simplex, prox_gradient


@app.cell
def _():
    mo.md(
        r"""
    # Proximal Gradient Method for Simplex-Constrained Optimization

    This notebook demonstrates the use of the `proximal` package, which provides efficient solvers for optimization problems of the form:

    $$\min_x \frac{1}{2} \|Ax - b\|^2 \quad \text{subject to} \quad x \geq 0, \sum_i x_i = 1$$

    These types of problems appear in various applications including portfolio optimization, machine learning, and signal processing.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## 1. Projection onto the Probability Simplex

    The first key function in the package is `proj_simplex`, which projects a vector onto the probability simplex.
    The probability simplex is the set of non-negative vectors that sum to 1:

    $$\Delta = \{x \in \mathbb{R}^n : x \geq 0, \sum_i x_i = 1\}$$

    Let's demonstrate this function with a simple example:
    """
    )
    return


@app.cell
def _():
    # Create a random vector
    rng = np.random.default_rng(42)
    x = rng.normal(size=5)

    # Project it onto the simplex
    x_proj = proj_simplex(x)

    # Display the results
    pd.DataFrame({"Original Vector": x, "Projected Vector": x_proj})

    mo.md(f"""
    Original vector sum: {x.sum():.4f}

    Projected vector sum: {x_proj.sum():.4f}
    """)

    return x, x_proj


@app.cell
def _(x, x_proj):
    # Visualize the projection
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x_pos = np.arange(len(x))

    ax.bar(x_pos - bar_width / 2, x, bar_width, label="Original Vector")
    ax.bar(x_pos + bar_width / 2, x_proj, bar_width, label="Projected Vector")

    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"x{i + 1}" for i in range(len(x))])
    ax.set_ylabel("Value")
    ax.set_title("Projection onto the Probability Simplex")
    ax.legend()

    plt.tight_layout()
    return


@app.cell
def _():
    mo.md(
        r"""
    Notice how the projection:

    1. Makes all negative values zero
    2. Scales the remaining positive values so they sum to 1

    ## 2. Proximal Gradient Method

    The main function in the package is `prox_gradient`, which solves optimization problems of the form:

    $$\min_x \frac{1}{2} \|Ax - b\|^2 \quad \text{subject to} \quad x \geq 0, \sum_i x_i = 1$$

    Let's first demonstrate this with a simple example:
    """
    )
    return


@app.cell
def _():
    # Create a simple problem
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([1, 2, 3])

    # Solve using proximal gradient
    x_sol = prox_gradient(a, b)

    # Compute the objective value
    obj_val = 0.5 * np.linalg.norm(a @ x_sol - b) ** 2

    mo.md(f"""
    Solution vector: {x_sol}

    Sum of solution: {x_sol.sum():.4f}

    Objective value: {obj_val:.4f}
    """)

    return


@app.cell
def _():
    mo.md(
        r"""
    ## 3. Portfolio Optimization Example

    A common application of this method is in portfolio optimization, where we want to find the optimal weights for assets in a portfolio.

    Let's use a covariance matrix similar to the one used in the tests:
    """
    )
    return


@app.cell
def _():
    # Try to load the test data if available
    test_data_path = None

    # First try to find the repository root
    repo_root = None
    current_dir = Path.cwd()

    # Check if we're already at the repo root
    if (current_dir / "src" / "proximal").exists():
        repo_root = current_dir
    # Check if we're in the book directory
    elif (current_dir.parent / "src" / "proximal").exists():
        repo_root = current_dir.parent
    # Check if we're in the book/marimo directory
    elif (current_dir.parent.parent / "src" / "proximal").exists():
        repo_root = current_dir.parent.parent

    if repo_root:
        test_data_path = repo_root / "src" / "tests" / "resources" / "CLA_Data.csv"
        if not test_data_path.exists():
            test_data_path = None

    # Fallback to relative paths if repo root detection failed
    if test_data_path is None:
        possible_paths = [
            Path("src/tests/resources/CLA_Data.csv"),  # If running from project root
            Path("../src/tests/resources/CLA_Data.csv"),  # If running from book directory
            Path("../../src/tests/resources/CLA_Data.csv"),  # If running from book/marimo directory
        ]

        for path in possible_paths:
            if path.exists():
                test_data_path = path
                break

    if test_data_path:
        # Load the covariance matrix from the test data
        data = np.genfromtxt(test_data_path, delimiter=",", skip_header=1)
        covar = data[3:]  # Skip the first 3 rows as in the test
        mo.md(f"Loaded covariance matrix from {test_data_path}")
    else:
        # Create a synthetic covariance matrix if test data not available
        rng = np.random.default_rng(42)
        n_assets = 10
        # Create a random correlation matrix
        a = rng.normal(size=(n_assets, n_assets))
        covar = a.T @ a
        # Normalize to get correlations
        d = np.diag(1.0 / np.sqrt(np.diag(covar)))
        covar = d @ covar @ d
        # Add some variance
        variances = rng.uniform(0.1, 0.5, size=n_assets)
        d_var = np.diag(np.sqrt(variances))
        covar = d_var @ covar @ d_var
        mo.md("Created synthetic covariance matrix (test data not found)")

    # Display the covariance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(covar, cmap="viridis")
    plt.colorbar(label="Covariance")
    plt.title("Asset Covariance Matrix")
    plt.tight_layout()

    return (covar,)


@app.cell
def _(covar):
    # Solve the portfolio optimization problem
    # We'll minimize the portfolio variance (risk) subject to the simplex constraints

    # For portfolio optimization, we typically want to minimize x^T Σ x
    # This is equivalent to minimizing ||Σ^(1/2) x - 0||^2
    # We can compute Σ^(1/2) using the Cholesky decomposition

    # Ensure the covariance matrix is positive definite
    covar_pd = covar @ covar.T + 1e-6 * np.eye(covar.shape[0])

    # Compute the Cholesky decomposition
    np.linalg.cholesky(covar_pd)

    # Solve using proximal gradient
    # We use a vector of ones as the target, which corresponds to an equal-weighted portfolio
    target = np.ones(covar.shape[0])
    weights = prox_gradient(covar, target)

    # Compute the portfolio variance
    portfolio_variance = weights.T @ covar @ weights

    # Display the results
    asset_names = [f"Asset {i + 1}" for i in range(len(weights))]
    weights_df = pd.DataFrame({"Asset": asset_names, "Weight": weights})

    # Sort by weight in descending order
    weights_df = weights_df.sort_values("Weight", ascending=False)

    mo.md(f"""
    ### Optimal Portfolio Weights

    Portfolio Variance: {portfolio_variance:.6f}
    """)

    return (weights_df,)


@app.cell
def _(weights_df):
    # Visualize the portfolio weights
    plt.figure(figsize=(12, 6))

    # Only show assets with non-zero weights
    non_zero_weights = weights_df[weights_df["Weight"] > 1e-6]

    plt.bar(non_zero_weights["Asset"], non_zero_weights["Weight"])
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.ylabel("Weight")
    plt.title("Optimal Portfolio Weights")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return


@app.cell
def _():
    mo.md(
        r"""
    ## Conclusion

    The `proximal` package provides efficient solvers for optimization problems with simplex constraints.
    These types of problems appear in various applications, with portfolio optimization being a prominent example.

    The key functions are:

    1. `proj_simplex`: Projects a vector onto the probability simplex
    2. `prox_gradient`: Solves optimization problems of the form $\min_x \frac{1}{2} \|Ax - b\|^2$ subject to simplex constraints

    These functions are implemented efficiently and can handle large-scale problems.
    """
    )
    return


if __name__ == "__main__":
    app.run()
