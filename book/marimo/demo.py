# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo==0.13.15",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "pandas==2.3.1",
#     "proximal-lq"
# ]
# ///
"""Demonstration of the Proximal Gradient Method for Simplex-Constrained Optimization."""

import marimo

__generated_with = "0.14.10"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from proximal_lq import proj_simplex, prox_gradient


@app.cell
def _():
    mo.md(
        r"""
    # Proximal Gradient Method for Simplex-Constrained Optimization

    This notebook demonstrates the use of the `proximal_lq` package, which provides
    efficient solvers for optimization problems of the form:

    $$\min_x \frac{1}{2} \|Ax - b\|^2 \quad \text{subject to} \quad x \geq 0, \sum_i x_i = 1$$

    These types of problems appear in various applications including portfolio optimization,
    machine learning, and signal processing.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## 1. Projection onto the Probability Simplex

    The first key function in the package is `proj_simplex`, which projects a
    vector onto the probability simplex.
    The probability simplex is the set of non-negative vectors that sum to 1:

    $$\Delta = \{x \in \mathbb{R}^n : x \geq 0, \sum_i x_i = 1\}$$

    Let's demonstrate this function with a simple example:
    """
    )
    return


@app.cell
def _():
    # Create a random vector
    _rng = np.random.default_rng(42)
    x = _rng.normal(size=5)

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

    The main function in the package is `prox_gradient`, which solves optimization
    problems of the form:

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
    ## Conclusion

    The `proximal_lq` package provides efficient solvers for optimization problems with
    simplex constraints.
    These types of problems appear in various applications, with portfolio
    optimization being a prominent example.

    The key functions are:

    1. `proj_simplex`: Projects a vector onto the probability simplex
    2. `prox_gradient`: Solves optimization problems of the form $\min_x \frac{1}{2} \|Ax - b\|^2$
    subject to simplex constraints

    These functions are implemented efficiently and can handle large-scale problems.
    """
    )
    return


if __name__ == "__main__":
    app.run()
