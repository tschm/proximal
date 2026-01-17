"""Proximal optimization library for simplex-constrained least squares.

This package provides efficient solvers for optimization problems of the form:

    minimize 0.5 ||mat @ x - vec||^2
    subject to x >= 0, sum(x) = 1

These problems arise in portfolio optimization, machine learning, and signal processing.

Functions
---------
proj_simplex
    Project a vector onto the probability simplex.
prox_gradient
    Solve simplex-constrained least squares via proximal gradient descent.

Examples:
--------
>>> import numpy as np
>>> from proximal_lq import prox_gradient
>>> mat = np.array([[1.0, 0.5], [0.5, 1.0]])
>>> vec = np.ones(2)
>>> result = prox_gradient(mat, vec)
>>> print(np.round(result, 4))
[0.5 0.5]

"""

import importlib.metadata

__version__ = importlib.metadata.version("proximal-lq")
__all__ = ["proj_simplex", "prox_gradient", "__version__"]

from .proximal import proj_simplex as proj_simplex
from .proximal import prox_gradient as prox_gradient
