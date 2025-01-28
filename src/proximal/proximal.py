"""
Fast solver for 0.5 ||mat @ x - vec||^2 s. t. {x >= 0, sum(x) = 1}
"""

from __future__ import annotations

import numpy as np


def proj_simplex(vec, rad=1):
    """Based on: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf,
    efficient O(n log(n)) projection onto the standard simplex of dimension n."""
    muu = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(muu) - rad)
    rho = max(np.where(muu > cummeans)[0])
    return np.maximum(vec - cummeans[rho], 0)


def prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000):
    """Solves the constrained least square problem:
    find x minimizing: 0.5 ||mat @ x - vec||^2 s. t. {x >= 0, sum(x) = 1}.
    The fixed step proximal gradient descent is used by doing alternatively:
    a gradient step for the quadratic term 0.5 ||mat * x - vec||^2
    and the projection onto the unit simplex."""
    prim_var = np.random.randn(mat.shape[1])
    sym_mat = np.matmul(mat.T, mat)
    lip = np.linalg.norm(sym_mat, 2)  # lip. constant of the gradient
    step = 0.5 / lip if abs(lip) > 1e-15 else 1

    out_prod = np.matmul(mat.T, vec)
    ite = 0
    err_rel = eps_rel + 1
    while err_rel > eps_rel and ite < max_iter:
        prim_var_new = proj_simplex(prim_var - step * (np.matmul(sym_mat, prim_var) - out_prod))
        err_rel = np.linalg.norm(prim_var - prim_var_new, 2)
        prim_var = prim_var_new.copy()
        ite += 1
    return prim_var
