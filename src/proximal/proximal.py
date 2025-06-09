"""Fast solver for 0.5 ||mat @ x - vec||^2 s. t. {x >= 0, sum(x) = 1}."""

from __future__ import annotations

import numpy as np


def proj_simplex(vec, rad=1):
    """Project a vector onto the probability simplex.

    This function computes the Euclidean projection of a given vector onto the probability
    simplex. The simplex is defined as the set of non-negative vectors that sum to a
    given radius, typically 1. The projection ensures that the resulting vector satisfies
    these constraints.

    Parameters
    ----------
    vec : ndarray
        Input vector that is to be projected onto the simplex.
    rad : float, optional
        Radius of the simplex. The projected vector will have components summing
        to this value. Default is 1.

    Returns
    -------
    ndarray
        The projected vector that lies on the probability simplex.

    Raises
    ------
    ValueError
        If the input vector is empty.

    """
    muu = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(muu) - rad)
    rho = max(np.where(muu > cummeans)[0])
    return np.maximum(vec - cummeans[rho], 0)


def prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000):
    """Perform a proximal gradient descent to solve an optimization problem.

    The function computes the projection on the simplex and iteratively finds
    the solution by minimizing the objective function using proximal gradient
    descent. It stops when the relative error between consecutive iterations
    is smaller than a specified threshold or when the maximum number of
    iterations is reached.

    Parameters
    ----------
    mat : np.ndarray
        A matrix of shape (n_samples, n_features) used in the optimization
        problem.
    vec : np.ndarray
        A vector of shape (n_samples,) used in the optimization problem.
    eps_rel : float, optional
        The relative error threshold for stopping criteria. Default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the algorithm. Default is 1000.

    Returns
    -------
    np.ndarray
        The solution vector of shape (n_features,) obtained after the
        optimization process.

    Raises
    ------
    None

    """
    rng = np.random.default_rng()
    prim_var = rng.standard_normal(mat.shape[1])
    sym_mat = mat.T @ mat
    lip = np.linalg.norm(sym_mat, 2)  # lip. constant of the gradient
    step = 0.5 / lip if abs(lip) > 1e-15 else 1

    out_prod = mat.T @ vec
    ite = 0
    err_rel = eps_rel + 1
    while err_rel > eps_rel and ite < max_iter:
        prim_var_new = proj_simplex(prim_var - step * (sym_mat @ prim_var - out_prod))
        err_rel = np.linalg.norm(prim_var - prim_var_new, 2)
        prim_var = prim_var_new.copy()
        ite += 1
    return prim_var
