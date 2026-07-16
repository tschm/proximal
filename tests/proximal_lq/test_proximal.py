"""Tests for the proximal_lq.proximal module.

Tests are written as module-level functions (not classes) to mirror the
function-only public surface of ``src/proximal_lq/proximal.py`` and keep the
test/source layout parity gate (``check_test_layout.py``) green.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from proximal_lq import proj_simplex, prox_gradient

# --------------------------------------------------------------------------- #
# proj_simplex
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 5, 10, 100, 1000])
def test_proj_simplex_output_sums_to_one(n: int) -> None:
    """Verify projected vector sums to 1 (default radius)."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(n)
    result = proj_simplex(vec)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)


@pytest.mark.parametrize("rad", [0.5, 1.0, 2.0, 10.0])
def test_proj_simplex_output_sums_to_radius(rad: float) -> None:
    """Verify projected vector sums to specified radius."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(10)
    result = proj_simplex(vec, rad=rad)
    np.testing.assert_allclose(result.sum(), rad, rtol=1e-10)


def test_proj_simplex_output_non_negative() -> None:
    """Verify all elements of projected vector are non-negative."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(100)
    result = proj_simplex(vec)
    assert np.all(result >= 0)


def test_proj_simplex_idempotent() -> None:
    """Projecting a valid simplex point should return the same point."""
    vec = np.array([0.2, 0.3, 0.5])
    result = proj_simplex(vec)
    np.testing.assert_allclose(result, vec, rtol=1e-10)


def test_proj_simplex_already_on_simplex() -> None:
    """Vector already on simplex should be unchanged."""
    vec = np.array([0.25, 0.25, 0.25, 0.25])
    result = proj_simplex(vec)
    np.testing.assert_allclose(result, vec, rtol=1e-10)


def test_proj_simplex_single_element() -> None:
    """Single element vector projects to [rad]."""
    result = proj_simplex(np.array([5.0]))
    np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-10)


def test_proj_simplex_all_negative_input() -> None:
    """All negative inputs should project to corner of simplex."""
    vec = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    result = proj_simplex(vec)
    # Should project to [1, 0, 0, 0, 0] since -1 is largest
    assert result[0] == pytest.approx(1.0, rel=1e-10)
    assert np.sum(result[1:]) == pytest.approx(0.0, abs=1e-10)


def test_proj_simplex_uniform_input() -> None:
    """Uniform input should project to uniform simplex point."""
    vec = np.array([3.0, 3.0, 3.0, 3.0])
    result = proj_simplex(vec)
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_proj_simplex_projection_is_closest_point(n: int) -> None:
    """Verify projection is the closest point on simplex."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(n)
    proj = proj_simplex(vec)

    # Any point on simplex should be farther from vec than proj
    # Test with some random simplex points
    for _ in range(10):
        random_simplex = rng.dirichlet(np.ones(n))
        dist_to_proj = np.linalg.norm(vec - proj)
        dist_to_random = np.linalg.norm(vec - random_simplex)
        assert dist_to_proj <= dist_to_random + 1e-10


def test_proj_simplex_two_elements() -> None:
    """Test projection with only 2 elements."""
    vec = np.array([2.0, -1.0])
    result = proj_simplex(vec)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
    assert np.all(result >= 0)


def test_proj_simplex_large_values() -> None:
    """Test projection with very large values."""
    vec = np.array([1e10, 1e10, 1e10])
    result = proj_simplex(vec)
    # Numerical precision degrades with very large values
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(result, np.array([1 / 3, 1 / 3, 1 / 3]), rtol=1e-5)


def test_proj_simplex_small_values() -> None:
    """Test projection with very small values."""
    vec = np.array([1e-10, 2e-10, 3e-10])
    result = proj_simplex(vec)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)


def test_proj_simplex_mixed_scales() -> None:
    """Test projection with mixed scale values."""
    vec = np.array([1e8, 1e-8, 1e4, 1e-4])
    result = proj_simplex(vec)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
    assert np.all(result >= 0)


def test_proj_simplex_empty_input_raises() -> None:
    """Empty input should raise a clear ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        proj_simplex(np.array([]))


# --------------------------------------------------------------------------- #
# prox_gradient
# --------------------------------------------------------------------------- #


def test_prox_gradient_cla(resource_dir) -> None:
    """Test against CLA expected values."""
    covar = np.genfromtxt(resource_dir / "CLA_Data.csv", delimiter=",", skip_header=1)[3:]
    result = prox_gradient(covar, np.ones(10), seed=0)
    expected = np.array([0, 0.41200, 0, 0, 0, 0, 0.27612, 0.31188, 0, 0])
    assert np.linalg.norm(result - expected, 1) < 1e-5


def test_prox_gradient_output_on_simplex() -> None:
    """Verify output satisfies simplex constraints."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((5, 3))
    vec = rng.standard_normal(5)
    result = prox_gradient(mat, vec, seed=0)

    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-6)
    assert np.all(result >= -1e-10)


@pytest.mark.parametrize("shape", [(3, 2), (5, 5), (10, 3), (20, 10)])
def test_prox_gradient_various_shapes(shape: tuple[int, int]) -> None:
    """Test with various matrix shapes."""
    rng = np.random.default_rng(42)
    m, n = shape
    mat = rng.standard_normal((m, n))
    vec = rng.standard_normal(m)
    result = prox_gradient(mat, vec, seed=0)

    assert result.shape == (n,)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
    assert np.all(result >= -1e-10)


def test_prox_gradient_identity_matrix() -> None:
    """Test with identity matrix - solution should be projection of vec."""
    n = 5
    mat = np.eye(n)
    vec = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # Already on simplex
    result = prox_gradient(mat, vec, seed=0)
    np.testing.assert_allclose(result, vec, rtol=1e-5)


def test_prox_gradient_convergence_tolerance() -> None:
    """Test that tighter tolerance gives valid results."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((10, 5))
    vec = rng.standard_normal(10)

    result_loose = prox_gradient(mat, vec, eps_rel=1e-3, seed=0)
    result_tight = prox_gradient(mat, vec, eps_rel=1e-10, seed=0)

    # Both should satisfy constraints
    np.testing.assert_allclose(result_loose.sum(), 1.0, rtol=1e-3)
    np.testing.assert_allclose(result_tight.sum(), 1.0, rtol=1e-10)


def test_prox_gradient_max_iter_respected() -> None:
    """Test that max_iter parameter is respected."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((100, 50))
    vec = rng.standard_normal(100)

    # Should still produce valid output even with few iterations
    result = prox_gradient(mat, vec, max_iter=10, seed=0)
    assert result.shape == (50,)


def test_prox_gradient_objective_convergence() -> None:
    """Test that a seeded solve is reproducible and near the minimum."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((5, 3))
    vec = rng.standard_normal(5)

    # Different initial seeds must reach the same objective (convex problem).
    objectives = []
    for init_seed in range(5):
        result = prox_gradient(mat, vec, eps_rel=1e-10, seed=init_seed)
        obj = 0.5 * np.linalg.norm(mat @ result - vec) ** 2
        objectives.append(obj)

    # All objectives should be close to the minimum
    np.testing.assert_allclose(objectives, objectives[0], rtol=1e-3)


def test_prox_gradient_seed_is_reproducible() -> None:
    """The same seed must yield bit-identical results."""
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((8, 4))
    vec = rng.standard_normal(8)

    first = prox_gradient(mat, vec, seed=123)
    second = prox_gradient(mat, vec, seed=123)
    np.testing.assert_array_equal(first, second)


def test_prox_gradient_near_zero_matrix() -> None:
    """Test with near-zero matrix (low Lipschitz constant)."""
    mat = np.array([[1e-10, 0], [0, 1e-10]])
    vec = np.array([1.0, 1.0])
    result = prox_gradient(mat, vec, seed=0)
    assert result.shape == (2,)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)


def test_prox_gradient_rank_deficient() -> None:
    """Test with rank-deficient matrix."""
    mat = np.array([[1, 2, 3], [2, 4, 6]])  # Rank 1
    vec = np.array([1.0, 2.0])
    result = prox_gradient(mat, vec, seed=0)
    assert result.shape == (3,)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)


def test_prox_gradient_ill_conditioned() -> None:
    """Test with ill-conditioned matrix."""
    # Create an ill-conditioned matrix
    u = np.array([[1, 0], [0, 1], [0, 0]])
    s = np.array([1e6, 1e-6])
    vt = np.array([[1, 0], [0, 1]])
    mat = u @ np.diag(s) @ vt
    vec = np.array([1.0, 1.0, 1.0])

    result = prox_gradient(mat, vec, eps_rel=1e-6, max_iter=5000, seed=0)
    assert result.shape == (2,)
    # May not converge perfectly but should satisfy constraints approximately
    assert abs(result.sum() - 1.0) < 0.01


@pytest.mark.parametrize("seed", range(5))
def test_prox_gradient_random_problems(seed: int) -> None:
    """Test with various random problem instances."""
    rng = np.random.default_rng(seed)
    m, n = rng.integers(5, 20), rng.integers(2, 10)
    mat = rng.standard_normal((m, n))
    vec = rng.standard_normal(m)

    result = prox_gradient(mat, vec, seed=0)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)
    assert np.all(result >= -1e-10)


# Correctness-at-scale checks (formerly in test_benchmarks.py). These assert the
# solver stays correct as problem size grows; true timing benchmarks belong to
# `make benchmark`.
@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_proj_simplex_scaling(n: int) -> None:
    """proj_simplex stays correct as input size grows."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(n)
    result = proj_simplex(vec)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
    assert np.all(result >= 0)


@pytest.mark.parametrize(
    ("m", "n", "max_iter", "rtol"),
    [(10, 5, 1000, 1e-5), (100, 50, 1000, 1e-5), (1000, 500, 2000, 1e-4)],
)
def test_prox_gradient_scaling(m: int, n: int, max_iter: int, rtol: float) -> None:
    """prox_gradient stays correct as problem size grows."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((m, n))
    vec = rng.standard_normal(m)
    result = prox_gradient(mat, vec, max_iter=max_iter, seed=0)
    np.testing.assert_allclose(result.sum(), 1.0, rtol=rtol)


# --------------------------------------------------------------------------- #
# Argument-shape validation (issue #332)
# --------------------------------------------------------------------------- #


def test_prox_gradient_dimension_mismatch_raises() -> None:
    """A vec length that does not match mat.shape[0] raises a clear ValueError."""
    mat = np.ones((5, 3))
    vec = np.ones(4)  # should be length 5
    with pytest.raises(ValueError, match=r"vec length \(4\) must match mat.shape\[0\] \(5\)"):
        prox_gradient(mat, vec)


def test_prox_gradient_empty_inputs_raise() -> None:
    """Empty mat/vec raise a clear ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        prox_gradient(np.zeros((0, 3)), np.zeros(0))


def test_prox_gradient_non_2d_mat_raises() -> None:
    """A non-2-D mat raises a clear ValueError."""
    with pytest.raises(ValueError, match="mat must be a 2-D array"):
        prox_gradient(np.ones(5), np.ones(5))


def test_prox_gradient_non_1d_vec_raises() -> None:
    """A non-1-D vec raises a clear ValueError."""
    with pytest.raises(ValueError, match="vec must be a 1-D array"):
        prox_gradient(np.ones((5, 3)), np.ones((5, 1)))


# --------------------------------------------------------------------------- #
# KKT / optimality properties (issues #333)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_proj_simplex_kkt(seed: int) -> None:
    """Verify KKT optimality conditions for simplex projection.

    For the projection problem::

        min 0.5 ||x - v||^2  s.t.  sum(x) = 1, x >= 0

    KKT requires primal feasibility and, for active components (x_i > 0),
    x_i = v_i - lambda with a single common multiplier lambda.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(10)
    x = proj_simplex(v)

    # Primal feasibility
    np.testing.assert_allclose(x.sum(), 1.0, rtol=1e-10)
    assert np.all(x >= -1e-12)

    # For x_i > 0, we have x_i = v_i - lambda, so lambda is common across support.
    active_mask = x > 1e-10
    if np.any(active_mask):
        lambdas = v[active_mask] - x[active_mask]
        np.testing.assert_allclose(lambdas, lambdas[0], rtol=1e-8)


def _stationarity_residual(mat: np.ndarray, vec: np.ndarray, x: np.ndarray) -> float:
    """First-order stationarity certificate for the projected-gradient optimum.

    At a KKT point of the simplex-constrained least-squares problem, ``x`` is a
    fixed point of the projected-gradient map ``x -> proj_simplex(x - t * grad)``
    for the algorithm's own step ``t``. Returns ``||x - proj_simplex(x - t*grad)||``,
    which is (up to floating point) zero at the optimum.
    """
    sym_mat = mat.T @ mat
    lip = np.linalg.norm(sym_mat, 2)
    step = 0.5 / lip if abs(lip) > 1e-15 else 1.0
    grad = sym_mat @ x - mat.T @ vec
    return float(np.linalg.norm(x - proj_simplex(x - step * grad), 2))


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    data=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 8), st.integers(1, 6)),
        elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    ),
)
@pytest.mark.property
def test_prox_gradient_kkt_optimality(data: np.ndarray) -> None:
    """Property: seeded prox_gradient returns a feasible, first-order-optimal point.

    Over generated matrices, the solution must be primal-feasible (on the
    simplex) and a stationary point of the projected-gradient map. Because the
    objective is convex, first-order stationarity implies global optimality.
    """
    mat = data
    m, n = mat.shape
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(m)

    x = prox_gradient(mat, vec, eps_rel=1e-12, max_iter=20000, seed=0)

    # Primal feasibility.
    assert x.shape == (n,)
    np.testing.assert_allclose(x.sum(), 1.0, rtol=1e-5)
    assert np.all(x >= -1e-8)

    # First-order optimality (stationarity of the projected-gradient map).
    assert _stationarity_residual(mat, vec, x) < 1e-5
