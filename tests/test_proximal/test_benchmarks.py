"""Performance benchmarks for proximal_lq functions.

Run with: pytest tests/test_proximal/test_benchmarks.py -v
For detailed benchmarks: pytest tests/test_proximal/test_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

from proximal_lq import proj_simplex, prox_gradient


@pytest.fixture
def small_problem():
    """Small optimization problem (n=10)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 5)), rng.standard_normal(10)


@pytest.fixture
def medium_problem():
    """Medium optimization problem (n=100)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 50)), rng.standard_normal(100)


@pytest.fixture
def large_problem():
    """Large optimization problem (n=1000)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1000, 500)), rng.standard_normal(1000)


class TestProjSimplexScaling:
    """Test that proj_simplex scales appropriately with input size."""

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_scaling(self, n: int) -> None:
        """Test algorithm completes and is correct for various sizes."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(n)

        result = proj_simplex(vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)
        assert np.all(result >= 0)


class TestProxGradientScaling:
    """Test that prox_gradient scales appropriately with problem size."""

    def test_small_problem(self, small_problem) -> None:
        """Test prox_gradient with small problem."""
        mat, vec = small_problem
        result = prox_gradient(mat, vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_medium_problem(self, medium_problem) -> None:
        """Test prox_gradient with medium problem."""
        mat, vec = medium_problem
        result = prox_gradient(mat, vec)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_large_problem(self, large_problem) -> None:
        """Test prox_gradient with large problem."""
        mat, vec = large_problem
        result = prox_gradient(mat, vec, max_iter=2000)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-4)
