"""Tests for the proximal module."""

import numpy as np

from proximal import prox_gradient


def test_cla(resource_dir):
    """Test the correctness of the prox_gradient function.

    Check output against expected values
    using a specific covariance matrix loaded from a CSV file.

    Parameters
    ----------
    resource_dir : Path
        The directory containing the "CLA_Data.csv" file with necessary data.

    Raises
    ------
    AssertionError
        If the output of the prox_gradient function does not match the expected
        values within the specified tolerance.

    """
    covar = np.genfromtxt(resource_dir / "CLA_Data.csv", delimiter=",", skip_header=1)[3:]

    assert (
        np.linalg.norm(
            prox_gradient(covar, np.ones(10)) - np.array([0, 0.41200, 0, 0, 0, 0, 0.27612, 0.31188, 0, 0]),
            1,
        )
        < 1e-5
    )
