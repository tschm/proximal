import numpy as np

from proximal.proximal import prox_gradient


def test_cla(resource_dir):
    covar = np.genfromtxt(resource_dir / "CLA_Data.csv", delimiter=",", skip_header=1)[3:]

    assert (
        np.linalg.norm(
            prox_gradient(covar, np.ones(10)) - np.array([0, 0.41200, 0, 0, 0, 0, 0.27612, 0.31188, 0, 0]),
            1,
        )
        < 1e-5
    )
