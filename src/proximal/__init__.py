"""Proximal."""

import importlib.metadata

__version__ = importlib.metadata.version("proximal")

from .proximal import proj_simplex as proj_simplex
from .proximal import prox_gradient as prox_gradient
