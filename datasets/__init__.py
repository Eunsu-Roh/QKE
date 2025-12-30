"""Dataset loading and generation modules."""

from .iris import load_iris_2d, load_iris_binary
from .synthetic import (
    generate_xor,
    generate_concentric_circles,
    generate_spirals,
    generate_checkerboard
)

__all__ = [
    'load_iris_2d',
    'load_iris_binary',
    'generate_xor',
    'generate_concentric_circles',
    'generate_spirals',
    'generate_checkerboard'
]
