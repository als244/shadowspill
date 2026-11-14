"""Compatibility imports; implementations live in their own subdirectories."""

from .implementations.registry import build
from .implementations.sonicmoe import SonicLayer
from .implementations.tensors import tensor

__all__ = ["SonicLayer", "build", "tensor"]
