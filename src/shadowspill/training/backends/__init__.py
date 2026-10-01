"""Execution choices, without data or model-family policy."""

from .pytorch import PyTorch
from .shadowspill import ShadowSpill

__all__ = ["PyTorch", "ShadowSpill"]
