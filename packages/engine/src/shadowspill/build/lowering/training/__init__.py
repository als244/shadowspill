"""Bind training computations to untimed IR."""

from .program import lower_training

__all__ = ["lower_training"]
