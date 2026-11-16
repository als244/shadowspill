"""Portable logical groups and input delivery, independent of model sharding."""

from .groups import GroupRef
from .inputs import InputFieldPlacement, InputPlacement, OutputPlacement

__all__ = ["GroupRef", "InputFieldPlacement", "InputPlacement", "OutputPlacement"]
