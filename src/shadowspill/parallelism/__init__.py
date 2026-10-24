"""Portable logical groups, parameter placement and storage declarations."""

from .groups import GroupRef
from .placements import Replicated, Sharded
from .spec import Parallelism
from .storage import ParameterStorage

__all__ = ["GroupRef", "Parallelism", "ParameterStorage", "Replicated", "Sharded"]
