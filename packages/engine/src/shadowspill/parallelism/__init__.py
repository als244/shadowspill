"""Portable logical groups, parameter placement and storage declarations."""

from .groups import GroupRef
from .inputs import InputFieldPlacement, InputPlacement, OutputPlacement
from .placements import Replicated, Sharded
from .spec import Parallelism
from .storage import ParameterStorage

__all__ = [
    "GroupRef",
    "InputFieldPlacement",
    "InputPlacement",
    "OutputPlacement",
    "Parallelism",
    "ParameterStorage",
    "Replicated",
    "Sharded",
]
