"""Persistent PyTorch state relocation and externalization."""

from .model import externalize_model_state, relocate_model_state
from .optimizer import externalize_optimizer_state, relocate_optimizer_state

__all__ = [
    "externalize_model_state",
    "externalize_optimizer_state",
    "relocate_model_state",
    "relocate_optimizer_state",
]
