"""Lightweight records for ShadowSpill training, without the engine or models."""

from .config import (
    BatchConfig,
    Execution,
    LoRA,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    Weights,
)
from .datum import Datum

__version__ = "0.1.0.dev0"
__all__ = [
    "BatchConfig",
    "Datum",
    "Execution",
    "LoRA",
    "Memory",
    "Model",
    "Objective",
    "Operation",
    "Optimizer",
    "Weights",
]
