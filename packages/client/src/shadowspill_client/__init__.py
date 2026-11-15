"""Lightweight ShadowSpill service and training clients, without engine or models."""

from .config import (
    BatchConfig,
    BuildCandidates,
    BuildCase,
    BuildOptions,
    Execution,
    LoRA,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    PlanningOptions,
    Policy,
    Preparation,
    ProfilingOptions,
    Weights,
)
from .datum import Datum
from .service import ServiceClient
from .training import TrainingClient
from .transport.ssh import SSH

__version__ = "0.1.0.dev0"
__all__ = [
    "SSH",
    "BatchConfig",
    "BuildCandidates",
    "BuildCase",
    "BuildOptions",
    "Datum",
    "Execution",
    "LoRA",
    "Memory",
    "Model",
    "Objective",
    "Operation",
    "Optimizer",
    "PlanningOptions",
    "Policy",
    "Preparation",
    "ProfilingOptions",
    "ServiceClient",
    "TrainingClient",
    "Weights",
]
