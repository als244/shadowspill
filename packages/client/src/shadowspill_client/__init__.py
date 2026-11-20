"""Lightweight ShadowSpill service and training clients, without engine or models."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

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
    Resources,
    Weights,
)
from .datum import Datum
from .protocol.requests import ArtifactRef
from .protocol.training import ResolvedTraining

if TYPE_CHECKING:
    from .service import ServiceClient as ServiceClient
    from .training import TrainingClient as TrainingClient
    from .transport.ssh import SSH as SSH

__version__ = "0.1.0.dev0"
__all__ = [
    "SSH",
    "ArtifactRef",
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
    "ResolvedTraining",
    "Resources",
    "ServiceClient",
    "TrainingClient",
    "Weights",
]


def __getattr__(name: str) -> Any:
    modules = {
        "ServiceClient": ".service",
        "TrainingClient": ".training",
        "SSH": ".transport.ssh",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
