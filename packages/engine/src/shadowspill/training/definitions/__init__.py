"""Public extension hooks shared by external models and supplied workloads."""

from .contracts import ModelDefinition, ObjectiveDefinition, simple_loss
from .registry import Registry, ResolvedModel, ResolvedObjective

__all__ = [
    "ModelDefinition",
    "ObjectiveDefinition",
    "Registry",
    "ResolvedModel",
    "ResolvedObjective",
    "simple_loss",
]
