"""Public extension hooks shared by external models and supplied workloads."""

from .contracts import ModelDefinition, ObjectiveDefinition, simple_loss
from .optimizer import OptimizerDefinition
from .registry import Registry, ResolvedModel, ResolvedObjective, ResolvedOptimizer

__all__ = [
    "ModelDefinition",
    "ObjectiveDefinition",
    "OptimizerDefinition",
    "Registry",
    "ResolvedModel",
    "ResolvedObjective",
    "ResolvedOptimizer",
    "simple_loss",
]
