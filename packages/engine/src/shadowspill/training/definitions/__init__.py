"""Public extension hooks shared by external models and supplied workloads."""

from shadowspill.training.inputs.adapters import InputAdapter, InputLayout, InputView

from .contracts import ModelDefinition, ObjectiveDefinition, simple_loss
from .optimizer import OptimizerDefinition
from .registry import Registry, ResolvedModel, ResolvedObjective, ResolvedOptimizer

__all__ = [
    "InputAdapter",
    "InputLayout",
    "InputView",
    "ModelDefinition",
    "ObjectiveDefinition",
    "OptimizerDefinition",
    "Registry",
    "ResolvedModel",
    "ResolvedObjective",
    "ResolvedOptimizer",
    "simple_loss",
]
