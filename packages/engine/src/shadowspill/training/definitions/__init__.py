"""Public extension hooks shared by external models and supplied workloads."""

from shadowspill.training.inputs.adapters import InputAdapter, InputLayout, InputView

from .contracts import ModelDefinition, ObjectiveDefinition, simple_loss
from .optimizer import OptimizerDefinition
from .policies import PolicyDefinition
from .preparation import PreparationContext, PreparationDefinition
from .references import ModelReferenceDefinition
from .registry import Registry, ResolvedModel, ResolvedObjective, ResolvedOptimizer

__all__ = [
    "InputAdapter",
    "InputLayout",
    "InputView",
    "ModelDefinition",
    "ModelReferenceDefinition",
    "ObjectiveDefinition",
    "OptimizerDefinition",
    "PolicyDefinition",
    "PreparationContext",
    "PreparationDefinition",
    "Registry",
    "ResolvedModel",
    "ResolvedObjective",
    "ResolvedOptimizer",
    "simple_loss",
]
