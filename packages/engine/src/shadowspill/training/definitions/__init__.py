"""Public extension hooks shared by external models and supplied workloads."""

from shadowspill.training.inputs.adapters import InputAdapter, InputLayout, InputView

from .build import ModelBuild, ModelBuildContext
from .contracts import ModelDefinition, ObjectiveDefinition, simple_loss
from .initialization import (
    InitializationContext,
    InitializationRuntime,
    ModelInitialization,
    TensorInitializer,
)
from .optimizer import OptimizerDefinition
from .packages import load_registry
from .policies import PolicyDefinition
from .preparation import PreparationContext, PreparationDefinition
from .references import ModelReferenceDefinition
from .registry import Registry, ResolvedModel, ResolvedObjective, ResolvedOptimizer
from .samples import RepresentativePattern, RepresentativeSample

__all__ = [
    "InitializationContext",
    "InitializationRuntime",
    "InputAdapter",
    "InputLayout",
    "InputView",
    "ModelBuild",
    "ModelBuildContext",
    "ModelDefinition",
    "ModelInitialization",
    "ModelReferenceDefinition",
    "ObjectiveDefinition",
    "OptimizerDefinition",
    "PolicyDefinition",
    "PreparationContext",
    "PreparationDefinition",
    "Registry",
    "RepresentativePattern",
    "RepresentativeSample",
    "ResolvedModel",
    "ResolvedObjective",
    "ResolvedOptimizer",
    "TensorInitializer",
    "load_registry",
    "simple_loss",
]
