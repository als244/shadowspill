"""Public extension hooks shared by external models and supplied workloads."""

from shadowspill.artifacts.component_layouts import ComponentLayout
from shadowspill.training.inputs.adapters import InputAdapter, InputLayout, InputView

from .build import ModelBuild, ModelBuildContext
from .contracts import (
    ModelDefinition,
    ObjectiveDefinition,
    ObjectiveEvaluation,
    simple_loss,
)
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
from .samples import RepresentativePattern, RepresentativeSample, UniformIntegers

__all__ = [
    "ComponentLayout",
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
    "ObjectiveEvaluation",
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
    "UniformIntegers",
    "load_registry",
    "simple_loss",
]
