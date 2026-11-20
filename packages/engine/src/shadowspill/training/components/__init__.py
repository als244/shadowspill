"""Public extension hooks shared by external models and supplied workloads."""

from shadowspill.artifacts.component_layouts import ComponentLayout
from shadowspill.training.inputs.adapters import InputAdapter, InputLayout, InputView

from .build import ModelBuild, ModelBuildContext
from .contracts import (
    ModelComponent,
    ObjectiveComponent,
    ObjectiveEvaluation,
    simple_loss,
)
from .initialization import (
    InitializationContext,
    InitializationRuntime,
    ModelInitialization,
    TensorInitializer,
)
from .optimizer import OptimizerComponent
from .packages import load_registry
from .policies import PolicyComponent
from .preparation import PreparationComponent, PreparationContext
from .references import ModelReferenceComponent
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
    "ModelComponent",
    "ModelInitialization",
    "ModelReferenceComponent",
    "ObjectiveComponent",
    "ObjectiveEvaluation",
    "OptimizerComponent",
    "PolicyComponent",
    "PreparationComponent",
    "PreparationContext",
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
