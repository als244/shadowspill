"""Public installed model/objective functions and their portable descriptions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from shadowspill_client import Model
from shadowspill_client._records import Value, mapping
from shadowspill_client.protocol.controls import ControlSchema
from shadowspill_client.protocol.definitions import OutputSet, Participation
from shadowspill_client.protocol.inputs import InputSchema
from shadowspill_client.protocol.operations import OperationKey

from .allocations import allocation_invariants

if TYPE_CHECKING:
    from shadowspill.parallelism import Parallelism
    from shadowspill.training.inputs.adapters import InputAdapter, InputLayout
    from shadowspill.training.inputs.validation import ValidatedBatch

    from .build import ModelBuild, ModelBuildContext
    from .initialization import InitializationContext, ModelInitialization
    from .registry import ResolvedModel

type Configuration = Callable[..., Any]
type Schema = InputSchema | Callable[[Any], InputSchema]
type Computation = Callable[
    [Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any
]


def versioned_id(value: str) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise ValueError("definition ID must be a versioned name")
    name, separator, version = value.rpartition("@")
    if not name or not separator or not version or "@" in name:
        raise ValueError("definition ID must have the form namespace/name@version")
    return value


@dataclass(frozen=True)
class ModelDefinition:
    """An ordinary constructor plus CPU description and preparation hooks.

    Optional ``configure(Model)`` validates implementation/precision/LoRA
    choices and translates them into the same typed constructor config. It must
    be stable when resolved defaults are supplied again. Common parameter-name
    globs apply trainable selection after construction, including tied aliases.
    Weight initialization is separate. ``default_parallelism`` describes named
    logical placement without reserving workers or constructing the model.
    """

    id: str
    config: Configuration
    create: Callable[[Any], Any]
    inputs: Schema
    capabilities: Sequence[str] = ()
    batching: Literal["independent", "grouped", "whole_batch"] = "independent"
    mutable_buffers: Sequence[str] = ()
    input_adapter: Callable[[Any, InputLayout], InputAdapter] | None = None
    preparation: str | None = None
    configure: Callable[[Model], Mapping[str, Value]] | None = None
    default_parallelism: Callable[[Any, InputSchema], Parallelism] | None = None
    build: Callable[[Any, ModelBuildContext], ModelBuild] | None = None
    capture_context: Callable[[Any], AbstractContextManager[Any]] | None = None
    input_groups: Callable[[Any, ValidatedBatch], Sequence[Sequence[int]]] | None = None
    initialization: (
        Callable[[Any, InitializationContext], ModelInitialization] | None
    ) = None
    temporary_initialization: (
        Callable[[Any, InitializationContext], ModelInitialization] | None
    ) = None

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.config) or not callable(self.create):
            raise TypeError("model config and create must be callable")
        if not isinstance(self.inputs, InputSchema) and not callable(self.inputs):
            raise TypeError("inputs must be an InputSchema or a schema resolver")
        if self.batching not in ("independent", "grouped", "whole_batch"):
            raise ValueError("declare independent, grouped or whole_batch behavior")
        if self.input_adapter is not None and not callable(self.input_adapter):
            raise TypeError("input_adapter must be an installed binding function")
        if self.batching == "grouped":
            if not callable(self.input_groups):
                raise TypeError("grouped inputs require an installed input_groups hook")
        elif self.input_groups is not None:
            raise ValueError("input_groups requires grouped batching")
        if self.preparation is not None:
            versioned_id(self.preparation)
        if self.configure is not None and not callable(self.configure):
            raise TypeError("configure must be an installed CPU settings function")
        if self.default_parallelism is not None and not callable(
            self.default_parallelism
        ):
            raise TypeError("default_parallelism must be an installed CPU function")
        if self.build is not None and not callable(self.build):
            raise TypeError("build must be an installed CPU description function")
        if self.capture_context is not None and not callable(self.capture_context):
            raise TypeError("capture_context must be an installed context factory")
        for name in ("initialization", "temporary_initialization"):
            hook = getattr(self, name)
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} must be an installed CPU description function")
        for name in ("capabilities", "mutable_buffers"):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, str) or not v for v in values) or len(
                set(values)
            ) != len(values):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Explicit shared model evaluation for independently reduced loss terms.

    ``compute(model, inputs, loss_inputs, requests)`` returns values consumed by
    objective reducers. Compatible terms use the same versioned ID and callable;
    their request mappings are merged, rejecting conflicting values. The hook
    must evaluate the model once and must not mutate its returned values.
    """

    id: str
    compute: Callable[
        [Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Value]], Any
    ]
    requests: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.compute):
            raise TypeError("objective evaluation compute must be callable")
        object.__setattr__(
            self, "requests", mapping(self.requests, "evaluation requests")
        )


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Installed objective semantics, including its explicit padding guarantee.

    ``padding_safe`` asserts that schema/adapter padding contributes no loss or
    scalar totals and preserves outputs for real data. Declaring a participation
    field alone does not prove that a callable honors it. The preparation binder
    requires this guarantee for padded operation families. A custom objective
    that intentionally includes padding in microbatch statistics must instead
    declare ``physical_batch_loss=True``. Its loss and gradients may change
    with padding or partitioning; it does not claim logical-batch equivalence.

    ``configure(values, model)`` can resolve model-dependent structural settings
    on CPU. Inputs, controls and outputs may resolve from the typed configuration.
    ``bind(config, OperationKey)`` selects the captured computation; a forward
    output override declares its separate input/control/result contract.
    With ``evaluate(config, key)``, compute/bind instead receive the shared
    evaluation's result as their first argument. Reducers must not mutate it.
    """

    id: str
    compute: Computation
    inputs: Schema
    config: Configuration = dict
    controls: ControlSchema | Callable[[Any], ControlSchema] = field(
        default_factory=lambda: ControlSchema({})
    )
    outputs: Mapping[str, OutputSet] | Callable[[Any], Mapping[str, OutputSet]] = field(
        default_factory=lambda: {"metrics": OutputSet()}
    )
    default_outputs: str = "metrics"
    requires: Sequence[str] = ()
    participation: Participation | Callable[[Any], Participation] = field(
        default_factory=Participation
    )
    reduction: Literal["sum", "custom"] = "sum"
    bind: Callable[[Any, OperationKey], Computation] | None = None
    padding_safe: bool | Callable[[Any], bool] = False
    physical_batch_loss: bool | Callable[[Any], bool] = False
    evaluate: Callable[[Any, OperationKey], ObjectiveEvaluation] | None = None
    allocation_invariants: Mapping[str, str] = field(default_factory=dict)
    configure: (
        Callable[[Mapping[str, Value], ResolvedModel], Mapping[str, Value]] | None
    ) = None

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.compute) or not callable(self.config):
            raise TypeError("objective compute and config must be callable")
        if not isinstance(self.inputs, InputSchema) and not callable(self.inputs):
            raise TypeError("inputs must be an InputSchema or a schema resolver")
        if (
            not isinstance(self.controls, ControlSchema) and not callable(self.controls)
        ) or not (
            isinstance(self.participation, Participation)
            or callable(self.participation)
        ):
            raise TypeError("objective requires declared controls and participation")
        if self.reduction not in ("sum", "custom"):
            raise ValueError("declare summed or custom objective semantics")
        if type(self.padding_safe) is not bool and not callable(self.padding_safe):
            raise TypeError("objective padding_safe must be boolean")
        if type(self.physical_batch_loss) is not bool and not callable(
            self.physical_batch_loss
        ):
            raise TypeError("objective physical_batch_loss must be boolean")
        if self.physical_batch_loss is True and (
            self.padding_safe is True or self.reduction != "custom"
        ):
            raise ValueError(
                "physical batch loss requires custom reduction and padding_safe=False"
            )
        if self.evaluate is not None and not callable(self.evaluate):
            raise TypeError("evaluate must be an installed objective evaluation hook")
        if self.configure is not None and not callable(self.configure):
            raise TypeError("configure must be an installed CPU settings function")
        outputs = None if callable(self.outputs) else dict(self.outputs)
        if outputs is not None and (not outputs or self.default_outputs not in outputs):
            raise ValueError("objective must name a declared default output set")
        if self.bind is not None and not callable(self.bind):
            raise TypeError("objective bind must be callable")
        if (outputs is None or len(outputs) > 1) and self.bind is None:
            raise ValueError(
                "multiple output sets require an explicit computation binding"
            )
        if outputs is not None and any(
            not isinstance(k, str) or not k or not isinstance(v, OutputSet)
            for k, v in outputs.items()
        ):
            raise TypeError("outputs must map names to OutputSet records")
        if outputs is not None and isinstance(self.controls, ControlSchema):
            for output in outputs.values():
                for concrete in (output, output.forward):
                    if concrete is not None and (
                        set(concrete.controls) - self.controls.fields.keys()
                    ):
                        raise ValueError("output set references an undeclared control")
        required = tuple(self.requires)
        if any(not isinstance(v, str) or not v for v in required) or len(
            set(required)
        ) != len(required):
            raise ValueError("invalid required capabilities")
        object.__setattr__(self, "requires", required)
        if outputs is not None:
            object.__setattr__(self, "outputs", MappingProxyType(outputs))
        object.__setattr__(
            self,
            "allocation_invariants",
            allocation_invariants(self.allocation_invariants),
        )


def simple_loss(
    loss: Callable[[Any, Any], Any],
    *,
    target: str = "targets",
) -> Computation:
    """Adapt loss(predictions, targets) to the existing model-aware capture path.

    The supplied loss owns its declared reduction/validity semantics. No average,
    weighting or mutable execution loop is introduced by this adapter.
    """
    if not callable(loss) or not target:
        raise ValueError("provide a loss callable and target field")

    def compute(
        model: Any,
        inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
        loss_params: Mapping[str, Any],
    ) -> Any:
        if loss_params:
            raise ValueError("simple_loss has no runtime coefficient bindings")
        return loss(model(**inputs), loss_inputs[target])

    return compute
