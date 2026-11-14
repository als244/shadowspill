"""Public installed model/objective functions and their portable descriptions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from shadowspill_client.protocol.controls import ControlSchema
from shadowspill_client.protocol.definitions import OutputSet, Participation
from shadowspill_client.protocol.inputs import InputSchema

if TYPE_CHECKING:
    from shadowspill.training.inputs.adapters import InputAdapter, InputLayout

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
    id: str
    config: Configuration
    create: Callable[[Any], Any]
    inputs: Schema
    capabilities: Sequence[str] = ()
    batching: Literal["independent", "grouped", "whole_batch"] = "independent"
    mutable_buffers: Sequence[str] = ()
    input_adapter: Callable[[Any, InputLayout], InputAdapter] | None = None
    preparation: str | None = None

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
        if self.preparation is not None:
            versioned_id(self.preparation)
        for name in ("capabilities", "mutable_buffers"):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, str) or not v for v in values) or len(
                set(values)
            ) != len(values):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Installed objective semantics, including its explicit padding guarantee.

    ``padding_safe`` asserts that schema/adapter padding contributes no loss or
    scalar totals and preserves outputs for real data. Declaring a participation
    field alone does not prove that a callable honors it. The preparation binder
    requires this guarantee for padded operation families.
    """

    id: str
    compute: Computation
    inputs: Schema
    config: Configuration = dict
    controls: ControlSchema = field(default_factory=lambda: ControlSchema({}))
    outputs: Mapping[str, OutputSet] = field(
        default_factory=lambda: {"metrics": OutputSet()}
    )
    default_outputs: str = "metrics"
    requires: Sequence[str] = ()
    participation: Participation = field(default_factory=Participation)
    reduction: Literal["sum", "custom"] = "sum"
    bind: Callable[[Any, str], Computation] | None = None
    padding_safe: bool = False

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.compute) or not callable(self.config):
            raise TypeError("objective compute and config must be callable")
        if not isinstance(self.inputs, InputSchema) and not callable(self.inputs):
            raise TypeError("inputs must be an InputSchema or a schema resolver")
        if not isinstance(self.controls, ControlSchema) or not isinstance(
            self.participation, Participation
        ):
            raise TypeError("objective requires declared controls and participation")
        if self.reduction not in ("sum", "custom"):
            raise ValueError("declare summed or custom objective semantics")
        if type(self.padding_safe) is not bool:
            raise TypeError("objective padding_safe must be boolean")
        outputs = dict(self.outputs)
        if not outputs or self.default_outputs not in outputs:
            raise ValueError("objective must name a declared default output set")
        if self.bind is not None and not callable(self.bind):
            raise TypeError("objective bind must be callable")
        if len(outputs) > 1 and self.bind is None:
            raise ValueError(
                "multiple output sets require an explicit computation binding"
            )
        if any(
            not isinstance(k, str) or not k or not isinstance(v, OutputSet)
            for k, v in outputs.items()
        ):
            raise TypeError("outputs must map names to OutputSet records")
        for output in outputs.values():
            if set(output.controls) - self.controls.fields.keys():
                raise ValueError("output set references an undeclared control")
        required = tuple(self.requires)
        if any(not isinstance(v, str) or not v for v in required) or len(
            set(required)
        ) != len(required):
            raise ValueError("invalid required capabilities")
        object.__setattr__(self, "requires", required)
        object.__setattr__(self, "outputs", MappingProxyType(outputs))


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
