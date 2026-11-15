"""CPU descriptions for geometry-aware construction and existing provider hooks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from shadowspill.parallelism import Parallelism
from shadowspill.preparation import OperatorPreparation
from shadowspill.resources import Resource
from shadowspill.training.inputs.adapters import InputLayout
from shadowspill_client._records import Value, mapping, plain

from .allocations import allocation_invariants


@dataclass(frozen=True)
class ModelBuildContext:
    """One capture's layout and the complete requested family of worker layouts.

    Shared provider capacities can cover the family once, independently of the
    current physical shape. Logical ranks and groups are explicit; no physical
    device, process group or live allocation is exposed during description.
    """

    layout: InputLayout
    family: Sequence[InputLayout]
    parallelism: Parallelism

    def __post_init__(self) -> None:
        family = tuple(self.family)
        if (
            not isinstance(self.layout, InputLayout)
            or not family
            or any(not isinstance(item, InputLayout) for item in family)
            or self.layout not in family
            or not isinstance(self.parallelism, Parallelism)
            or any(
                item.placement != self.parallelism.inputs
                or not 0 <= item.logical_rank < self.parallelism.world_size
                for item in family
            )
        ):
            raise ValueError("model build context requires matching logical layouts")
        object.__setattr__(self, "family", family)


@dataclass(frozen=True)
class ModelBuild:
    """Constructor config and portable descriptions, never live provider objects.

    The same ordinary model factory receives this normalized config at capture.
    Existing Resource and OperatorPreparation APIs own creation, binary capture,
    measurement, restoration and cleanup. ``grad_dtype`` selects the existing
    captured gradient storage/accumulation precision; None keeps the core default.
    Policies and optimizer options remain separate and cannot be replaced by
    this hook.
    """

    config: Mapping[str, Value]
    resources: Mapping[str, Resource] = field(default_factory=dict)
    operator_preparation: Sequence[OperatorPreparation] = ()
    allocation_invariants: Mapping[str, str] = field(default_factory=dict)
    grad_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.grad_dtype not in (None, "float16", "bfloat16", "float32", "float64"):
            raise ValueError("model build grad_dtype must name a floating dtype")
        resources = dict(self.resources)
        if any(
            not isinstance(name, str) or not name or not isinstance(item, Resource)
            for name, item in resources.items()
        ):
            raise TypeError("model build resources must be named Resource descriptions")
        preparations = tuple(self.operator_preparation)
        if any(not isinstance(item, OperatorPreparation) for item in preparations):
            raise TypeError("model build requires OperatorPreparation descriptions")
        object.__setattr__(self, "config", mapping(self.config, "model build config"))
        object.__setattr__(self, "resources", MappingProxyType(resources))
        object.__setattr__(self, "operator_preparation", preparations)
        object.__setattr__(
            self,
            "allocation_invariants",
            allocation_invariants(self.allocation_invariants),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": plain(self.config),
            "resources": {
                name: item.to_dict() for name, item in self.resources.items()
            },
            "operator_preparation": [
                item.to_dict() for item in self.operator_preparation
            ],
            "allocation_invariants": dict(self.allocation_invariants),
            "grad_dtype": self.grad_dtype,
        }
