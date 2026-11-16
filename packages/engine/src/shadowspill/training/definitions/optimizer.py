"""Installed capturable optimizer factories and their declared runtime controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill_client import Optimizer
from shadowspill_client._records import Value
from shadowspill_client.protocol.controls import ControlSchema

from .allocations import allocation_invariants
from .contracts import Configuration, versioned_id

if TYPE_CHECKING:
    from shadowspill.pytorch.optimizer.artifacts import OptimizerCapture


@dataclass(frozen=True)
class OptimizerDefinition:
    """Create an ordinary optimizer for existing capture, never a second updater.

    The factory receives the model, validated configuration and tensor-valued
    controls prepared by the build worker. Public control names map explicitly
    to captured bindings; a declared override must affect the saved computation.
    Optional ``configure(Optimizer)`` validates groups/precision and translates
    them into that same factory config on CPU, with stable resolved defaults.
    ``master_dtype`` may be a dtype name or CPU config resolver; it selects the
    existing optimizer-parameter capture, without changing model dtype. Ordinary
    parameters need no separate copy when they already meet that precision.
    """

    id: str
    create: Callable[[Any, Any, Mapping[str, Any]], Any]
    controls: ControlSchema
    control_bindings: Mapping[str, Sequence[str]]
    config: Configuration = dict
    defaults: Mapping[str, Any] = field(default_factory=dict)
    configure: Callable[[Optimizer], Mapping[str, Value]] | None = None
    allocation_invariants: Mapping[str, str] = field(default_factory=dict)
    master_dtype: str | Callable[[Any], str | None] | None = None

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.config) or not callable(self.create):
            raise TypeError("optimizer config and create must be callable")
        if self.configure is not None and not callable(self.configure):
            raise TypeError("configure must be an installed CPU settings function")
        if not callable(self.master_dtype) and self.master_dtype not in (
            None,
            "float16",
            "bfloat16",
            "float32",
            "float64",
        ):
            raise ValueError("optimizer master_dtype must name a floating dtype")
        if not isinstance(self.controls, ControlSchema):
            raise TypeError("optimizer requires a ControlSchema")
        if self.control_bindings.keys() != self.controls.fields.keys():
            raise ValueError(
                "every optimizer control requires explicit captured bindings"
            )
        bindings = {}
        used = set()
        for control, values in self.control_bindings.items():
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{control} requires a nonempty binding sequence")
            for name in values:
                if not isinstance(name, str) or not name or name in used:
                    raise ValueError(
                        "optimizer bindings must be distinct nonempty names"
                    )
                used.add(name)
            bindings[control] = tuple(values)
        object.__setattr__(self, "control_bindings", MappingProxyType(bindings))
        object.__setattr__(
            self, "defaults", self.controls.resolve(defaults=self.defaults)
        )
        object.__setattr__(
            self,
            "allocation_invariants",
            allocation_invariants(self.allocation_invariants),
        )

    def validate_capture(self, captured: OptimizerCapture) -> None:
        """Refuse advertised controls that the real captured optimizer cannot read."""
        bindings = {binding.name: binding for binding in captured.bindings}
        consumed = set()
        for task in captured.update_tasks:
            placeholders = [
                node
                for node in task.artifact.graph_module.graph.nodes
                if node.op == "placeholder"
            ]
            for name, node in zip(task.binding_names, placeholders, strict=True):
                if any(user.op != "output" for user in node.users):
                    consumed.add(name)
        for names in self.control_bindings.values():
            for name in names:
                binding = bindings.get(name)
                if (
                    binding is None
                    or binding.role != "hyperparameter"
                    or name not in consumed
                ):
                    raise ValueError(
                        f"optimizer control has no captured binding: {name}"
                    )
                if binding.tensor.numel() != 1 or binding.mutable:
                    raise ValueError(
                        f"optimizer control is not a read-only scalar: {name}"
                    )
