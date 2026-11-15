"""Inspect checkpoint compatibility against saved Plans before opening workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.execution.controls import control_descriptors
from shadowspill.execution.tensor_layout import TensorLayout
from shadowspill.execution.tensors import owned_tensors, tensor_descriptors
from shadowspill.planning.shared_state import shared_state

from .logical import LogicalTensor, logical_tensors

if TYPE_CHECKING:
    import torch


class CheckpointState(Protocol):
    @property
    def plan(self) -> Plan: ...

    @property
    def plans(self) -> Mapping[str, Plan]: ...

    @property
    def logical_rank(self) -> int: ...

    def tensor_chunks(self) -> Iterator[TensorChunk]: ...


class CheckpointExport(CheckpointState, Protocol):
    """The encoder's existing borrowed tensor interface, without an Executor."""

    def export_tensors(
        self,
        callback: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int,
    ) -> None: ...


class PlanState:
    """Use the execution layout and shared owners, with no tensor allocation."""

    def __init__(
        self, plan: Plan, plans: Mapping[str, Plan], *, logical_rank: int
    ) -> None:
        self.plan, self.plans, self.logical_rank = (
            plan,
            MappingProxyType(dict(plans)),
            logical_rank,
        )
        if "" in plans:
            raise ValueError("additional Plan names must be nonempty strings")
        named = {"": plan, **plans}
        self._shared = shared_state(
            {name: item.program for name, item in named.items()},
            logical_rank=logical_rank,
        )
        self._layouts = {
            name: TensorLayout(
                item,
                logical_rank=logical_rank,
                shared_aliases=self._shared.aliases[name],
            )
            for name, item in named.items()
        }

    def tensor_chunks(self) -> Iterator[TensorChunk]:
        for name, layout in self._layouts.items():
            prefix = f"plans.{name}." if name else ""
            yield from tensor_descriptors(layout, skip_shared=bool(name), prefix=prefix)
            yield from control_descriptors(
                layout.plan.program.bindings.get("optimizer_bindings", ()),
                shared_host=self._shared.host[name],
                prefix=prefix,
            )

    def logical_tensors(self) -> Iterator[LogicalTensor]:
        """Checkpointable logical views over the same canonical export regions."""
        return logical_tensors(self._layouts, self._shared, self.logical_rank)

    def temporary_initializers(
        self,
    ) -> Iterator[tuple[TensorChunk, Mapping[str, Any] | None]]:
        """Describe target-only defaults; checkpointable state is never returned."""
        protected = {
            item.archive.name
            for item in self.logical_tensors()
            if item.archive is not None
        }
        for name, layout in self._layouts.items():
            prefix = f"plans.{name}." if name else ""
            for tensor in owned_tensors(layout, skip_shared=bool(name)):
                chunk = tensor.chunk(0, prefix)
                if chunk.name not in protected:
                    yield (
                        chunk,
                        (
                            {"kind": "constant", "value": 0}
                            if tensor.name.startswith("gradient.")
                            else tensor.initializer
                        ),
                    )
            for binding in layout.plan.program.bindings.get("optimizer_bindings", ()):
                for chunk in control_descriptors(
                    (binding,), shared_host=self._shared.host[name], prefix=prefix
                ):
                    if chunk.name not in protected:
                        yield chunk, binding.get("initialization")
