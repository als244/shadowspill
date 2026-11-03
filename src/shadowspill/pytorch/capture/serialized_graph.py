"""Lossless persistence for explicit FX task graphs.

``torch.save(GraphModule)`` serializes generated Python and symbolically traces
that Python again when loading.  Tensor-producing subgraphs with no Proxy
inputs can consequently be constant-folded during deserialization.  A cached
training task must preserve the admitted FX nodes exactly, so ShadowSpill
stores the graph records directly and reconstructs them without execution.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_aggregate

from shadowspill.errors import CaptureError

from .pointwise import (
    SavedOutput,
    pointwise_precision_barrier,
    restore_saved_outputs,
    saved_outputs,
)
from .source import SourceOperation, set_source_operation, source_operation


@dataclass(frozen=True, slots=True)
class FxNodeReference:
    """Stable reference to an earlier node in one serialized graph."""

    name: str


@dataclass(frozen=True, slots=True)
class FxCallableTarget:
    """Import-independent identity for one FX ``call_function`` target."""

    kind: str
    module: str
    name: str
    overload: str | None = None


@dataclass(frozen=True, slots=True)
class FxNodeRecord:
    """An FX node with explicit source provenance and aggregate arguments."""

    name: str
    op: str
    target: str | FxCallableTarget
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    source_operation: SourceOperation | None = None
    partitioner_tag: str | None = None
    low_precision_pointwise_barrier: bool = False
    saved_outputs: tuple[SavedOutput, ...] = ()


@dataclass(frozen=True, slots=True)
class SerializedFxGraph:
    """An explicit task graph that round-trips without symbolic tracing."""

    nodes: tuple[FxNodeRecord, ...]

    @classmethod
    def capture(cls, graph_module: GraphModule) -> SerializedFxGraph:
        records: list[FxNodeRecord] = []
        for node in graph_module.graph.nodes:
            if node.op in {"call_module", "get_attr"}:
                raise CaptureError(
                    "persistent AOT tasks must have explicit state and operators: "
                    f"node={node.name}, op={node.op}, target={node.target}"
                )
            target: str | FxCallableTarget
            if node.op == "call_function":
                target = _encode_callable(node.target)
            elif isinstance(node.target, str):
                target = node.target
            else:
                raise CaptureError(
                    "FX node target is not serializable: "
                    f"node={node.name}, op={node.op}, target={node.target!r}"
                )
            args = map_aggregate(node.args, _encode_argument_leaf)
            kwargs = map_aggregate(node.kwargs, _encode_argument_leaf)
            if not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise CaptureError("FX node aggregate shape changed during capture")
            phase = node.meta.get("partitioner_tag")
            if phase is not None and not isinstance(phase, str):
                raise CaptureError("FX partitioner tag must be a string")
            records.append(
                FxNodeRecord(
                    node.name,
                    node.op,
                    target,
                    args,
                    kwargs,
                    source_operation(node),
                    phase,
                    pointwise_precision_barrier(node),
                    saved_outputs(node),
                )
            )
        return cls(tuple(records))

    def restore(self) -> GraphModule:
        graph = Graph()
        nodes: dict[str, Node] = {}
        for record in self.nodes:
            target: Any = (
                _decode_callable(record.target)
                if isinstance(record.target, FxCallableTarget)
                else record.target
            )

            def decode(value: Any, record_name: str = record.name) -> Any:
                if isinstance(value, FxCallableTarget):
                    return _decode_callable(value)
                if not isinstance(value, FxNodeReference):
                    return value
                try:
                    return nodes[value.name]
                except KeyError as exc:
                    raise CaptureError(
                        "serialized FX graph contains a forward node reference: "
                        f"node={record_name}, reference={value.name}"
                    ) from exc

            args = map_aggregate(record.args, decode)
            kwargs = map_aggregate(record.kwargs, decode)
            node = graph.create_node(
                record.op,
                target,
                args,
                kwargs,
                name=record.name,
            )
            if record.source_operation is not None:
                set_source_operation(node, record.source_operation)
            if record.partitioner_tag is not None:
                node.meta["partitioner_tag"] = record.partitioner_tag
            if record.low_precision_pointwise_barrier:
                node.meta["low_precision_pointwise_barrier"] = True
            restore_saved_outputs(node, record.saved_outputs)
            nodes[record.name] = node
        try:
            graph.lint()
            return GraphModule({}, graph)
        except BaseException as exc:
            raise CaptureError(
                f"serialized FX graph cannot be restored: {exc}"
            ) from exc


def _encode_argument_leaf(value: Any) -> Any:
    if isinstance(value, Node):
        return FxNodeReference(value.name)
    if isinstance(value, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        # Higher-order calls take another operator as a constant argument.
        # Its dispatch-table internals cannot be pickled as Python state.
        return _encode_callable(value)
    if isinstance(value, torch.Tensor):
        raise CaptureError(
            "explicit FX task contains a literal Tensor argument; lift it to an input"
        )
    return value


def _encode_callable(target: Any) -> FxCallableTarget:
    if isinstance(target, torch._ops.HigherOrderOperator):
        name = getattr(target, "__name__", None)
        if (
            not isinstance(name, str)
            or getattr(torch.ops.higher_order, name, None) is not target
        ):
            raise CaptureError(f"higher-order target has no stable identity: {target}")
        return FxCallableTarget("higher_order", "higher_order", name)
    if isinstance(target, torch._ops.OpOverload):
        schema = target._schema
        namespace, separator, operator_name = schema.name.partition("::")
        if separator == "" or not namespace or not operator_name:
            raise CaptureError(f"operator target has an invalid schema: {schema}")
        return FxCallableTarget(
            "operator",
            namespace,
            operator_name,
            schema.overload_name or "default",
        )
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise CaptureError(f"call_function target is not importable: {target!r}")
    resolved = _resolve_symbol(module, name)
    if resolved is not target:
        raise CaptureError(
            "call_function target does not have a stable import identity: "
            f"{module}.{name}"
        )
    return FxCallableTarget("python", module, name, None)


def _decode_callable(target: FxCallableTarget) -> Any:
    if target.kind == "higher_order":
        if target.module != "higher_order" or target.overload is not None:
            raise CaptureError(f"serialized higher-order target is invalid: {target}")
        resolved = getattr(torch.ops.higher_order, target.name, None)
        if not isinstance(resolved, torch._ops.HigherOrderOperator):
            raise CaptureError(
                f"serialized higher-order operator is unavailable: {target.name}"
            )
        return resolved
    if target.kind == "python":
        if target.overload is not None:
            raise CaptureError(f"serialized Python target has an overload: {target}")
        return _resolve_symbol(target.module, target.name)
    if target.kind != "operator":
        raise CaptureError(f"serialized FX target kind is invalid: {target.kind!r}")
    if not target.module or not target.name or not target.overload:
        raise CaptureError(f"serialized operator target is invalid: {target}")
    try:
        packet = getattr(getattr(torch.ops, target.module), target.name)
        return getattr(packet, target.overload)
    except AttributeError as exc:
        raise CaptureError(
            "serialized operator is unavailable in this process: "
            f"{target.module}::{target.name}.{target.overload}"
        ) from exc


def _resolve_symbol(module: str, name: str) -> Any:
    try:
        value: Any = importlib.import_module(module)
        for component in name.split("."):
            value = getattr(value, component)
        return value
    except (ImportError, AttributeError) as exc:
        raise CaptureError(
            f"serialized Python target is unavailable: {module}.{name}"
        ) from exc


__all__ = ["SerializedFxGraph"]
