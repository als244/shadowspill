"""Explicit, grouped destination contracts for opaque gradient producers.

Rules are caller-owned build inputs. The replacement is an ordinary dispatcher
operator, so its mutations and workspace pass through the normal artifact and
compilation paths. No callback or registry is consulted by an executable.
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._prims_common import get_computation_dtype
from torch.fx import Graph, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError

from .nodes import _CONVERTING, _MOVING_ONLY, _record_value, _undo_move


@dataclass(frozen=True, slots=True)
class GradientDestination:
    """One producer output, its final gradient, and its existing destination.

    ``operations`` runs backwards from the final gradient to the producer's
    output. It excludes the tuple getitem. A mapper must preserve the meaning
    of these operations, including casts, scaling and layout changes.
    """

    output_index: int
    gradient: Node
    destination: Node
    operations: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class DestinationView:
    """A storage-preserving operation applied to a supplied destination."""

    target: Callable[..., Any]
    arguments: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class GradientDestinationMapping:
    """Outputs to consume together, writable views, and provider-specific data.

    Each ``views`` entry is applied in order to that output's destination. An
    empty entry passes the destination itself. Extra ``keywords`` become actual
    operator arguments, not ignored graph metadata. The mapper does not edit FX.
    """

    output_indices: tuple[int, ...]
    views: tuple[tuple[DestinationView, ...], ...]
    keywords: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GradientDestinationContext:
    """Build-only information supplied once for one custom producer."""

    producer: Node
    destinations: tuple[GradientDestination, ...]
    fuse_narrow_gradient_accumulation: bool


@dataclass(frozen=True, slots=True)
class GradientDestinationRule:
    """Map a fresh-output operator to a versioned accumulating overload.

    Targets are dispatcher names, e.g. ``provider.backward.default``. The
    replacement takes the original arguments plus keyword arguments
    ``destinations: Tensor[]``, ``output_indices: int[]`` and
    ``contract_version: int``. Its schema must declare the destination list
    mutable. It preserves the original result tuple and unconsumed outputs;
    consumed positions may return ``None``. The graph returns the destinations.

    With no mapper, only reversible views/copies at the same dtype are consumed.
    A mapper may consume a subset and describe a provider's layout/scale/cast
    epilogue. It must honor the numerical permission in the context; changing
    a cast or sum's rounding requires explicit support by that provider. Returning
    ``None`` declines fusion. The normal in-place addition remains available.

    The version is passed to the operator and therefore participates in artifact
    identity and serialization. Changing an adapter's contract requires changing
    its version. A rule never authorizes duplicate backward evaluation.
    """

    target: str
    replacement: str
    output_indices: tuple[int, ...]
    version: int
    mapper: (
        Callable[[GradientDestinationContext], GradientDestinationMapping | None] | None
    ) = None

    def __post_init__(self) -> None:
        for name in (self.target, self.replacement):
            if len(name.split(".")) != 3 or not all(
                p.isidentifier() for p in name.split(".")
            ):
                raise ValueError(
                    "gradient destination targets require namespace.name.overload"
                )
        if type(self.version) is not int or self.version < 1:
            raise ValueError("gradient destination contract version must be positive")
        if not self.output_indices or any(
            type(i) is not int or i < 0 for i in self.output_indices
        ):
            raise ValueError(
                "gradient destination output indices must be nonnegative integers"
            )
        if len(set(self.output_indices)) != len(self.output_indices):
            raise ValueError("gradient destination output indices must be distinct")


_UNARY_PATH = (
    _MOVING_ONLY
    | _CONVERTING
    | {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
    }
)
_RESERVED = {"destinations", "output_indices", "contract_version"}


def _resolve(name: str) -> torch._ops.OpOverload:
    namespace, operation, overload = name.split(".")
    try:
        result = getattr(getattr(getattr(torch.ops, namespace), operation), overload)
    except AttributeError as error:
        raise CaptureError(
            f"gradient destination operator {name} is not registered"
        ) from error
    if not isinstance(result, torch._ops.OpOverload):
        raise CaptureError(f"gradient destination target {name} is not an overload")
    return result


def _origin(
    gradient: Node,
) -> tuple[Node, int, tuple[Node, ...], tuple[Node, ...]] | None:
    node = gradient
    operations: list[Node] = []
    while node.op == "call_function" and node.target in _UNARY_PATH:
        tensors = [
            n
            for n in node.all_input_nodes
            if isinstance(n.meta.get("val"), torch.Tensor)
        ]
        if (
            len(tensors) != 1
            or not node.args
            or tensors[0] is not node.args[0]
            or len(node.users) != 1
        ):
            return None
        operations.append(node)
        node = tensors[0]
    removed = [*operations, node]
    if node.target is operator.getitem:
        producer, index = node.args
        if not isinstance(producer, Node) or type(index) is not int or index < 0:
            return None
        if len(node.users) != 1 or any(
            user.target is not operator.getitem
            or len(user.args) != 2
            or type(user.args[1]) is not int
            for user in producer.users
        ):
            return None
        if sum(user.args[1] == index for user in producer.users) != 1:
            return None
    else:
        producer, index = node, 0
        removed = operations
        if len(producer.users) != 1:
            return None
    schema = getattr(producer.target, "_schema", None)
    if (
        schema is None
        or not 0 <= index < len(schema.returns)
        or schema.returns[index].alias_info is not None
    ):
        return None
    if any(
        a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
    ):
        return None
    return producer, index, tuple(operations), tuple(removed)


def _default_mapping(
    context: GradientDestinationContext,
) -> GradientDestinationMapping | None:
    indices, views = [], []
    for item in context.destinations:
        if any(node.target not in _MOVING_ONLY for node in item.operations):
            continue
        value = item.destination.meta["val"]
        if (
            not context.fuse_narrow_gradient_accumulation
            and value.dtype != get_computation_dtype(value.dtype)
        ):
            continue
        steps = tuple(
            DestinationView(*step)
            for step in map(_undo_move, item.operations)
            if step is not None
        )
        raw = context.producer.meta["val"]
        raw = raw[item.output_index] if isinstance(raw, (tuple, list)) else raw
        try:
            for step in steps:
                value = step.target(value, *step.arguments)
            if (
                value.untyped_storage()._cdata
                != item.destination.meta["val"].untyped_storage()._cdata
                or value.shape != raw.shape
                or value.dtype != raw.dtype
                or torch._debug_has_internal_overlap(value) != 0
            ):
                continue
        except RuntimeError:
            continue
        indices.append(item.output_index)
        views.append(steps)
    return GradientDestinationMapping(tuple(indices), tuple(views)) if indices else None


def fuse_gradient_destinations(
    graph: Graph,
    requests: Sequence[tuple[Node, Node]],
    rules: Sequence[GradientDestinationRule],
    *,
    outputs: Sequence[object],
    fake_mode: Any,
    fuse_narrow_gradient_accumulation: bool,
) -> tuple[dict[Node, Node], tuple[Node, ...]]:
    """Consume eligible gradient expressions, replacing each producer once.

    Callers replace uses of returned gradient nodes with the matching priors,
    then erase the returned dead expression nodes in their returned order.
    Destinations must already exist before the producer and must not alias any
    other supplied destination or any tensor the producer reads.
    """
    if not requests or not rules:
        return {}, ()
    by_target = {rule.target: rule for rule in rules}
    if len(by_target) != len(rules):
        raise CaptureError("gradient destination rules repeat an operator target")
    positions = {node: i for i, node in enumerate(graph.nodes)}
    owners: dict[Node, list[tuple[GradientDestination, tuple[Node, ...]]]] = (
        defaultdict(list)
    )
    storage = Counter(
        prior.meta["val"].untyped_storage()._cdata for _, prior in requests
    )
    for gradient, prior in requests:
        origin = _origin(gradient)
        if origin is None or outputs.count(gradient) != 1:
            continue
        producer, index, operations, removed = origin
        rule = by_target.get(str(producer.target))
        if (
            rule is None
            or index not in rule.output_indices
            or positions[prior] >= positions[producer]
        ):
            continue
        value = prior.meta["val"]
        identity = value.untyped_storage()._cdata
        if storage[identity] != 1 or any(
            isinstance(n.meta.get("val"), torch.Tensor)
            and n.meta["val"].untyped_storage()._cdata == identity
            for n in producer.all_input_nodes
        ):
            continue
        owners[producer].append(
            (GradientDestination(index, gradient, prior, operations), removed)
        )
    consumed: dict[Node, Node] = {}
    dead: list[Node] = []
    for producer, items in owners.items():
        rule = by_target[str(producer.target)]
        candidates = {item.output_index: item for item, _ in items}
        if len(candidates) != len(items):
            continue
        context = GradientDestinationContext(
            producer, tuple(candidates.values()), fuse_narrow_gradient_accumulation
        )
        mapping = (rule.mapper or _default_mapping)(context)
        if mapping is None:
            continue
        indices = mapping.output_indices
        if (
            not indices
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)
            or set(indices) - candidates.keys()
            or len(mapping.views) != len(indices)
        ):
            raise CaptureError(
                "gradient destination mapper returned invalid output indices"
            )
        if _RESERVED & (set(mapping.keywords) | set(producer.kwargs)):
            raise CaptureError("gradient destination mapper uses reserved keywords")
        replacement = _resolve(rule.replacement)
        schema = replacement._schema
        argument = next((a for a in schema.arguments if a.name == "destinations"), None)
        if (
            argument is None
            or argument.alias_info is None
            or not argument.alias_info.is_write
        ):
            raise CaptureError(
                "gradient destination replacement must declare destinations mutable"
            )
        if any(
            item.name != "destinations"
            and item.alias_info is not None
            and item.alias_info.is_write
            for item in schema.arguments
        ):
            raise CaptureError(
                "gradient destination replacement may mutate only destinations"
            )
        # Validate every proposed writable view before changing the graph.
        prepared = []
        with fake_mode if fake_mode is not None else nullcontext():
            for index, steps in zip(indices, mapping.views, strict=True):
                prior = candidates[index].destination
                value = prior.meta["val"]
                values = []
                for step in steps:
                    value = step.target(value, *step.arguments)
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.untyped_storage()._cdata
                        != prior.meta["val"].untyped_storage()._cdata
                    ):
                        raise CaptureError(
                            "gradient destination mapping must preserve "
                            "destination storage"
                        )
                    if torch._debug_has_internal_overlap(value) == 1:
                        raise CaptureError(
                            "gradient destination mapping creates overlapping elements"
                        )
                    values.append(value)
                prepared.append(values)
        with graph.inserting_before(producer):
            destinations = []
            for index, steps, values in zip(
                indices, mapping.views, prepared, strict=True
            ):
                destination = candidates[index].destination
                for step, value in zip(steps, values, strict=True):
                    destination = graph.call_function(
                        step.target, (destination, *step.arguments)
                    )
                    _record_value(destination, value)
                destinations.append(destination)
            keywords = (
                dict(producer.kwargs)
                | dict(mapping.keywords)
                | dict(
                    destinations=destinations,
                    output_indices=list(indices),
                    contract_version=rule.version,
                )
            )
            call = graph.call_function(replacement, producer.args, keywords)
            with fake_mode if fake_mode is not None else nullcontext():
                args, kwargs = torch.fx.node.map_arg(
                    (call.args, call.kwargs), lambda n: n.meta["val"]
                )
                value = replacement(*args, **kwargs)
            before, _ = tree_flatten(producer.meta["val"])
            after, _ = tree_flatten(value)
            if len(before) != len(after):
                raise CaptureError(
                    "gradient destination replacement changed output arity"
                )
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                if index in indices:
                    destination = candidates[index].destination.meta["val"]
                    if new is not None and (
                        not isinstance(new, torch.Tensor)
                        or new.untyped_storage()._cdata
                        != destination.untyped_storage()._cdata
                    ):
                        raise CaptureError(
                            "a consumed gradient must return None or its destination"
                        )
                    continue
                if index not in indices and (
                    isinstance(old, torch.Tensor) != isinstance(new, torch.Tensor)
                    or (
                        isinstance(old, torch.Tensor)
                        and (
                            old.shape != new.shape
                            or old.dtype != new.dtype
                            or old.stride() != new.stride()
                            or old.device != new.device
                        )
                    )
                    or (not isinstance(old, torch.Tensor) and old != new)
                ):
                    raise CaptureError(
                        "gradient destination replacement changed an unconsumed output"
                    )
            call.meta = dict(producer.meta, val=value)
        producer.replace_all_uses_with(call)
        graph.erase_node(producer)
        for item, removed in items:
            if item.output_index in indices:
                consumed[item.gradient] = item.destination
                dead.extend(removed)
    return consumed, tuple(dead)


__all__ = [
    "DestinationView",
    "GradientDestination",
    "GradientDestinationContext",
    "GradientDestinationMapping",
    "GradientDestinationRule",
]
