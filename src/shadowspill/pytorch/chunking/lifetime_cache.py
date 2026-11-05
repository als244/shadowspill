"""Build-local reuse of semantic lifetime analysis, without retaining graphs.

Only analysis is reused. Policy callbacks, partitioning, movement, replay and
compaction still resolve and validate each occurrence. Keys describe the exact
ordered graph, value trees, input geometry/backings and current effect contracts;
they deliberately exclude node names and source/module provenance. Results use
ordinals and are rebound to the requesting graph, including storage producers.
Unsupported or symbolic values take the ordinary uncached path.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.immutable_collections import immutable_dict, immutable_list

from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.task.storage import TaskStorageContract

if TYPE_CHECKING:
    from .lifetime_analysis import LifetimeGraph


class _Uncacheable(Exception):
    pass


_MISSING = object()


def _value_key(value: Any, nodes: Mapping[Node, int], *, tensors: bool) -> Any:
    if isinstance(value, Node):
        return (Node, nodes[value])
    if isinstance(value, torch.Tensor):
        if not tensors or type(value) not in (
            torch.Tensor,
            torch.nn.Parameter,
            FakeTensor,
        ):
            raise _Uncacheable
        if value.layout is not torch.strided:
            raise _Uncacheable
        shape, stride, offset = (
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
        )
        if any(type(item) is not int for item in (*shape, *stride, offset)):
            raise _Uncacheable
        return (
            torch.Tensor,
            shape,
            stride,
            offset,
            value.dtype,
            value.device,
            value.requires_grad,
            value.is_conj(),
            value.is_neg(),
        )
    if value is _MISSING:
        return _MISSING
    if value is None or type(value) in (bool, int, str):
        return (type(value), value)
    if type(value) is float:
        return (float, value.hex())
    if type(value) in (tuple, list, immutable_list):
        return (
            tuple if isinstance(value, tuple) else list,
            tuple(_value_key(v, nodes, tensors=tensors) for v in value),
        )
    if type(value) in (dict, immutable_dict):
        return (
            dict,
            tuple(
                (
                    _value_key(k, nodes, tensors=False),
                    _value_key(v, nodes, tensors=tensors),
                )
                for k, v in value.items()
            ),
        )
    if type(value) in (torch.dtype, torch.device, torch.layout, torch.memory_format):
        return (type(value), value)
    raise _Uncacheable


def analysis_key(
    graph: GraphModule,
    inputs: tuple[object, ...],
    fences: Mapping[Node, str],
    readonly: Mapping[str, str],
) -> tuple[Any, ...] | None:
    """Return a conservative in-process key; never identify real tensor values."""
    nodes = {node: index for index, node in enumerate(graph.graph.nodes)}
    aliases: dict[int, int] = {}
    arguments = []
    try:
        for value in inputs:
            # The alias catalog below covers flat tensor arguments. Containers
            # may carry other literals, but hidden tensor aliases need a richer
            # input contract before their analysis can be reused.
            item = _value_key(value, nodes, tensors=isinstance(value, torch.Tensor))
            storage_key = None
            if isinstance(value, torch.Tensor):
                size = value.untyped_storage().nbytes()
                if type(size) is not int:
                    raise _Uncacheable
                storage_key = (
                    aliases.setdefault(live_storage_identity(value), len(aliases)),
                    size,
                )
            arguments.append((item, storage_key))
        operations = tuple(
            (
                node.op,
                node.target if node.op == "call_function" else None,
                str(getattr(node.target, "_schema", "")),
                _value_key(node.args, nodes, tensors=False),
                _value_key(node.kwargs, nodes, tensors=False),
                _value_key(node.meta.get("val", _MISSING), nodes, tensors=True),
                fences.get(node),
            )
            for node in nodes
        )
    except _Uncacheable:
        return None
    return operations, tuple(arguments), tuple(sorted(readonly.items()))


@dataclass(frozen=True)
class _Record:
    storage: TaskStorageContract
    names: tuple[str, ...]
    values: tuple[frozenset[int], ...]
    reads: tuple[frozenset[int], ...]
    writes: tuple[frozenset[int], ...]
    fences: tuple[tuple[int, str], ...]
    readonly: tuple[tuple[str, str], ...]
    external_roots: frozenset[int]

    @classmethod
    def capture(cls, analysis: LifetimeGraph) -> _Record:
        return cls(
            analysis.storage,
            tuple(node.name for node in analysis.order),
            tuple(analysis.values[node] for node in analysis.order),
            tuple(analysis.reads[node] for node in analysis.order),
            tuple(analysis.writes[node] for node in analysis.order),
            tuple(
                (i, analysis.fences[n])
                for i, n in enumerate(analysis.order)
                if n in analysis.fences
            ),
            tuple(analysis.readonly_contracts.items()),
            analysis.external_roots,
        )

    def bind(self, order: tuple[Node, ...]) -> LifetimeGraph:
        from .lifetime_analysis import LifetimeGraph

        storage = self.storage
        names = tuple(node.name for node in order)
        if names != self.names:
            rename = dict(zip(self.names, names, strict=True))
            storage = TaskStorageContract.build(
                tuple(
                    replace(root, producer_node=rename[root.producer_node])
                    if root.producer_node is not None
                    else root
                    for root in storage.roots
                ),
                storage.output_views,
                tuple(
                    replace(mutation, producer_node=rename[mutation.producer_node])
                    for mutation in storage.mutations
                ),
            )
        return LifetimeGraph(
            storage,
            MappingProxyType(dict(zip(order, self.values, strict=True))),
            MappingProxyType(dict(zip(order, self.reads, strict=True))),
            MappingProxyType(dict(zip(order, self.writes, strict=True))),
            MappingProxyType({order[index]: reason for index, reason in self.fences}),
            MappingProxyType(dict(self.readonly)),
            self.external_roots,
            order,
        )


class LifetimeAnalysisCache:
    """Bounded metadata only; no tensors, FX nodes or occurrence policy objects."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("lifetime analysis cache capacity must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], _Record] = OrderedDict()
        self.hits = self.misses = self.bypasses = 0

    def get(
        self, key: tuple[Any, ...] | None, order: tuple[Node, ...]
    ) -> LifetimeGraph | None:
        if key is None:
            self.bypasses += 1
            return None
        record = self._entries.get(key)
        if record is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return record.bind(order)

    def put(self, key: tuple[Any, ...] | None, analysis: LifetimeGraph) -> None:
        if key is None:
            return
        self._entries[key] = _Record.capture(analysis)
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


current_analysis_cache: ContextVar[LifetimeAnalysisCache | None] = ContextVar(
    "shadowspill_lifetime_analysis_cache", default=None
)


@contextmanager
def lifetime_analysis_scope(
    *, max_entries: int = 128
) -> Iterator[LifetimeAnalysisCache]:
    """Share analysis during one synchronous build, releasing it on exit/error."""
    existing = current_analysis_cache.get()
    if existing is not None:
        yield existing
        return
    cache = LifetimeAnalysisCache(max_entries)
    token = current_analysis_cache.set(cache)
    try:
        yield cache
    finally:
        current_analysis_cache.reset(token)
