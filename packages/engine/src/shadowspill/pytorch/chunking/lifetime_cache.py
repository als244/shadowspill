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

from torch.fx import GraphModule, Node

from shadowspill.pytorch.capture.structural_key import graph_input_key
from shadowspill.task.storage import TaskStorageContract

if TYPE_CHECKING:
    from .lifetime_analysis import LifetimeGraph


def analysis_key(
    graph: GraphModule,
    inputs: tuple[object, ...],
    fences: Mapping[Node, str],
    readonly: Mapping[str, str],
) -> tuple[Any, ...] | None:
    key = graph_input_key(graph, inputs)
    return (
        (
            key,
            tuple(fences.get(n) for n in graph.graph.nodes),
            tuple(sorted(readonly.items())),
        )
        if key is not None
        else None
    )


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
