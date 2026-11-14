"""Bounded, build-local storage evidence for identical explicit tensor graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from torch.fx import GraphModule

from shadowspill.task.storage import TaskStorageContract

from ..structural_key import graph_input_key


@dataclass(frozen=True)
class _Record:
    contract: TaskStorageContract
    names: tuple[str, ...]

    def bind(self, graph: GraphModule) -> TaskStorageContract:
        names = tuple(node.name for node in graph.graph.nodes)
        if names == self.names:
            return self.contract
        rename = dict(zip(self.names, names, strict=True))
        return TaskStorageContract.build(
            tuple(
                replace(root, producer_node=rename[root.producer_node])
                if root.producer_node is not None
                else root
                for root in self.contract.roots
            ),
            self.contract.output_views,
            tuple(
                replace(m, producer_node=rename[m.producer_node])
                for m in self.contract.mutations
            ),
        )


class StorageContractCache:
    """Cache metadata only; roots and mutations bind to the requesting graph."""

    def __init__(self, max_entries: int = 512) -> None:
        if max_entries < 1:
            raise ValueError("storage contract cache capacity must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], _Record] = OrderedDict()
        self.hits = self.misses = self.bypasses = 0

    def get(
        self, key: tuple[Any, ...] | None, graph: GraphModule
    ) -> TaskStorageContract | None:
        if key is None:
            self.bypasses += 1
            return None
        record = self._entries.get(key)
        if record is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return record.bind(graph)

    def put(
        self,
        key: tuple[Any, ...] | None,
        graph: GraphModule,
        contract: TaskStorageContract,
    ) -> None:
        if key is None:
            return
        self._entries[key] = _Record(
            contract, tuple(node.name for node in graph.graph.nodes)
        )
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


current_storage_cache: ContextVar[StorageContractCache | None] = ContextVar(
    "shadowspill_storage_contract_cache", default=None
)


@contextmanager
def storage_contract_scope(*, max_entries: int = 512) -> Iterator[StorageContractCache]:
    existing = current_storage_cache.get()
    if existing is not None:
        yield existing
        return
    cache = StorageContractCache(max_entries)
    token = current_storage_cache.set(cache)
    try:
        yield cache
    finally:
        current_storage_cache.reset(token)
        cache._entries.clear()


__all__ = ["current_storage_cache", "graph_input_key", "storage_contract_scope"]
