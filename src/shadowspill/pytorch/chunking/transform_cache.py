"""Build-local reuse after a leaf region's axis policy has been resolved.

Partition, lifetime and chunk-policy callbacks still run for every occurrence,
including nested regions. Only a leaf's explicit, resolved tensor transform is
shared. The cache owns no persistent files and is discarded after the build.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.gradients import GradientDestinationRule
from shadowspill.pytorch.capture.storage.cache import storage_contract_scope

from .geometry import Layout
from .lifetime_cache import analysis_key, lifetime_analysis_scope
from .policy import AxisChunks


def transform_key(
    artifact: GraphArtifact,
    inputs: tuple[Layout, ...],
    rows: int,
    config: AxisChunks,
    fuse_narrow_gradient_accumulation: bool,
    gradient_destination_rules: Sequence[GradientDestinationRule],
) -> tuple[Any, ...] | None:
    if config.regions is not None:
        return None  # Inner policies must resolve against this occurrence.
    graph = artifact.graph_module
    key = analysis_key(graph, artifact.example_arguments, {}, {})
    if key is None:
        return None
    result = (
        artifact.compatibility_digest,
        key,
        tuple(p.role for p in artifact.input_provenance),
        artifact.input_components,
        tuple(n.is_impure(impure_random=True) for n in graph.graph.nodes),
        inputs,
        rows,
        config.chunk_size,
        config.domain,
        config.rules,
        fuse_narrow_gradient_accumulation,
        tuple(gradient_destination_rules),
    )
    try:
        hash(result)
    except TypeError:
        return None  # Callable rule objects need not be hashable.
    return result


class ChunkTransformCache:
    """Bounded graph templates; occurrence inputs stay on the requesting parent."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("chunk transform cache capacity must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], GraphArtifact] = OrderedDict()
        self.hits = self.misses = self.bypasses = 0

    def get(
        self, key: tuple[Any, ...] | None, artifact: GraphArtifact
    ) -> GraphArtifact | None:
        if key is None:
            self.bypasses += 1
            return None
        previous = self._entries.get(key)
        if previous is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        # Child code is a structural template, just as it is during compile
        # deduplication. No policy receives its first occurrence's metadata:
        # nested regions never enter this cache. The parent retains the current
        # graph, source provenance, real representative values and fake inputs.
        return replace(
            artifact,
            compatibility_digest=previous.compatibility_digest,
            chunking=previous.chunking,
        )

    def put(self, key: tuple[Any, ...] | None, artifact: GraphArtifact) -> None:
        if key is None:
            return
        self._entries[key] = artifact
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


current_transform_cache: ContextVar[ChunkTransformCache | None] = ContextVar(
    "shadowspill_chunk_transform_cache", default=None
)


@contextmanager
def execution_analysis_scope(
    *, max_entries: int = 256
) -> Iterator[ChunkTransformCache]:
    """Reuse resolved analysis within a synchronous build, including inner cuts."""
    with lifetime_analysis_scope(), storage_contract_scope():
        existing = current_transform_cache.get()
        if existing is not None:
            yield existing
            return
        cache = ChunkTransformCache(max_entries)
        token = current_transform_cache.set(cache)
        try:
            yield cache
        finally:
            current_transform_cache.reset(token)
            cache.clear()
