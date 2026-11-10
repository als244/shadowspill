"""Build-local reuse after a region's outer axis policy has been resolved.

Partition, lifetime and chunk-policy callbacks still run for every occurrence,
including nested regions. Only the explicit, resolved tensor transform is shared.
Nested policies run after reusing compatible outer slices. The cache owns no
persistent files and is discarded after the build.
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
from .observations import SplitObservations, _rebindable
from .policy import AxisChunks
from .sources import bind_chunk_sources


def transform_key(
    artifact: GraphArtifact,
    inputs: tuple[Layout, ...],
    rows: int,
    config: AxisChunks,
    fuse_narrow_gradient_accumulation: bool,
    gradient_destination_rules: Sequence[GradientDestinationRule],
) -> tuple[Any, ...] | None:
    graph = artifact.graph_module
    key = analysis_key(graph, artifact.example_arguments, {}, {})
    if key is None:
        return None
    result: tuple[Any, ...] = (
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
        config.regions is not None,
    )
    try:
        if config.regions is not None and not _rebindable(artifact.example_arguments):
            return None
        if config.regions is not None:
            result += (tuple((n.name, str(n.target)) for n in graph.graph.nodes),)
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
        self._entries: OrderedDict[
            tuple[Any, ...], tuple[GraphArtifact, GraphArtifact]
        ] = OrderedDict()
        self.hits = self.misses = self.bypasses = 0
        self.observations = SplitObservations(max_entries)

    def get(
        self, key: tuple[Any, ...] | None, artifact: GraphArtifact, config: AxisChunks
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
        template, source = previous
        chunks = template.chunking
        if config.regions is not None:
            chunks = bind_chunk_sources(template, source, artifact, config.domain)
        return replace(
            artifact,
            compatibility_digest=template.compatibility_digest,
            chunking=chunks,
        )

    def put(
        self,
        key: tuple[Any, ...] | None,
        artifact: GraphArtifact,
        source: GraphArtifact,
    ) -> None:
        if key is None:
            return
        self._entries[key] = (artifact, source)
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.observations.entries.clear()


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
