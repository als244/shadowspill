"""Stage and restore immutable rank files through the existing checkpoint loader."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadowspill_client._records import mapping
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall
from .checkpoint_records import RankRestore

if TYPE_CHECKING:
    from shadowspill.checkpoint.state import CheckpointState
    from shadowspill.execution import Executor

    from .randomness import WorkerRandomness


def rank_randomness(
    metadata: Mapping[str, Any], source: RankRestore, *, required: bool = False
) -> Mapping[str, Any] | None:
    """Check generation metadata and the bounded, rank-owned RNG attachment."""
    from shadowspill.checkpoint.random_streams import read_snapshot

    boundary = dict(metadata)
    randomness = boundary.pop("randomness", None)
    if mapping(boundary, "rank metadata") != mapping(
        source.info.rank_metadata(), "checkpoint metadata"
    ):
        raise ValueError("rank archive differs from its checkpoint generation")
    if randomness is None:
        if required:
            raise ValueError("checkpoint has no declared worker RNG state")
        return None
    if not isinstance(randomness, Mapping):
        raise ValueError("checkpoint RNG attachment requires a snapshot")
    _, states = read_snapshot(randomness)
    if not states or any(state.stream.rank != source.rank.rank for state in states):
        raise ValueError("checkpoint RNG state belongs to another logical rank")
    return randomness


def validate_rank(
    state: CheckpointState,
    path: Path,
    source: RankRestore,
    *,
    chunk_bytes: int,
    world_size: int,
    require_randomness: bool = False,
) -> Mapping[str, Any] | None:
    """The same complete preflight for saved descriptions and live workers."""
    from shadowspill.checkpoint.load import validate_checkpoint

    info = source.info
    if info.byteorder != sys.byteorder or info.world_size != world_size:
        raise ValueError("restore ownership or byte order requires conversion")
    if info.frozen_state_id is not None:
        raise ValueError("worker has no declared frozen-base restore identity")
    if any(chunk.dtype.itemsize > chunk_bytes for chunk in state.tensor_chunks()):
        raise ValueError("restore staging cannot hold one tensor element per chunk")
    manifest = validate_checkpoint(state, path)
    if (
        manifest["state"] != source.rank.state_sha256
        or manifest["weights"] != info.weights
    ):
        raise ValueError("rank archive differs from its checkpoint generation")
    return rank_randomness(manifest["metadata"], source, required=require_randomness)


class WorkerCheckpointReader:
    """Borrow verified file grants and the worker's admitted serial staging."""

    def __init__(
        self,
        executor: Executor,
        budget: BufferBudget,
        resolve: Callable[[ArtifactRef], Path],
        *,
        chunk_bytes: int,
        world_size: int,
        randomness: WorkerRandomness | None = None,
    ) -> None:
        if type(chunk_bytes) is not int or not 0 < 2 * chunk_bytes <= budget.max_bytes:
            raise ValueError("restore chunk exceeds admitted worker staging")
        self.executor, self.budget, self.resolve = executor, budget, resolve
        self.chunk_bytes, self.world_size = chunk_bytes, world_size
        self.randomness = randomness

    def prepare(self, call: BoundWorkerCall) -> BoundWorkerCall:
        source = call.command.restore
        if source is None:
            raise ValueError("restore has no immutable rank input")
        return replace(call, restore=self.prepare_source(source))

    def prepare_source(self, source: RankRestore) -> Path:
        path = self.resolve(source.rank.archive)
        with self.budget.reserve(2 * self.chunk_bytes, timeout=0):
            randomness = validate_rank(
                self.executor,
                path,
                source,
                chunk_bytes=self.chunk_bytes,
                world_size=self.world_size,
                require_randomness=self.randomness is not None,
            )
            if self.randomness is not None:
                assert randomness is not None
                self.randomness.prepare(randomness)
        return path

    def __call__(self, call: BoundWorkerCall) -> None:
        source = call.command.restore
        if source is None or call.restore is None:
            raise ValueError("restore was not validated before worker release")
        self.load(source, call.restore)

    def load(self, source: RankRestore, path: Path) -> None:
        from shadowspill.checkpoint import load_checkpoint

        # Reconstruction can retain one cast chunk beside the destination.
        with self.budget.reserve(2 * self.chunk_bytes, timeout=0):
            metadata = load_checkpoint(
                self.executor, path, chunk_bytes=self.chunk_bytes
            )
            self.executor.synchronize()
            randomness = rank_randomness(
                metadata, source, required=self.randomness is not None
            )
            if self.randomness is not None:
                assert randomness is not None
                self.randomness.restore(randomness)
