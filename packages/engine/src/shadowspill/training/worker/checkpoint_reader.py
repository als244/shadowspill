"""Stage and restore immutable rank files through the existing checkpoint loader."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill_client._records import mapping
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall

if TYPE_CHECKING:
    from shadowspill.execution import Executor


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
    ) -> None:
        if type(chunk_bytes) is not int or not 0 < 2 * chunk_bytes <= budget.max_bytes:
            raise ValueError("restore chunk exceeds admitted worker staging")
        self.executor, self.budget, self.resolve = executor, budget, resolve
        self.chunk_bytes, self.world_size = chunk_bytes, world_size

    def prepare(self, call: BoundWorkerCall) -> BoundWorkerCall:
        from shadowspill.checkpoint.load import validate_checkpoint

        source = call.command.restore
        if source is None:
            raise ValueError("restore has no immutable rank input")
        info = source.info
        if info.byteorder != sys.byteorder or info.world_size != self.world_size:
            raise ValueError("restore ownership or byte order requires conversion")
        if info.frozen_state_id is not None:
            raise ValueError("worker has no declared frozen-base restore identity")
        if any(
            chunk.dtype.itemsize > self.chunk_bytes
            for chunk in self.executor.tensor_chunks()
        ):
            raise ValueError("restore staging cannot hold one tensor element per chunk")
        path = self.resolve(source.rank.archive)
        with self.budget.reserve(2 * self.chunk_bytes, timeout=0):
            manifest = validate_checkpoint(self.executor, path)
            if (
                manifest["state"] != source.rank.state_sha256
                or manifest["weights"] != info.weights
                or mapping(manifest["metadata"], "rank metadata")
                != mapping(info.rank_metadata(), "checkpoint metadata")
            ):
                raise ValueError("rank archive differs from its checkpoint generation")
        return replace(call, restore=path)

    def __call__(self, call: BoundWorkerCall) -> None:
        from shadowspill.checkpoint import load_checkpoint

        source = call.command.restore
        if source is None or call.restore is None:
            raise ValueError("restore was not validated before worker release")
        # Reconstruction can retain one cast chunk beside the destination.
        with self.budget.reserve(2 * self.chunk_bytes, timeout=0):
            metadata = load_checkpoint(
                self.executor, call.restore, chunk_bytes=self.chunk_bytes
            )
            self.executor.synchronize()
            if mapping(metadata, "restored metadata") != mapping(
                source.info.rank_metadata(), "checkpoint metadata"
            ):
                raise ValueError("checkpoint metadata changed after staging")
