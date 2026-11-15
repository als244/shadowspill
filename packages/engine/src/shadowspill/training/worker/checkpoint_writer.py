"""Bind the existing rank encoder to a quiescent private worker invocation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, BinaryIO

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import RANK_MEDIA_TYPE, SaveInfo
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef

from .binary import WorkerStream
from .bindings import BoundWorkerCall
from .checkpoint_records import RankSave, RankSaved

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor

    from .randomness import WorkerRandomness


class WorkerCheckpointWriter:
    """Borrow initialized state; never build, initialize, publish or advance it.

    Tensor copies borrow the same serial staging budget as ordinary execution.
    Rank descriptions/manifest encoding use the deployment's decoded-artifact
    allowance. The private stream owns framing, not the encoder or its buffers.
    """

    def __init__(
        self,
        executor: Executor,
        budget: BufferBudget,
        stream: BinaryIO,
        *,
        plans: ArtifactRef,
        chunk_bytes: int,
        randomness: WorkerRandomness | None = None,
        frozen_base: Callable[[], FrozenBase | None] = lambda: None,
    ) -> None:
        if type(chunk_bytes) is not int or not 0 < chunk_bytes <= budget.max_bytes:
            raise ValueError("checkpoint chunk exceeds admitted worker staging")
        self.executor, self.budget, self.stream = executor, budget, stream
        self.plans, self.chunk_bytes = plans, chunk_bytes
        self.randomness = randomness
        self.frozen_base = frozen_base

    def _describe(self, info: SaveInfo) -> RankSave:
        from shadowspill.checkpoint import checkpoint_size
        from shadowspill.checkpoint.identity import state_identity

        if info.plans != self.plans or info.byteorder != sys.byteorder:
            raise ValueError("checkpoint provenance differs from the ready worker")
        if info.frozen_base != self.frozen_base() or (
            info.frozen_state_id is not None and info.frozen_base is None
        ):
            raise ValueError(
                "checkpoint frozen base differs from the initialized worker"
            )
        randomness = None if self.randomness is None else self.randomness.capture()
        metadata = info.rank_metadata()
        if randomness is not None:
            metadata["randomness"] = randomness
        return RankSave(
            self.executor.logical_rank,
            checkpoint_size(
                self.executor,
                metadata=metadata,
                weights=info.weights,
                frozen_state_id=info.frozen_state_id,
            ),
            state_identity(self.executor),
            self.executor.plan.program.digest,
            {name: plan.program.digest for name, plan in self.executor.plans.items()},
            randomness,
        )

    def prepare(self, call: BoundWorkerCall) -> BoundWorkerCall:
        info = call.command.checkpoint
        if info is None:
            raise ValueError("checkpoint has no coordinator boundary")
        with self.budget.reserve(self.chunk_bytes, timeout=0):
            return replace(call, checkpoint=self._describe(info))

    def __call__(self, call: BoundWorkerCall) -> RankSaved:
        from shadowspill.checkpoint import write_checkpoint

        info, admitted = call.command.checkpoint, call.checkpoint
        if info is None or admitted is None:
            raise ValueError("checkpoint encoder was not prepared before release")
        with self.budget.reserve(self.chunk_bytes, timeout=0):
            if self._describe(info) != admitted:
                raise ValueError("checkpoint state description changed after staging")
            with WorkerStream(self.stream, size_bytes=admitted.size_bytes) as sink:
                manifest = write_checkpoint(
                    self.executor,
                    sink,
                    metadata=admitted.metadata(info),
                    chunk_bytes=self.chunk_bytes,
                    weights=info.weights,
                    frozen_state_id=info.frozen_state_id,
                )
                admitted.confirm(info, manifest)
                sink.finish()
                # Send only confirmed identity fields, never the tensor inventory
                # a second time inside a bounded metadata reply.
                return RankSaved(
                    sink.reference("checkpoint-rank", RANK_MEDIA_TYPE).sha256,
                    admitted.confirmation(info),
                )
