"""Bind the existing rank encoder to a quiescent private worker invocation."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, BinaryIO

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import RANK_MEDIA_TYPE, CheckpointInfo
from shadowspill_client.protocol.requests import ArtifactRef

from .binary import WorkerStream
from .bindings import BoundWorkerCall
from .checkpoint_records import RankSave, RankSaved

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor


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
    ) -> None:
        if type(chunk_bytes) is not int or not 0 < chunk_bytes <= budget.max_bytes:
            raise ValueError("checkpoint chunk exceeds admitted worker staging")
        self.executor, self.budget, self.stream = executor, budget, stream
        self.plans, self.chunk_bytes = plans, chunk_bytes

    def _describe(self, info: CheckpointInfo) -> RankSave:
        from shadowspill.checkpoint import checkpoint_size
        from shadowspill.checkpoint.identity import state_identity

        if info.plans != self.plans or info.byteorder != sys.byteorder:
            raise ValueError("checkpoint provenance differs from the ready worker")
        if info.frozen_state_id is not None:
            raise ValueError("worker has no declared frozen-base checkpoint identity")
        return RankSave(
            self.executor.logical_rank,
            checkpoint_size(
                self.executor, metadata=info.rank_metadata(), weights=info.weights
            ),
            state_identity(self.executor),
            self.executor.plan.program.digest,
            {name: plan.program.digest for name, plan in self.executor.plans.items()},
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
                    metadata=info.rank_metadata(),
                    chunk_bytes=self.chunk_bytes,
                    weights=info.weights,
                )
                admitted.confirm(info, manifest)
                sink.finish()
                # Send only confirmed identity fields, never the tensor inventory
                # a second time inside a bounded metadata reply.
                return RankSaved(
                    sink.reference("checkpoint-rank", RANK_MEDIA_TYPE).sha256,
                    admitted.confirmation(info),
                )
