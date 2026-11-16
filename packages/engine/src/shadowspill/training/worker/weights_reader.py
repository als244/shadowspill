"""Admitted exported weights initialize model values with fresh training state."""

from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill_client._records import mapping
from shadowspill_client.buffers import BufferBudget

from .checkpoint_records import RankWeights
from .initialization import (
    PreparedInitialization,
    open_initialization,
    restore_staging,
)

if TYPE_CHECKING:
    from shadowspill.checkpoint.state import CheckpointState
    from shadowspill.execution.executor import Executor
    from shadowspill.training.definitions import InitializationRuntime


def validate_rank(
    state: CheckpointState,
    path: Path,
    source: RankWeights,
    *,
    chunk_bytes: int,
    world_size: int,
) -> None:
    """Verify the complete archive before any worker enters initialization."""
    from shadowspill.checkpoint.weights_reading import read_weights

    info = source.info
    if info.byteorder != sys.byteorder or info.world_size != world_size:
        raise ValueError("weights ownership or byte order requires conversion")
    if info.frozen_state_id is not None and info.frozen_base is None:
        raise ValueError("worker has no declared frozen-base initialization identity")
    if any(chunk.dtype.itemsize > chunk_bytes for chunk in state.tensor_chunks()):
        raise ValueError("weights staging cannot hold one tensor element per chunk")
    with read_weights(
        state, path, chunk_bytes=chunk_bytes, frozen_state_id=info.frozen_state_id
    ) as reader:
        if (
            reader.manifest["state"] != source.rank.state_sha256
            or reader.manifest["weights"] != info.weights
            or mapping(reader.manifest["metadata"], "rank metadata")
            != mapping(info.rank_metadata(), "weights metadata")
        ):
            raise ValueError("rank archive differs from its weights generation")


def initialize_weights(
    executor: Executor,
    path: Path,
    source: RankWeights,
    initial: PreparedInitialization | None,
    runtime: InitializationRuntime,
    budget: BufferBudget,
    *,
    chunk_bytes: int,
) -> None:
    """Use the shared loader; optimizer defaults and worker RNG stay fresh."""
    from shadowspill.checkpoint.weights_load import load_weights

    chunk, peak = restore_staging(initial, budget.max_bytes, limit=chunk_bytes)
    with ExitStack() as cleanup:
        cleanup.enter_context(budget.reserve(peak, timeout=0))
        fill = (
            None
            if initial is None
            else cleanup.enter_context(open_initialization(initial, runtime))
        )
        metadata = load_weights(
            executor,
            path,
            chunk_bytes=chunk,
            frozen_state_id=source.info.frozen_state_id,
            fill_tensor=fill,
        )
        if mapping(metadata, "rank metadata") != mapping(
            source.info.rank_metadata(), "weights metadata"
        ):
            executor._initialized = False
            raise ValueError("rank archive differs from its weights generation")
        executor.synchronize()
