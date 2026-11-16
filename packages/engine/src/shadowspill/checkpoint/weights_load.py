"""Fresh model-only initialization through the ordinary Executor tensor loader."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, BinaryIO

from shadowspill.execution.distributed.checks import coordinated

from .load import _initialize
from .reading import RankReader
from .state import PlanState
from .tensors import region_bytes
from .weights_reading import read_weights

if TYPE_CHECKING:
    import torch

    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.execution.executor import Executor


def load_weights(
    executor: Executor,
    path: str | os.PathLike[str] | BinaryIO,
    *,
    chunk_bytes: int = 64 << 20,
    frozen_state_id: str | None = None,
    fill_tensor: Callable[[TensorChunk, torch.Tensor], None] | None = None,
) -> dict[str, Any]:
    """Initialize from a weights export already mapped to this worker's Plans.

    Model parameters, persistent buffers and masters use saved values. Optimizer
    state, gradients and call-local controls use the Executor's fresh defaults.
    RNG initialization remains with the worker's existing randomness owner.
    A partial export requires its exact frozen identity and a bounded base loader.

    Verification precedes mutation on every rank; existing bounded casts and
    collective replica reconstruction handle the selected representation. Payload
    memory is at most two chunks (destination plus cast scratch), in addition to
    separately admitted archive/Plan metadata and any frozen-base loader.
    """
    with ExitStack() as owned:

        def prepare() -> tuple[RankReader, frozenset[str], set[str]]:
            state = PlanState(
                executor.plan, executor.plans, logical_rank=executor.logical_rank
            )
            names = frozenset(
                item.archive.name
                for item in state.logical_tensors(
                    roles=("parameter", "buffer", "master")
                )
                if item.archive is not None
            )
            source = owned.enter_context(
                read_weights(
                    state,
                    path,
                    chunk_bytes=chunk_bytes,
                    frozen_state_id=frozen_state_id,
                )
            )
            omitted = set(source.manifest["omitted_parameters"])
            if omitted and fill_tensor is None:
                raise ValueError(
                    "partial weights require a loader for the exact frozen base"
                )
            temporary = {
                chunk.name
                for chunk, initial in state.temporary_initializers()
                if region_bytes(chunk)
                and (initial is None or initial.get("kind") == "external")
            }
            if temporary and fill_tensor is None:
                raise ValueError(
                    "weights require a temporary-state initializer: "
                    + ", ".join(sorted(temporary))
                )
            return source, names, omitted | temporary

        source, names, external = coordinated(executor._distributed, prepare)
        try:
            _initialize(
                executor,
                source._reader,
                source.manifest,
                {chunk.name: chunk for chunk in executor.tensor_chunks()},
                external,
                chunk_bytes=chunk_bytes,
                fill_tensor=fill_tensor,
                restore=names,
            )
            # A late source failure invalidates every worker before startup can
            # publish readiness. No peer is left entering a collective alone.
            coordinated(executor._distributed, owned.close)
        except BaseException:
            executor._initialized = False
            raise
        return dict(source.manifest["metadata"])
