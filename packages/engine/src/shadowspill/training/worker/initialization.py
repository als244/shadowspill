"""Admit installed initializers before opening sources or touching mutable state."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill.training.definitions.initialization import (
    InitializationContext,
    InitializationRuntime,
    ModelInitialization,
    TensorInitializer,
)
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import DefinitionSource

if TYPE_CHECKING:
    import torch

    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.execution.executor import Executor


@dataclass(frozen=True)
class PreparedInitialization:
    context: InitializationContext
    description: ModelInitialization
    source: DefinitionSource

    def signature(self) -> dict[str, Any]:
        """Portable startup evidence; no callback or filesystem path is serialized."""
        return {
            "logical_rank": self.context.logical_rank,
            "weights": self.context.weights.to_dict(),
            "model_source": self.source.to_dict(),
            "peak_bytes": self.description.peak_bytes,
            "chunk_bytes": self.description.chunk_bytes,
        }


def prepare_initialization(
    setup: ResolvedSetup,
) -> Mapping[int, PreparedInitialization]:
    """CPU creation preflight; inspection/preparation may omit initial weights."""
    model = setup.config.model
    if model.initialize is None:
        raise ValueError("worker creation requires explicit weights or a checkpoint")
    result = {}
    for rank in range(setup.context.parallelism.world_size):
        context = InitializationContext(
            model, model.initialize, setup.context.parallelism, rank
        )
        description = setup.model.describe_initialization(context)
        if description.peak_bytes > setup.config.execution.max_staging_bytes:
            raise ValueError(
                "model initialization exceeds the configured host staging limit"
            )
        result[rank] = PreparedInitialization(context, description, setup.model.source)
    return MappingProxyType(result)


def initialize(
    executor: "Executor",
    prepared: PreparedInitialization,
    runtime: InitializationRuntime,
    budget: BufferBudget,
) -> None:
    """Use the existing owned-region/default/master/optimizer initialization path."""
    if executor.logical_rank != prepared.context.logical_rank:
        raise ValueError("initializer belongs to another logical worker rank")
    description = prepared.description
    with (
        budget.reserve(description.peak_bytes, timeout=0),
        open_initialization(prepared, runtime) as fill,
    ):
        executor.initialize(fill, chunk_bytes=description.chunk_bytes)


@contextmanager
def open_initialization(
    prepared: PreparedInitialization, runtime: InitializationRuntime
) -> Iterator[TensorInitializer]:
    """Borrow the existing validated loader under the caller's host reservation."""
    description = prepared.description
    with description.open(runtime) as fill:
        if not callable(fill):
            raise TypeError("initialization context must yield a tensor filler")

        def fill_chunk(request: "TensorChunk", destination: "torch.Tensor") -> None:
            if (
                destination.device.type != "cpu"
                or destination.ndim != 1
                or not destination.is_contiguous()
                or destination.dtype != request.dtype
                or destination.numel() * destination.element_size()
                > description.chunk_bytes
            ):
                raise ValueError("initializer received an incompatible owned CPU chunk")
            fill(request, destination)

        yield fill_chunk
