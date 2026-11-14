"""Installed CPU initialization descriptions and bounded worker loader ownership."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from shadowspill.parallelism import Parallelism
from shadowspill_client import Model, Weights
from shadowspill_client._records import integer
from shadowspill_client.protocol.requests import identifier

if TYPE_CHECKING:
    import torch

    from shadowspill.artifacts.tensors import TensorChunk


@dataclass(frozen=True)
class InitializationContext:
    model: Model
    weights: Weights
    parallelism: Parallelism
    logical_rank: int

    def __post_init__(self) -> None:
        if self.weights != self.model.initialize:
            raise ValueError("initial weights differ from the resolved model settings")
        integer(self.logical_rank, "initialization rank")
        if self.logical_rank >= self.parallelism.world_size:
            raise ValueError("initialization rank is outside the prepared placement")
        if self.weights.kind == "artifact":
            identifier(self.weights.reference, "weights artifact ID")
        if self.weights.kind == "pretrained" and self.weights.revision is None:
            raise ValueError("pretrained initialization requires a pinned revision")


@dataclass(frozen=True)
class InitializationRuntime:
    """Worker-only access to explicit cache storage and owner-granted artifacts.

    The resolver checks ownership and immutable content before returning a local
    path. A request's artifact ID is never interpreted directly as a file path.
    Neither this context nor an open loader enters a saved Plan or wire record.
    """

    cache_root: Path
    artifact: Callable[[str], Path]


class TensorInitializer(Protocol):
    def __call__(self, request: "TensorChunk", destination: "torch.Tensor") -> None: ...


@dataclass(frozen=True)
class ModelInitialization:
    """Source-pinned maximum host footprint, separate from admitted model pools.

    ``peak_bytes`` includes retained loader state, all temporary CPU work and the
    maximum destination chunk. ``open`` runs only after reserving that bound.
    Installed definitions own initialization semantics; generic routing never
    guesses distributions, parameter names, missing buffers or weight formats.
    """

    peak_bytes: int
    chunk_bytes: int
    open: Callable[[InitializationRuntime], AbstractContextManager[TensorInitializer]]

    def __post_init__(self) -> None:
        integer(self.peak_bytes, "initialization peak bytes", minimum=1)
        integer(self.chunk_bytes, "initialization chunk bytes", minimum=1)
        if self.chunk_bytes > self.peak_bytes:
            raise ValueError("initialization peak must include its maximum chunk")
        if not callable(self.open):
            raise TypeError("initialization open must be an installed loader factory")
