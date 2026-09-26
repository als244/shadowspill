"""Two ways to run a training step: plain PyTorch, and ShadowSpill.

Both build the same model from the same seed, take the same packed
microbatches and learning rate, and return one loss per microbatch, so the
trainer treats them alike. What differs is where the model lives: on the
device for PyTorch; in pinned host memory for ShadowSpill, which plans what the
device holds, fetches, evicts and recomputes so that each step fits a budget.

A step is ``max_tokens_per_step`` tokens, split into microbatches of at most
``max_tokens_per_microbatch`` -- the geometry, as (tokens per microbatch,
microbatches). PyTorch is told it; ShadowSpill searches for the fastest one at
its budget when it is not given. Either way, ``geometry`` is what the backend
runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from training.data import PackedTokens
from training.objectives import Objective

GIB = 1 << 30

Microbatch = list[torch.Tensor]


@dataclass(frozen=True)
class Setup:
    """What a backend builds a run from."""

    module: Objective  # the model and its objective, on meta
    optimizer: type[torch.optim.Optimizer]
    optimizer_args: Mapping[str, Any]
    data: PackedTokens
    max_seq_len: int
    max_tokens_per_step: int
    max_tokens_per_microbatch: int | None
    hyperparams: tuple[str, ...]  # what the step is given every time: ("lr",) or ()
    seed: int
    run_dir: Path
    artifact_store: Path  # where planning's captures, graphs, profiles and plans go


class Backend(Protocol):
    geometry: tuple[int, int]  # (tokens per microbatch, microbatches per step)
    plan: Any  # ShadowSpill's PlanSummary, or None when nothing is planned
    planning: Any  # the planned geometry and ordering, or None
    checkpoint_device: str  # where a checkpoint is mapped to be loaded

    def setup(self, setup: Setup) -> None: ...

    def step(self, microbatches: list[Microbatch], lr: float | None) -> list[float]: ...

    def synchronize(self) -> None: ...  # return once the device has finished the step

    def evaluate(self, microbatches: list[Microbatch]) -> list[float]: ...

    def device_peak_gib(self) -> float: ...

    def save(self, path: Path) -> None: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def close(self) -> None: ...
