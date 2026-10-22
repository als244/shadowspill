"""Data-parallel configuration with process groups owned by a Trainer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.pytorch.distributed import Distributed, TransferSharding


@dataclass(frozen=True)
class DataParallel:
    """Use torchrun's ranks, creating groups when the Trainer prepares.

    Gloo coordinates pool allocation and planning before CUDA initialization;
    NCCL is created after the runtime installs its allocator. The Trainer closes
    groups it creates after releasing its runtime. An existing Gloo default group
    is borrowed. For explicit group/subgroup control, supply ``Distributed``.

    Weight/gradient transfer sharding is separate from ``Trainer.shard_optimizer``.
    Gradient reductions are sums: the caller normalizes loss by global batch size.
    """

    shard_weights: bool = False
    shard_gradients: bool = False
    symmetric_planning: bool | None = None
    timeout: float = 1800.0

    def __post_init__(self) -> None:
        if not isinstance(self.shard_weights, bool) or not isinstance(
            self.shard_gradients, bool
        ):
            raise TypeError("sharding options must be bools")
        if self.shard_gradients and not self.shard_weights:
            raise ValueError("gradient sharding requires weight transfer sharding")
        if self.symmetric_planning is not None and not isinstance(
            self.symmetric_planning, bool
        ):
            raise TypeError("symmetric_planning must be a bool or None")
        if self.shard_weights and self.symmetric_planning is False:
            raise ValueError("transfer sharding requires symmetric planning")
        if self.timeout <= 0:
            raise ValueError("distributed timeout must be positive")

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))

    @property
    def world_size(self) -> int:
        return (
            dist.get_world_size()
            if dist.is_initialized()
            else int(os.getenv("WORLD_SIZE", "1"))
        )

    def _open_control(self) -> _Groups:
        owned = not dist.is_initialized()
        if owned:
            dist.init_process_group("gloo", timeout=timedelta(seconds=self.timeout))
        elif str(dist.get_backend()).lower() != "gloo":
            raise ValueError(
                "DataParallel requires a Gloo default group before opening the runtime; "
                "use Distributed for caller-owned groups"
            )
        return _Groups(self, dist.group.WORLD, owned)


@dataclass
class _Groups:
    configuration: DataParallel
    control: Any
    owns_control: bool
    execution: Any = None

    def bind(self, device: torch.device) -> Distributed:
        self.execution = dist.new_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(seconds=self.configuration.timeout),
        )
        return Distributed(
            self.execution,
            transfer_sharding=(
                TransferSharding(self.execution, host_group=self.control)
                if self.configuration.shard_weights
                else None
            ),
            shard_gradients=self.configuration.shard_gradients,
            symmetric_planning=self.configuration.symmetric_planning,
            timeout=self.configuration.timeout,
        )

    def close(self) -> None:
        try:
            if self.execution is not None:
                dist.destroy_process_group(self.execution)
                self.execution = None
        finally:
            if self.owns_control:
                dist.destroy_process_group(self.control)
                self.owns_control = False


__all__ = ["DataParallel"]
