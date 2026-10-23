"""Preparation metadata for execution objects stored as spill shards.

Only setup/checkpoint operations reconstruct tensors on the CPU. Runtime copies
use the C++ descriptor and never enter Python or the Gloo control channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn

from shadowspill.ir import TransferPhase, TransferSpec
from shadowspill.pytorch.distributed._collectives import (
    all_gather_single,
    reduce_scatter_single,
)

if TYPE_CHECKING:
    from shadowspill.runtime import Runtime

    from . import BoundDistributed
    from ._backing import TransferSharding


def collective_timing_class(
    operation: str, group_size: int, dtype: torch.dtype, replica_size: int = 1
) -> str:
    """A local cost class, independent of process-local communicator names.

    Equal-size sharding groups share a conservative overlap envelope; their
    actual isolated collective durations remain separate transfer costs.
    """
    return f"sharded:{operation}:{group_size}:{dtype}:{replica_size}"


@dataclass(frozen=True)
class ShardedStorage:
    layout: TransferSharding
    host_group: dist.ProcessGroup
    group_name: str
    execution_bytes: int
    shard_bytes: int
    offset: int
    transfer: TransferSpec
    gradient_dtype: torch.dtype | None = None
    replica_group_name: str | None = None
    logical_bytes: int = -1

    def descriptor(self) -> int:
        return int(
            torch.ops.shadowspill._create_sharded_transfer(
                self.group_name,
                torch.cuda.current_device(),
                self.execution_bytes,
                self.gradient_dtype,
                self.replica_group_name,
                self.logical_bytes,
            )
        )

    def owned(self, full: torch.Tensor) -> torch.Tensor:
        """Produce the physical shard, including deterministic tail padding."""
        shard = torch.zeros(self.shard_bytes, dtype=torch.uint8)
        length = max(0, min(self.shard_bytes, full.numel() - self.offset))
        if length:
            shard[:length].copy_(full[self.offset : self.offset + length])
        return shard

    def synchronize_owned(self, full: torch.Tensor) -> torch.Tensor:
        """Publish the host-group leader's value directly as owned shards.

        Setup only needs each rank's shard. Broadcasting the complete tensor
        to every rank first adds traffic for bytes that rank immediately drops.
        Views avoid a second full-sized send buffer; only tail padding allocates.
        """
        members = dist.get_process_group_ranks(self.host_group)
        pieces = None
        if dist.get_rank() == members[0]:
            pieces = []
            for rank in range(len(members)):
                start = rank * self.shard_bytes
                length = max(0, min(self.shard_bytes, full.numel() - start))
                if length == self.shard_bytes:
                    piece = full[start : start + length]
                else:
                    piece = torch.zeros(self.shard_bytes, dtype=torch.uint8)
                    if length:
                        piece[:length].copy_(full[start : start + length])
                pieces.append(piece)
        shard = torch.empty(self.shard_bytes, dtype=torch.uint8)
        dist.scatter(
            shard, scatter_list=pieces, src=members[0], group=self.host_group
        )
        return shard


def storage_layout(
    bound: BoundDistributed, name: str, size: int, alignment: int
) -> ShardedStorage | None:
    layout = bound.transfer_sharding.get(name)
    if layout is None or size == 0:
        return None
    count = dist.get_world_size(layout.group)
    shard = ((size + count * alignment - 1) // (count * alignment)) * alignment
    padded = shard * count
    host = layout.host_group
    if host is None:
        if tuple(dist.get_process_group_ranks(layout.group)) != bound.control.members:
            raise ValueError(
                "a sharding subgroup requires its matching Gloo host_group"
            )
        host = bound.control.group
    alias = bound.groups.source_names[layout.group.group_name]
    key = alias, padded
    if key not in bound.transfer_calibration:
        bound.transfer_calibration[key] = calibrate_gather(layout.group, host, padded)
    record = bound.transfer_calibration[key]
    return ShardedStorage(
        layout,
        host,
        alias,
        padded,
        shard,
        shard * dist.get_rank(layout.group),
        TransferSpec(
            spill_size_bytes=shard,
            fetch=(
                TransferPhase("copy", bytes=shard),
                TransferPhase(
                    "timed",
                    bytes=padded - shard,
                    duration_ns=record["duration_ns"],
                    resource="sharded-collectives",
                    submission_ordered=True,
                    timing_class=collective_timing_class("gather", count, torch.uint8),
                ),
            ),
            # Weights have already been published by the optimizer's gather.
            # Only our owned bytes must be retained on the CPU.
            evict=(TransferPhase("copy", bytes=shard),),
        ),
    )


def calibrate_gather(
    group: dist.ProcessGroup, host_group: dist.ProcessGroup, size: int
) -> dict[str, Any]:
    """Measure the in-place byte all-gather at the actual tensor size.

    Use the slowest rank for every sample. CUDA events cover completion, and
    warm-up keeps communicator creation out of the transfer estimate.
    """
    count = dist.get_world_size(group)
    if count == 1:
        return {"duration_ns": 0, "samples_ns": [], "bytes": size}
    full = torch.empty(size, dtype=torch.uint8, device="cuda")
    local = full.narrow(0, dist.get_rank(group) * (size // count), size // count)
    samples = []
    for index in range(10):
        dist.barrier(group=host_group)
        start, end = (
            torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
            torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
        )
        start.record()
        all_gather_single(full, local, group=group)
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            round(start.elapsed_time(end) * 1_000_000), dtype=torch.int64
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=host_group)
        if index >= 3:
            samples.append(int(elapsed))
    return {"duration_ns": round(median(samples)), "samples_ns": samples, "bytes": size}


def prepare_gradient_transfers(
    bound: BoundDistributed, model: nn.Module, *, grad_dtype: torch.dtype | None = None
) -> None:
    if not bound.specification.shard_gradients:
        return
    for name, parameter in model.named_parameters():
        layout = bound.transfer_sharding.get(name)
        if layout is None or not parameter.requires_grad:
            continue
        dtype = grad_dtype or parameter.dtype
        geometry = torch.empty_strided(
            parameter.shape, parameter.stride(), dtype=dtype, device="meta"
        )
        size, width = geometry.untyped_storage().nbytes(), geometry.element_size()
        count = dist.get_world_size(layout.group)
        shard = ((size + count * width - 1) // (count * width)) * width
        padded = shard * count
        host = layout.host_group or bound.control.group
        alias = bound.groups.source_names[layout.group.group_name]
        replicas = (
            None
            if layout.replica_group is None
            else bound.groups.source_names[layout.replica_group.group_name]
        )
        key = alias, padded, str(dtype), replicas
        if key not in bound.transfer_calibration:
            bound.transfer_calibration[key] = calibrate_gradient(
                layout, host, padded, dtype
            )
        record = bound.transfer_calibration[key]
        bound.gradient_transfers[name] = ShardedStorage(
            layout,
            host,
            alias,
            padded,
            shard,
            shard * dist.get_rank(layout.group),
            TransferSpec(
                spill_size_bytes=shard,
                fetch=(
                    TransferPhase(
                        "timed", duration_ns=record["zero_ns"], resource="gradient-zero"
                    ),
                    TransferPhase("copy", bytes=shard),
                ),
                evict=(
                    TransferPhase(
                        "timed",
                        bytes=padded - shard,
                        duration_ns=record["duration_ns"],
                        resource="sharded-collectives",
                        submission_ordered=True,
                        timing_class=collective_timing_class(
                            "reduce",
                            count,
                            dtype,
                            1
                            if layout.replica_group is None
                            else dist.get_world_size(layout.replica_group),
                        ),
                    ),
                    TransferPhase("copy", bytes=shard),
                ),
            ),
            dtype,
            replicas,
            size,
        )


def calibrate_gradient(
    layout: TransferSharding, host: dist.ProcessGroup, size: int, dtype: torch.dtype
) -> dict[str, Any]:
    """Time in-place reduction and accumulator initialization independently."""
    width = torch.empty((), dtype=dtype).element_size()
    full = torch.empty(size // width, dtype=dtype, device="cuda")
    count = dist.get_world_size(layout.group)
    local = full.narrow(
        0, dist.get_rank(layout.group) * (full.numel() // count), full.numel() // count
    )
    samples, zero_samples = [], []
    for index in range(10):
        dist.barrier(group=host)
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]  # type: ignore[no-untyped-call]
        start.record()
        full.zero_()
        middle.record()
        reduce_scatter_single(local, full, group=layout.group)
        if layout.replica_group is not None:
            dist.all_reduce(local, group=layout.replica_group)
        end.record()
        end.synchronize()
        times = torch.tensor(
            [
                round(start.elapsed_time(middle) * 1_000_000),
                round(middle.elapsed_time(end) * 1_000_000),
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(times, op=dist.ReduceOp.MAX, group=host)
        if index >= 3:
            zero_samples.append(int(times[0]))
            samples.append(int(times[1]))
    return {
        "duration_ns": round(median(samples)),
        "samples_ns": samples,
        "zero_ns": round(median(zero_samples)),
        "zero_samples_ns": zero_samples,
        "bytes": size,
        "dtype": str(dtype),
    }


def synchronize_state(
    bound: BoundDistributed, model: nn.Module, runtime: Runtime
) -> frozenset[str]:
    from shadowspill.pytorch.state.storage import (
        _storage_bytes,
        _write_storage_bytes,
        persistent_state,
    )

    from ._ownership import storage_parameter_names

    state = persistent_state(runtime, model)
    if state is None:
        return frozenset()
    parameters = {item.name: item for item in bound.parameters}
    parameter_names = storage_parameter_names(bound.parameters)
    skipped = set()
    for item in state.storages:
        if item.sharding is None:
            continue
        names = {
            parameter_names[view.name]
            for view in item.views
            if view.name in parameter_names
        }
        name = min(names)
        bound.control.agree("initialize/sharding", name)
        if tuple(dist.get_process_group_ranks(item.sharding.host_group)) == (
            bound.control.members
        ):
            # This one group collectively owns the model's authoritative
            # bytes. Every replica would reconstruct exactly the same value;
            # gathering it, broadcasting it, and repartitioning is redundant.
            skipped.update(names)
            continue
        full = _storage_bytes(item, runtime=runtime)
        dist.broadcast(
            full, src=min(parameters[name].replicas), group=bound.control.group
        )
        _write_storage_bytes(runtime, item, full)
        skipped.update(names)
    return frozenset(skipped)
