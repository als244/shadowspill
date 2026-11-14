"""Bounded host reconstruction of replicas from collectively owned shards."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.parallelism import GroupRef

from .distributed.checks import coordinated
from .distributed.worker import DistributedWorker
from .storage import Storage
from .tensors import OwnedTensor, owned_tensors


@dataclass(frozen=True)
class _Copy:
    target_storage: Storage
    target: OwnedTensor
    source_storage: Storage
    source: OwnedTensor
    members: tuple[int, ...]
    contract: tuple[Any, ...]


def _prepare(
    storages: Mapping[str, Storage],
    copies: Mapping[str, tuple[str, GroupRef]],
    logical_rank: int,
    chunk_bytes: int,
) -> dict[str, _Copy]:
    tensors = {
        prefix + tensor.name: (state, tensor)
        for prefix, state in storages.items()
        for tensor in owned_tensors(state, skip_shared=bool(prefix))
    }
    result = {}
    for name, (source_name, reference) in sorted(copies.items()):
        target_state, target = tensors[name]
        source_state, source = tensors[source_name]
        source_request, target_request = source.chunk(0), target.chunk(0)
        members = reference.members(logical_rank)
        total = source_request.storage_elements
        capacity = (total + len(members) - 1) // len(members)
        offset = min(capacity * members.index(logical_rank), total)
        if (
            (target.alias == source.alias and target_state is source_state)
            or target.offset != 0
            or target.length != total
            or source.offset != offset
            or source.length != min(capacity, total - offset)
            or not source.dtype.is_floating_point
            or not target.dtype.is_floating_point
            or any(
                getattr(source_request, key) != getattr(target_request, key)
                for key in ("shape", "stride", "storage_offset")
            )
            or chunk_bytes < max(source.dtype.itemsize, target.dtype.itemsize)
        ):
            raise ValueError(f"invalid replicated tensor reconstruction: {name}")
        result[name] = _Copy(
            target_state,
            target,
            source_state,
            source,
            members,
            (
                name,
                source_name,
                members,
                total,
                str(source.dtype),
                str(target.dtype),
                source_request.shape,
                source_request.stride,
                chunk_bytes,
            ),
        )
    return result


def _check(group: dist.ProcessGroup, error: Exception | None) -> None:
    errors: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, None if error is None else str(error), group=group)
    if any(item is not None for item in errors):
        raise RuntimeError(
            f"distributed tensor reconstruction failed: {errors}"
        ) from error


def reconstruct_replicas(
    storages: Mapping[str, Storage],
    copies: Mapping[str, tuple[str, GroupRef]],
    worker: DistributedWorker | None,
    chunk_bytes: int,
) -> None:
    """Read existing host shards, cast and broadcast bounded pieces to replicas.

    Archives and model names are absent from this path. The caller has completed
    coordinated local initialization before entering it. No GPU scratch, update
    task, optimizer step or model-sized staging tensor is needed.
    """
    if worker is None:
        if copies:
            raise ValueError(
                "replicated tensor reconstruction requires distributed workers"
            )
        return
    groups = worker.groups
    local = coordinated(
        worker, lambda: _prepare(storages, copies, groups.logical_rank, chunk_bytes)
    )
    descriptions = groups.gather([item.contract for item in local.values()])
    contracts = sorted({contract for rows in descriptions for contract in rows})
    for contract in contracts:
        members = contract[2]
        for launcher in groups.assignment.workers(members):
            if contract not in descriptions[launcher]:
                raise ValueError("workers disagree on replicated tensor reconstruction")
    if not contracts:
        return
    groups.bind_host_groups([contract[2] for contract in contracts])
    for contract in contracts:
        if groups.logical_rank not in contract[2]:
            continue
        item = local[contract[0]]
        group = groups.host_group(item.members)
        total = contract[3]
        capacity = (total + len(item.members) - 1) // len(item.members)
        width = chunk_bytes // max(
            item.source.dtype.itemsize, item.target.dtype.itemsize
        )
        buffer = None
        error = None
        try:
            buffer = torch.empty(
                min(width, total), dtype=item.target.dtype, device="cpu"
            )
        except Exception as failure:
            error = failure
        _check(group, error)
        assert buffer is not None
        for index, owner in enumerate(item.members):
            begin = min(index * capacity, total)
            end = min(begin + capacity, total)
            for start in range(begin, end, width):
                count = min(width, end - start)
                value = buffer[:count]
                error = None
                try:
                    if groups.logical_rank == owner:
                        with item.source_storage.spill.buffer(
                            item.source.alias,
                            (start - begin) * item.source.dtype.itemsize,
                            count * item.source.dtype.itemsize,
                        ) as source:
                            value.copy_(source.view(item.source.dtype))
                        del source
                except Exception as failure:
                    error = failure
                _check(group, error)
                dist.broadcast(
                    value.view(torch.uint8),
                    src=groups.assignment.workers((owner,))[0],
                    group=group,
                )
                error = None
                try:
                    with item.target_storage.spill.buffer(
                        item.target.alias,
                        start * item.target.dtype.itemsize,
                        count * item.target.dtype.itemsize,
                        read=False,
                        write=True,
                    ) as target:
                        target.view(item.target.dtype).copy_(value)
                    del target
                except Exception as failure:
                    error = failure
                _check(group, error)
