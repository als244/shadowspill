"""Bounded host access to spill leases; backend addresses never escape as tensors."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import cast

import torch

from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.plan.common import require_status

CHUNK_BYTES = 64 << 20


class SpillStorage:
    def __init__(self, bridge: RuntimeBridge, sizes: Mapping[str, int]):
        self.bridge = bridge
        self.objects = bridge.objects
        self.sizes = dict(sizes)
        self.addressable = next(
            pool.addressable
            for pool in bridge.runtime.pools.values()
            if pool.pool_id == bridge.spill_pool_id
        )

    @contextmanager
    def buffer(
        self,
        alias: str,
        start: int,
        size: int,
        *,
        read: bool = True,
        write: bool = False,
    ) -> Iterator[torch.Tensor]:
        if start < 0 or size < 0 or start + size > self.sizes[alias]:
            raise ValueError("spill range lies outside its physical lease")
        if self.addressable:
            location = self.objects.spill_location(alias)
            if location is None:
                raise RuntimeError(f"no current spill lease for {alias}")
            address, capacity = location
            owner = cast(
                torch.Tensor,
                torch.ops.shadowspill._make_runtime_cpu_storage(
                    torch.empty(0, dtype=torch.uint8),
                    self.bridge.spill_pool_id,
                    address,
                    self.objects.runtime_object_id(alias),
                    capacity,
                ),
            )
            yield owner[start : start + size]
        else:
            value = torch.empty(size, dtype=torch.uint8, device="cpu")
            if read:
                self._transfer(alias, value, start, "read")
            yield value
            if write:
                self._transfer(alias, value, start, "write")

    def _transfer(
        self, alias: str, value: torch.Tensor, start: int, direction: str
    ) -> None:
        require_status(
            self.bridge.library,
            getattr(
                self.bridge.runtime_library, f"shadowspill_{direction}_object_range"
            )(
                self.bridge.runtime._runtime_handle,
                self.objects.runtime_object_id(alias),
                self.bridge.spill_pool_id,
                value.data_ptr(),
                start,
                value.nbytes,
            ),
            f"{direction} spill range for {alias}",
        )

    def chunks(
        self,
        alias: str,
        *,
        dtype: torch.dtype = torch.uint8,
        start: int = 0,
        length: int | None = None,
        chunk_bytes: int = CHUNK_BYTES,
        read: bool = True,
        write: bool = False,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Element offsets into the local lease, with at most one staging chunk."""
        if chunk_bytes < dtype.itemsize:
            raise ValueError("chunk_bytes must hold at least one tensor element")
        length = (
            self.sizes[alias] // dtype.itemsize - start if length is None else length
        )
        if (
            start < 0
            or length < 0
            or (start + length) * dtype.itemsize > self.sizes[alias]
        ):
            raise ValueError("spill range lies outside its physical lease")
        width = chunk_bytes // dtype.itemsize
        for offset in range(start, start + length, width):
            count = min(width, start + length - offset)
            with self.buffer(
                alias,
                offset * dtype.itemsize,
                count * dtype.itemsize,
                read=read,
                write=write,
            ) as value:
                yield offset, value.view(dtype)

    def copy_from(self, alias: str, value: torch.Tensor) -> None:
        if (
            value.ndim != 1
            or not value.is_contiguous()
            or value.nbytes != self.sizes[alias]
        ):
            raise ValueError("input storage differs from the declared spill payload")
        for offset, destination in self.chunks(
            alias, dtype=value.dtype, read=False, write=True
        ):
            destination.copy_(value[offset : offset + destination.numel()])
