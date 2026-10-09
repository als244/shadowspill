"""Stream pool objects into a standard torch checkpoint without a host snapshot."""

from __future__ import annotations

import ctypes
import mmap
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.representations import storage_view
from shadowspill.pytorch.spill import read_spill_tensor
from shadowspill.runtime.plan import RuntimeBridge


class PoolCheckpoint:
    """Declare CPU tensor geometry now and read its bytes when the file is ready.

    Fake tensors preserve storage aliases while torch.save reserves their file
    ranges. The mapped output then receives one pool object at a time. Flushing
    and dropping each range bounds resident file pages by the largest object.
    """

    def __init__(self, bridge: RuntimeBridge) -> None:
        self.bridge = bridge
        self.mode = FakeTensorMode()
        self.owners: dict[object, torch.Tensor] = {}
        self.readers: dict[int, Callable[[torch.Tensor], None]] = {}

    def alias(self, name: str, size: int) -> torch.Tensor:
        if name not in self.owners:
            objects = self.bridge.objects
            self._declare(
                name,
                size,
                lambda target: read_spill_tensor(objects, name, target),
            )
        return self.owners[name]

    def value(self, tensor: torch.Tensor) -> torch.Tensor:
        """Include an ordinary CPU control value alongside pool-owned objects."""
        if tensor.device.type != "cpu":
            raise RuntimeError("checkpoint tensor is not associated with a pool object")
        source = tensor.detach()
        key = ("cpu", int(source.untyped_storage()._cdata))
        if key not in self.owners:
            raw = (
                source.as_strided((0,), (1,), 0)
                .view(torch.uint8)
                .as_strided((source.untyped_storage().nbytes(),), (1,), 0)
            )

            def copy_value(target: torch.Tensor) -> None:
                target.copy_(raw)

            self._declare(key, raw.numel(), copy_value)
        return self.view(self.owners[key], tensor)

    def _declare(
        self, key: object, size: int, reader: Callable[[torch.Tensor], None]
    ) -> None:
        with self.mode:
            owner = torch.empty(size, dtype=torch.uint8, device="cpu")
        self.owners[key] = owner
        self.readers[int(owner.untyped_storage()._cdata)] = reader

    def view(self, owner: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        with self.mode:
            return storage_view(
                owner,
                template.dtype,
                template.shape,
                template.stride(),
                template.storage_offset(),
            )

    def save(self, payload: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
        """Publish atomically after every object's bytes are flushed to disk."""
        self.bridge.wait_runtime_idle()
        destination = Path(path)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(fd)
        try:
            with torch.serialization.skip_data(materialize_fake_tensors=True):
                torch.save(payload, temporary)
            with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
                mapped = torch.load(temporary, mmap=True, weights_only=True)
            declarations, structure = tree_flatten(payload)
            targets, loaded_structure = tree_flatten(mapped)
            if structure != loaded_structure:
                raise RuntimeError("checkpoint structure changed during serialization")
            copied: set[int] = set()
            for declared, target in zip(declarations, targets, strict=True):
                if not isinstance(declared, torch.Tensor):
                    continue
                identity = int(declared.untyped_storage()._cdata)
                if identity in copied:
                    continue
                copied.add(identity)
                raw = (
                    target.as_strided((0,), (1,), 0)
                    .view(torch.uint8)
                    .as_strided((target.untyped_storage().nbytes(),), (1,), 0)
                )
                self.readers[identity](raw)
                _flush_and_drop(raw)
            del mapped, targets
            with open(temporary, "rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _flush_and_drop(tensor: torch.Tensor) -> None:
    """Only called for writable file mappings owned by this checkpoint writer."""
    size = tensor.untyped_storage().nbytes()
    if not size or os.name != "posix":
        return
    page = mmap.PAGESIZE
    address = tensor.untyped_storage().data_ptr()
    start = address // page * page
    end = (address + size + page - 1) // page * page
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.msync(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 4) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Clean pages can be reclaimed immediately; tensor metadata remains valid.
    if (
        libc.madvise(
            ctypes.c_void_p(start), ctypes.c_size_t(end - start), mmap.MADV_DONTNEED
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
