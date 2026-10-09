"""Temporary SSD spill storage, owned and closed by the ordinary Runtime."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool

SSD_POOL_KIND = 3


class SSDConfiguration(ctypes.Structure):
    _fields_ = [
        ("directory", ctypes.c_char_p),
        ("staging_bytes", ctypes.c_uint64),
        ("chunk_bytes", ctypes.c_uint64),
        ("queue_depth", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class SSDPool(SpillPool):
    """Reserve a temporary direct-I/O file on the specified filesystem.

    ``capacity`` bounds SSD storage. ``staging_bytes`` separately bounds host
    payload buffers shared by imports and transfer routes. Each direction uses
    at most ``queue_depth`` chunks; increasing that depth needs more staging.
    The directory must exist. The pool file has no persistent name and is
    released by normal runtime close, including after process termination.
    Checkpoints must be stored separately.
    """

    directory: str | Path = ""
    staging_bytes: int = 256 << 20
    chunk_bytes: int = 2 << 20
    queue_depth: int = 16
    kind: int = SSD_POOL_KIND
    kind_name: str = "ssd"
    addressable: ClassVar[bool] = False
    _configuration: SSDConfiguration = field(
        default_factory=SSDConfiguration, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        SpillPool.__post_init__(self)
        if not isinstance(self.directory, (str, Path)) or not str(self.directory):
            raise ValueError("directory must name an existing SSD directory")
        directory = Path(self.directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"SSD directory does not exist: {directory}")
        for name in ("staging_bytes", "chunk_bytes", "queue_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.chunk_bytes % 4096:
            raise ValueError("chunk_bytes must be a multiple of 4096")
        if self.chunk_bytes > 1 << 30:
            raise ValueError("chunk_bytes must be at most 1 GiB")
        if self.queue_depth > 1024:
            raise ValueError("queue_depth must be at most 1024")
        if self.staging_bytes < self.chunk_bytes:
            raise ValueError("staging_bytes must hold at least one import chunk")
        if self.staging_bytes >= 1 << 63:
            raise ValueError("staging_bytes must fit a signed 64-bit byte count")
        object.__setattr__(self, "directory", directory)
        self._configuration.directory = os.fsencode(directory)
        self._configuration.staging_bytes = self.staging_bytes
        self._configuration.chunk_bytes = self.chunk_bytes
        self._configuration.queue_depth = self.queue_depth

    @property
    def library(self) -> Path:
        path = resolve_library("libshadowspill_ssd.so")
        if path is None:
            raise RuntimeError("This build does not include the Linux SSD extension")
        return path

    def configuration(self) -> ctypes.Structure:
        return self._configuration


def ssd(
    *,
    capacity: int,
    directory: str | Path,
    staging_bytes: int = 256 << 20,
    chunk_bytes: int = 2 << 20,
    queue_depth: int = 16,
) -> SSDPool:
    """Configure an SSD pool; allocation and cleanup belong to Runtime."""
    return SSDPool(
        capacity=capacity,
        directory=directory,
        staging_bytes=staging_bytes,
        chunk_bytes=chunk_bytes,
        queue_depth=queue_depth,
    )


__all__ = ["SSDPool", "ssd"]
