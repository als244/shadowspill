"""SSD spill storage, owned and closed by the ordinary Runtime.

A pool is one temporary, unlinked direct-I/O file.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool

SSD_POOL_KIND = 3
#: The direct-I/O unit a chunk must be a multiple of.
DIRECT_IO_ALIGNMENT = 4096


class SSDConfiguration(ctypes.Structure):
    _fields_ = [
        ("directory", ctypes.c_char_p),
        ("staging_bytes", ctypes.c_uint64),
        ("chunk_bytes", ctypes.c_uint64),
        ("queue_depth", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class SSDPool(SpillPool):
    """A spill pool in a direct-I/O file on a local drive.

    `ssd` returns one. Constructing it checks the settings and allocates
    nothing: the runtime creates the file when it opens and releases it when
    it closes. The file is temporary, created unlinked in ``directory``, so
    it outlives neither the runtime nor the process. Bytes move through host
    staging, which the pool and each route's lane reserve from
    ``staging_bytes`` when the runtime opens.

    Parameters
    ----------
    capacity : int
        Pool size, in bytes. Positive.
    kind : int, default 3
        The runtime's ``ShadowSpillPoolKind`` value for the SSD kind.
    kind_name : str, default "ssd"
        The kind's name, as the runtime's pool registry reports it.
    directory : str or path
        Existing directory for the temporary file, on a filesystem that
        supports direct I/O and preallocation, such as ext4 on an NVMe drive.
        Stored as an absolute path. Required: the empty default is refused.
    staging_bytes : int, default 256 MiB
        Host memory the pool's staging may take, in bytes: one chunk for
        reading and writing state, and for each route ``queue_depth`` chunks
        plus one alignment unit. Opening fails if the routes need more. At
        least ``chunk_bytes`` and below 2**63.
    chunk_bytes : int, default 2 MiB
        Size of each staged piece of a transfer, in bytes: a multiple of
        4096 and of the filesystem's direct-I/O alignment, at most 1 GiB.
    queue_depth : int, default 16
        Chunks each route's lane stages at once, from 1 to 1024.

    Raises
    ------
    TypeError
        If ``capacity``, ``staging_bytes``, ``chunk_bytes`` or
        ``queue_depth`` is not an integer.
    ValueError
        If a size or count is out of range, or ``directory`` is not an
        existing directory.
    """

    directory: str | Path = ""
    staging_bytes: int = 256 << 20
    chunk_bytes: int = 2 << 20
    queue_depth: int = 16
    kind: int = SSD_POOL_KIND
    kind_name: str = "ssd"

    #: Always False: a lease's address stands for an offset in the pool's
    #: file and maps no memory. The kind's ``write`` and ``read``, and the
    #: pool's lanes, move its bytes.
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
        if self.chunk_bytes % DIRECT_IO_ALIGNMENT:
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
        """The SSD extension library, ``libshadowspill_ssd.so``.

        Raises
        ------
        RuntimeError
            If this build does not include it.
        """
        path = resolve_library("libshadowspill_ssd.so")
        if path is None:
            raise RuntimeError("This build does not include the Linux SSD extension")
        return path

    def configuration(self) -> ctypes.Structure:
        """Return what the SSD kind is told about this pool.

        Returns
        -------
        ctypes.Structure
            The ``ShadowSpillSSDConfiguration`` filled from the pool's fields.
            It lives as long as the pool.
        """
        return self._configuration

    def calibration_identity(self) -> dict[str, Any]:
        """Return the pool's configuration as stable JSON, for pool memory evidence.

        Returns
        -------
        dict
            ``directory``, ``staging_bytes``, ``chunk_bytes`` and
            ``queue_depth``.
        """
        return {
            "directory": str(self.directory),
            "staging_bytes": self.staging_bytes,
            "chunk_bytes": self.chunk_bytes,
            "queue_depth": self.queue_depth,
        }


def ssd(
    *,
    capacity: int,
    directory: str | Path,
    staging_bytes: int = 256 << 20,
    chunk_bytes: int = 2 << 20,
    queue_depth: int = 16,
) -> SSDPool:
    """Return an SSD spill pool configuration.

    Parameters
    ----------
    capacity : int
        Pool size, in bytes. Positive.
    directory : str or path
        Existing directory for the pool's temporary file, as in `SSDPool`.
    staging_bytes : int, default 256 MiB
        Host memory the pool's staging may take, in bytes.
    chunk_bytes : int, default 2 MiB
        Size of each staged piece of a transfer, in bytes.
    queue_depth : int, default 16
        Chunks each route's lane stages at once.

    Returns
    -------
    SSDPool
        The configuration.

    Raises
    ------
    TypeError
        If a size or count is not an integer.
    ValueError
        If an argument is out of range or ``directory`` is missing, as
        `SSDPool` describes.
    """
    return SSDPool(
        capacity=capacity,
        directory=directory,
        staging_bytes=staging_bytes,
        chunk_bytes=chunk_bytes,
        queue_depth=queue_depth,
    )


__all__ = ["SSDPool", "ssd"]
