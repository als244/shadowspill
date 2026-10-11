"""SSD spill storage, owned and closed by the ordinary Runtime.

A pool is one direct-I/O file. By default that file is temporary and unlinked.
With an image it is persistent: a pool created with ``create_image=True`` *is*
the named image file, so state imported into it outlives the process; a pool
opened on an existing image reads that file in place as its first bytes and
keeps everything new in a temporary scratch file behind it.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool

SSD_POOL_KIND = 3
IMAGE_NONE, IMAGE_CREATE, IMAGE_ADOPT = 0, 1, 2
IMAGE_ALIGNMENT = 4096


class SSDConfiguration(ctypes.Structure):
    _fields_ = [
        ("directory", ctypes.c_char_p),
        ("staging_bytes", ctypes.c_uint64),
        ("chunk_bytes", ctypes.c_uint64),
        ("queue_depth", ctypes.c_uint32),
        ("image_mode", ctypes.c_uint32),
        ("image_path", ctypes.c_char_p),
    ]


@dataclass(frozen=True, slots=True)
class SSDPool(SpillPool):
    """Reserve a direct-I/O file on the specified filesystem.

    ``capacity`` bounds SSD storage. ``staging_bytes`` separately bounds host
    payload buffers shared by imports and transfer routes. Each direction uses
    at most ``queue_depth`` chunks; increasing that depth needs more staging.
    The directory must exist and holds the temporary file, which has no
    persistent name and is released by normal runtime close, including after
    process termination.

    ``image`` names a pool image. With ``create_image`` the pool is that file,
    created here and kept at release, for a conversion that imports frozen
    state once. Without it the existing image is the pool's first
    ``image_bytes`` bytes, read in place and never written, and the rest of
    the capacity is the temporary file. Checkpoints are stored separately.
    """

    directory: str | Path = ""
    staging_bytes: int = 256 << 20
    chunk_bytes: int = 2 << 20
    queue_depth: int = 16
    image: str | Path | None = None
    create_image: bool = False
    image_bytes: int = 0
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
        if self.chunk_bytes % IMAGE_ALIGNMENT:
            raise ValueError("chunk_bytes must be a multiple of 4096")
        if self.chunk_bytes > 1 << 30:
            raise ValueError("chunk_bytes must be at most 1 GiB")
        if self.queue_depth > 1024:
            raise ValueError("queue_depth must be at most 1024")
        if self.staging_bytes < self.chunk_bytes:
            raise ValueError("staging_bytes must hold at least one import chunk")
        if self.staging_bytes >= 1 << 63:
            raise ValueError("staging_bytes must fit a signed 64-bit byte count")
        mode, image_bytes, image = IMAGE_NONE, 0, None
        if self.image is not None:
            image = Path(self.image).expanduser().resolve()
            if self.create_image:
                if image.exists():
                    raise ValueError(f"pool image already exists: {image}")
                if not image.parent.is_dir():
                    raise ValueError(
                        f"pool image directory does not exist: {image.parent}"
                    )
                mode = IMAGE_CREATE
            else:
                if not image.is_file():
                    raise ValueError(f"pool image does not exist: {image}")
                image_bytes = image.stat().st_size
                if image_bytes == 0 or image_bytes % IMAGE_ALIGNMENT:
                    raise ValueError(
                        f"pool image {image} holds {image_bytes} bytes; an image is a "
                        "positive multiple of 4096 bytes"
                    )
                if image_bytes > self.capacity:
                    raise ValueError(
                        f"pool image {image} holds {image_bytes} bytes, more than the "
                        f"capacity of {self.capacity}"
                    )
                mode = IMAGE_ADOPT
        elif self.create_image:
            raise ValueError("create_image needs the image path to create")
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "image_bytes", image_bytes)
        self._configuration.directory = os.fsencode(directory)
        self._configuration.staging_bytes = self.staging_bytes
        self._configuration.chunk_bytes = self.chunk_bytes
        self._configuration.queue_depth = self.queue_depth
        self._configuration.image_mode = mode
        self._configuration.image_path = None if image is None else os.fsencode(image)

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
    image: str | Path | None = None,
    create_image: bool = False,
) -> SSDPool:
    """Configure an SSD pool; allocation and cleanup belong to Runtime."""
    return SSDPool(
        capacity=capacity,
        directory=directory,
        staging_bytes=staging_bytes,
        chunk_bytes=chunk_bytes,
        queue_depth=queue_depth,
        image=image,
        create_image=create_image,
    )


__all__ = ["SSDPool", "ssd"]
