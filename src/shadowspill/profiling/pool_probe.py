"""Independent native pool lifecycle for capacity-bound memory observations.

This internal probe measures resources, not an admission certificate. Its device
control pool is independent of the selected spill capacity. It never installs
the framework allocator, and leaves the backend alive after releasing the pools
so callers can distinguish released resources from retained driver state.
"""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool
from shadowspill.runtime.abi.configuration import PhysicalMemory
from shadowspill.runtime.abi.statistics import BackendStatistics


def _require(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"pool memory probe {operation} failed: status={status}")


def _positive_u64(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value < 1 << 64:
        raise ValueError(f"{name} must be a positive uint64 byte count")


class PoolProbe:
    """Caller-serialized session; close it before opening an execution worker."""

    def __init__(
        self,
        spill: SpillPool,
        device: int,
        *,
        execution_bytes: int = 4 << 20,
        disable_numa_binding: bool = False,
        backend_library: Path | None = None,
    ) -> None:
        if getattr(spill, "image", None) is not None:
            raise ValueError("probe a scratch spill pool, not a persistent SSD image")
        _positive_u64(spill.capacity, "spill.capacity")
        _positive_u64(execution_bytes, "execution_bytes")
        if type(device) is not int or not 0 <= device < 1 << 31:
            raise ValueError("device must be a nonnegative int32 ordinal")
        if type(spill.kind) is not int or not 0 <= spill.kind < 256:
            raise ValueError("spill.kind must be a uint8 pool kind")
        if not isinstance(disable_numa_binding, bool):
            raise TypeError("disable_numa_binding must be a bool")
        adapter = resolve_library("libshadowspill_pytorch.so")
        backend = backend_library or resolve_library("libshadowspill_backend_cuda.so")
        if adapter is None or backend is None:
            raise RuntimeError("pool memory probing requires native libraries")
        self.spill = spill
        self.execution_bytes = execution_bytes
        self.disable_numa_binding = disable_numa_binding
        self.library = ct.CDLL(str(adapter))
        self.handle = ct.c_void_p()
        self._acquired = False
        self._configuration = spill.configuration()
        self._bind()
        extension = spill.library
        _require(
            self.library.shadowspill_pytorch_spill_probe_open(
                os.fsencode(backend),
                device,
                None if extension is None else os.fsencode(extension),
                ct.byref(self.handle),
            ),
            "open",
        )

    def _bind(self) -> None:
        signatures = {
            "open": (
                [ct.c_char_p, ct.c_int32, ct.c_char_p, ct.POINTER(ct.c_void_p)],
                ct.c_uint32,
            ),
            "acquire": (
                [
                    ct.c_void_p,
                    ct.c_uint8,
                    ct.c_void_p,
                    ct.c_uint64,
                    ct.c_uint64,
                    ct.c_uint8,
                ],
                ct.c_uint32,
            ),
            "snapshot": (
                [
                    ct.c_void_p,
                    ct.POINTER(PhysicalMemory),
                    ct.POINTER(BackendStatistics),
                ],
                ct.c_uint32,
            ),
            "measure": (
                [
                    ct.c_void_p,
                    ct.c_uint64,
                    ct.c_uint64,
                    ct.c_uint32,
                    ct.c_uint32,
                    ct.POINTER(ct.c_uint64),
                ],
                ct.c_uint32,
            ),
            "release": ([ct.c_void_p], None),
            "destroy": ([ct.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            try:
                function = getattr(
                    self.library, f"shadowspill_pytorch_spill_probe_{name}"
                )
            except AttributeError as error:
                raise RuntimeError(
                    "native adapter lacks the pool memory probe API; rebuild the "
                    "selected native libraries"
                ) from error
            function.argtypes, function.restype = arguments, result

    def _require_open(self) -> None:
        if not self.handle:
            raise RuntimeError("pool memory probe is closed")

    def acquire(self) -> None:
        self._require_open()
        if self._acquired:
            raise RuntimeError("pool memory probe already owns pools")
        _require(
            self.library.shadowspill_pytorch_spill_probe_acquire(
                self.handle,
                self.spill.kind,
                None if self._configuration is None else ct.byref(self._configuration),
                self.execution_bytes,
                self.spill.capacity,
                self.disable_numa_binding,
            ),
            "acquire",
        )
        self._acquired = True

    def warmup(self) -> None:
        """Exercise both real routes with bounded copies, including overlap.

        These observations establish lazy route state, not transfer profiles for
        a training Plan. Production communication timing remains independent.
        """
        self._require_open()
        if not self._acquired:
            raise RuntimeError("pool memory probe has no acquired pools")
        values = (ct.c_uint64 * 10)()
        _require(
            self.library.shadowspill_pytorch_spill_probe_measure(
                self.handle, 4096, 65536, 1, 2, values
            ),
            "warmup",
        )

    def snapshot(self) -> dict[str, dict[str, int]]:
        self._require_open()
        physical, statistics = PhysicalMemory(), BackendStatistics()
        _require(
            self.library.shadowspill_pytorch_spill_probe_snapshot(
                self.handle, ct.byref(physical), ct.byref(statistics)
            ),
            "snapshot",
        )
        return {
            "physical": {
                field[0]: getattr(physical, field[0]) for field in physical._fields_
            },
            "backend": {
                field[0]: getattr(statistics, field[0]) for field in statistics._fields_
            },
        }

    def release(self) -> None:
        self._require_open()
        self.library.shadowspill_pytorch_spill_probe_release(self.handle)
        self._acquired = False

    def close(self) -> None:
        self.library.shadowspill_pytorch_spill_probe_destroy(self.handle)
        self.handle = ct.c_void_p()
        self._acquired = False

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
