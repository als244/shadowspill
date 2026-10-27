"""Native transfer-lane probes without installing the execution allocator."""

import ctypes as ct
import os
from types import TracebackType
from typing import Any, Self

from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool


def _require(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"spill calibration {operation} failed: status={status}")


class SpillProbe:
    def __init__(self, spill: SpillPool, device: int, maximum: int):
        if getattr(spill, "image", None) is not None:
            raise ValueError(
                "calibrate a scratch spill pool, not a persistent SSD image"
            )
        if maximum > spill.capacity:
            raise ValueError("calibration size exceeds the selected spill capacity")
        adapter = resolve_library("libshadowspill_pytorch.so")
        backend = resolve_library("libshadowspill_backend_cuda.so")
        if adapter is None or backend is None:
            raise RuntimeError("spill calibration requires the native CUDA libraries")
        self.library = ct.CDLL(str(adapter))
        create = self.library.shadowspill_pytorch_spill_probe_create
        create.argtypes = [
            ct.c_char_p,
            ct.c_int32,
            ct.c_char_p,
            ct.c_uint8,
            ct.c_void_p,
            ct.c_uint64,
            ct.POINTER(ct.c_void_p),
        ]
        create.restype = ct.c_uint32
        measure = self.library.shadowspill_pytorch_spill_probe_measure
        measure.argtypes = [
            ct.c_void_p,
            ct.c_uint64,
            ct.c_uint64,
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_uint64),
        ]
        measure.restype = ct.c_uint32
        destroy = self.library.shadowspill_pytorch_spill_probe_destroy
        destroy.argtypes, destroy.restype = [ct.c_void_p], None
        self.handle = ct.c_void_p()
        extension, config = spill.library, spill.configuration()
        # Two disjoint large copies plus alignment slack, not the full model pool.
        capacity = ((2 * maximum + (2 << 20) - 1) // (2 << 20) + 1) * (2 << 20)
        _require(
            create(
                os.fsencode(backend),
                device,
                None if extension is None else os.fsencode(extension),
                spill.kind,
                None if config is None else ct.byref(config),
                capacity,
                ct.byref(self.handle),
            ),
            "create",
        )

    def measure(
        self, small: int, large: int, warmup: int, count: int
    ) -> list[dict[str, Any]]:
        values = (ct.c_uint64 * 10)()
        _require(
            self.library.shadowspill_pytorch_spill_probe_measure(
                self.handle, small, large, warmup, count, values
            ),
            "measure",
        )
        return [
            dict(
                zip(
                    (
                        "latency_ns",
                        "solo_bytes_per_second",
                        "concurrent_bytes_per_second",
                        "solo_ns",
                        "concurrent_ns",
                    ),
                    values[index : index + 5],
                    strict=True,
                )
            )
            for index in (0, 5)
        ]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.library.shadowspill_pytorch_spill_probe_destroy(self.handle)
        self.handle = ct.c_void_p()
