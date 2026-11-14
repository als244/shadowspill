"""Pool deployment identities without initializing a framework allocator."""

from __future__ import annotations

import ctypes as ct
import hashlib
import socket
import uuid
from typing import Any

from shadowspill.artifacts._values import freeze, plain
from shadowspill.libraries import resolve_library
from shadowspill.memory import SpillPool


def pool_configuration(spill: SpillPool) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": spill.kind,
        "kind_name": spill.kind_name,
        "capacity": spill.capacity,
        "configuration": plain(freeze(spill.calibration_identity())),
        "numa_binding": True,
    }


def device_uuid(device: int) -> str:
    """Use CUDA's process-local ordinal mapping, including visibility masks."""
    if type(device) is not int or device < 0:
        raise ValueError("device must be a nonnegative ordinal")
    driver = ct.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ct.c_uint]
    driver.cuInit.restype = ct.c_int
    driver.cuDeviceGet.argtypes = [ct.POINTER(ct.c_int), ct.c_int]
    driver.cuDeviceGet.restype = ct.c_int
    driver.cuDeviceGetUuid_v2.argtypes = [ct.c_void_p, ct.c_int]
    driver.cuDeviceGetUuid_v2.restype = ct.c_int
    ordinal, identifier = ct.c_int(), (ct.c_ubyte * 16)()
    if (
        driver.cuInit(0)
        or driver.cuDeviceGet(ct.byref(ordinal), device)
        or driver.cuDeviceGetUuid_v2(ct.byref(identifier), ordinal)
    ):
        raise RuntimeError("could not resolve CUDA device identity")
    return "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))


def deployment_identity(spill: SpillPool, device: int) -> dict[str, Any]:
    import pynvml  # type: ignore[import-untyped]

    identifier = device_uuid(device)
    libraries = {}
    for name in (
        "libshadowspill.so",
        "libshadowspill_pytorch.so",
        "libshadowspill_backend_cuda.so",
    ):
        path = resolve_library(name)
        if path is None:
            raise RuntimeError(f"pool memory requires {name}")
        libraries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if spill.library is not None:
        libraries[spill.library.name] = hashlib.sha256(
            spill.library.read_bytes()
        ).hexdigest()
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
        return {
            "gpu_uuid": identifier,
            "gpu_name": pynvml.nvmlDeviceGetName(handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "host": socket.gethostname(),
            "libraries": libraries,
        }
    finally:
        pynvml.nvmlShutdown()
