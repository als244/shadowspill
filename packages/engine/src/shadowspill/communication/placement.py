"""Give a calibration process the host placement an execution runtime has.

Rates measured under another placement do not describe execution: a pinned
buffer on the far NUMA node copies at a fraction of the device-local rate, and
a distributed calibration keeps the slowest worker's rate for every worker.
"""

import ctypes as ct
import os
from pathlib import Path
from typing import Any

from shadowspill.libraries import resolve_library
from shadowspill.memory import PinnedHostPool

# The pools exist only so a runtime can be created; nothing is stored in them.
EXECUTION_BYTES = 4 << 20
SPILL_BYTES = 2 << 20

NODES = Path("/sys/devices/system/node")


def bind_to_device(device: int) -> str | None:
    """Place this process as runtime creation does; return why it could not.

    Creating a runtime is where the engine narrows every thread's CPU affinity
    to the device's host NUMA node and prefers that node's memory for the
    creating thread. A probe runtime with minimal pools performs exactly that
    initialization without installing the execution allocator. The placement
    outlives the probe, so buffers this thread pins afterwards are local to the
    device, as an execution pool's pages are.

    Never raises. A process that cannot be placed keeps its caller's placement,
    as a runtime does when it cannot bind, and the reason is returned.
    """
    adapter = resolve_library("libshadowspill_pytorch.so")
    backend = resolve_library("libshadowspill_backend_cuda.so")
    if adapter is None or backend is None:
        return "the native CUDA libraries are unavailable"
    try:
        library = ct.CDLL(str(adapter))
        open_probe = library.shadowspill_pytorch_spill_probe_open
        acquire = library.shadowspill_pytorch_spill_probe_acquire
        destroy = library.shadowspill_pytorch_spill_probe_destroy
    except (AttributeError, OSError) as error:
        return str(error)
    open_probe.argtypes = [
        ct.c_char_p,
        ct.c_int32,
        ct.c_char_p,
        ct.POINTER(ct.c_void_p),
    ]
    open_probe.restype = ct.c_uint32
    acquire.argtypes = [
        ct.c_void_p,
        ct.c_uint8,
        ct.c_void_p,
        ct.c_uint64,
        ct.c_uint64,
        ct.c_uint8,
    ]
    acquire.restype = ct.c_uint32
    destroy.argtypes, destroy.restype = [ct.c_void_p], None
    handle = ct.c_void_p()
    status = open_probe(os.fsencode(backend), device, None, ct.byref(handle))
    if status:
        return f"the placement probe could not open its backend: status={status}"
    try:
        # The last argument is disable_numa_binding: zero binds, as execution does.
        status = acquire(
            handle,
            PinnedHostPool(SPILL_BYTES).kind,
            None,
            EXECUTION_BYTES,
            SPILL_BYTES,
            0,
        )
    finally:
        destroy(handle)
    if status:
        return f"the placement probe could not create its runtime: status={status}"
    return None


def host_placement() -> dict[str, Any]:
    """The CPUs this thread may run on and the NUMA nodes that hold them."""
    cpus = os.sched_getaffinity(0)
    nodes = []
    for path in sorted(NODES.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        try:
            listed = _cpu_list((path / "cpulist").read_text())
        except (OSError, ValueError):
            continue
        if cpus & listed:
            nodes.append(int(path.name[4:]))
    return {"cpus": len(cpus), "numa_nodes": nodes}


def _cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            first, _, last = part.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
    return cpus
