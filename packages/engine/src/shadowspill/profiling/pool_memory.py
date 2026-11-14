"""Full-capacity resource calibration, independent of task/transfer profiling."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from shadowspill.artifacts.pool_memory import PoolMemoryProfile
from shadowspill.memory import SpillPool

from .pool_identity import deployment_identity, pool_configuration
from .pool_probe import PoolProbe


class PoolMemoryMeasurementError(RuntimeError):
    """Unresolved measurements preserve partial raw evidence for inspection."""

    def __init__(self, message: str, observations: dict[str, Any]):
        super().__init__(message)
        self.observations = observations


def measure_pool_memory(
    *, spill: SpillPool, device: int = 0, logical_rank: int = 0
) -> PoolMemoryProfile:
    """Acquire the exact spill pool and real lanes twice on an otherwise idle GPU.

    A separate 4 MiB execution control keeps device allocation independent of
    spill capacity. Synchronous observations establish retained costs only;
    startup peaks, physical pinned pages and remote host usage remain unknown.
    Use a disposable measurement process before starting a cold Executor.
    Reuse this artifact across chunk/policy/sequence candidates, then combine
    rank artifacts for DP on distinct GPUs. No distributed collectives run here.
    """
    import pynvml  # type: ignore[import-untyped]

    from shadowspill.runtime.bootstrap import installed_runtime

    if installed_runtime() is not None:
        raise ValueError("measure pool memory before installing an execution runtime")
    if type(logical_rank) is not int or logical_rank < 0:
        raise ValueError("logical_rank must be a nonnegative integer")
    pool = pool_configuration(spill)
    identity = deployment_identity(spill, device)
    rank: dict[str, Any] = {
        "logical_rank": logical_rank,
        "identity": identity,
        "method": "full_capacity_pool_and_routes_v1",
        "execution_control_bytes": 4 << 20,
        "startup_peak_bytes": None,
        "host_physical_pinned_bytes": None,
        "cycles": [],
    }
    raw = {"pool": pool, "ranks": [rank]}
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(identity["gpu_uuid"])
        with PoolProbe(spill, device) as probe:

            def samples() -> list[dict[str, Any]]:
                result = []
                for _ in range(2):
                    started = time.monotonic_ns()
                    snapshot: dict[str, Any] = probe.snapshot()
                    compute = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    processes = {
                        *[p.pid for p in compute],
                        *[
                            p.pid
                            for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(
                                handle
                            )
                        ],
                    }
                    snapshot["other_processes"] = sorted(processes - {os.getpid()})
                    snapshot["own_process_present"] = os.getpid() in processes
                    own = [p.usedGpuMemory for p in compute if p.pid == os.getpid()]
                    snapshot["own_process_memory_available"] = bool(own) and all(
                        type(v) is int and 0 <= v < 1 << 63 for v in own
                    )
                    snapshot["started_ns"] = started
                    snapshot["elapsed_ns"] = time.monotonic_ns() - started
                    host = {}
                    for line in Path("/proc/self/status").read_text().splitlines():
                        key, _, value = line.partition(":")
                        if key in {"VmRSS", "VmHWM", "VmLck", "VmPin", "VmPTE"}:
                            host[key] = int(value.split()[0]) * 1024
                    snapshot["host"] = host
                    result.append(snapshot)
                    time.sleep(0.01)
                return result

            for _ in range(2):
                cycle: dict[str, Any] = {}
                rank["cycles"].append(cycle)
                cycle["before"] = samples()
                if any(s["other_processes"] for s in cycle["before"]):
                    raise ValueError("pool calibration requires an otherwise idle GPU")
                probe.acquire()
                cycle["live"] = samples()
                probe.warmup()
                cycle["warmed"] = samples()
                probe.release()
                cycle["released"] = samples()
        return PoolMemoryProfile.from_dict(raw)
    except Exception as error:
        raise PoolMemoryMeasurementError(str(error), raw) from error
    finally:
        pynvml.nvmlShutdown()
