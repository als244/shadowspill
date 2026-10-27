"""A ShadowSpill runtime, opened for PyTorch.

`shadowspill.runtime.Runtime` is framework-neutral and is opened with a
`RuntimeFrontend`. This is that call with PyTorch's frontend supplied, and is
available to low-level callers. Executor sets up the same frontend itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from shadowspill.memory import (
    MemoryPoolConfig,
    TransferRoute,
)
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.bootstrap import DEFAULT_BACKGROUND_WINDOW_BYTES
from shadowspill.runtime.core import Runtime as NeutralRuntime


class Runtime(NeutralRuntime):
    """The neutral runtime, driven by PyTorch.

    Accelerator allocator selection is process-global and irreversible after
    PyTorch initializes the accelerator, so construct exactly one of these
    before any accelerator tensor allocation when using the low-level runtime
    API. The public Executor handles this setup itself.

    By default, initialization discovers the device's host NUMA node, narrows
    existing process-thread CPU masks to permitted local CPUs, and prefers
    local pages before pinning spill pools. Remote-page fallback warns.
    ``numa_binding=False`` preserves the caller's placement. Affinity and the
    initializing thread's memory policy remain in effect for the process;
    existing allocations are not migrated.
    """

    def __init__(
        self,
        *,
        pools: Mapping[str, MemoryPoolConfig],
        routes: Mapping[str, TransferRoute],
        library_path: str | Path | None = None,
        calibrate: bool = True,
        numa_binding: bool = True,
        worker_poll_nanoseconds: int = 1_000,
        background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
        backend: str | None = None,
    ) -> None:
        super().__init__(
            frontend=PyTorchFrontend(),
            pools=pools,
            routes=routes,
            library_path=library_path,
            calibrate=calibrate,
            numa_binding=numa_binding,
            worker_poll_nanoseconds=worker_poll_nanoseconds,
            background_transfer_window_bytes=background_transfer_window_bytes,
            backend=backend,
        )


__all__ = ["Runtime"]
