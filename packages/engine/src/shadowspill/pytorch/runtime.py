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
from shadowspill.runtime.core import Runtime as NeutralRuntime


class Runtime(NeutralRuntime):
    """The process's runtime, opened with PyTorch as its frontend.

    It is the neutral `~shadowspill.runtime.Runtime` with PyTorch's frontend
    supplied, for callers that drive the runtime themselves; `Executor` opens
    its own. Opening it makes the runtime PyTorch's device allocator for the
    rest of the process, so a process opens one, before PyTorch allocates any
    device memory.

    Parameters
    ----------
    pools : mapping of str to DevicePool or SpillPool
        The pools, by name: one device pool and at least one spill pool.
    routes : mapping of str to TransferRoute
        The directed routes between the device pool and the spill pools, by
        name; at least one.
    library_path : str or path-like, optional
        The framework adapter library to load. By default, the one installed
        beside the package.
    calibrate : bool, default True
        Whether to measure every route before returning.
    numa_binding : bool, default True
        Whether to place the process on the device's NUMA node. False leaves
        the caller's placement alone.
    worker_poll_nanoseconds : int, default 1000
        How long the transfer worker waits before asking again whether a
        pending completion has finished, in nanoseconds. At least zero.
    backend : str, optional
        The backend library the adapter loads, by name or path. By default,
        the one accelerator backend installed beside the libraries.

    Raises
    ------
    RuntimeConfigurationError
        If the pools or routes are invalid, the adapter library is not found, a
        runtime has already been opened in this process, a pool's image cannot
        be reserved, or calibration fails.
    TypeError
        If ``pools`` or ``routes`` is not a mapping, or holds something other
        than a pool or route configuration.
    RuntimeError
        If the allocator cannot be installed.
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
            backend=backend,
        )


__all__ = ["Runtime"]
