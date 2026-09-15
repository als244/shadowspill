"""A ShadowSpill runtime, opened for PyTorch.

`shadowspill.runtime.Runtime` is framework-neutral and is opened with a
`RuntimeFrontend`. This is that call with PyTorch's frontend supplied, and is
what `plan_step`, `plan_forward` and the planned callables are given.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from shadowspill.memory import MemoryPoolConfig, TransferRoute
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.bootstrap import DEFAULT_BACKGROUND_WINDOW_BYTES
from shadowspill.runtime.core import Runtime as NeutralRuntime


class Runtime(NeutralRuntime):
    """The neutral runtime, driven by PyTorch.

    Accelerator allocator selection is process-global and irreversible after
    PyTorch initializes the accelerator, so construct exactly one of these
    before any accelerator tensor allocation, then pass it to every
    ``plan_step`` or ``plan_forward`` call.
    """

    def __init__(
        self,
        *,
        pools: Mapping[str, MemoryPoolConfig],
        routes: Mapping[str, TransferRoute],
        library_path: str | Path | None = None,
        calibrate: bool = True,
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
            worker_poll_nanoseconds=worker_poll_nanoseconds,
            background_transfer_window_bytes=background_transfer_window_bytes,
            backend=backend,
        )


__all__ = ["Runtime"]
