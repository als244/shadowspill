"""A ShadowSpill runtime, opened for PyTorch.

`shadowspill.runtime.Runtime` is framework-neutral and is opened with a
`RuntimeFrontend`. This is that call with PyTorch's frontend supplied, and is
what `plan_step`, `plan_forward` and the planned callables are given.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from .distributed import BoundDistributed, Distributed

import torch.distributed as dist

from shadowspill.memory import (
    DevicePool,
    MemoryPoolConfig,
    PinnedHostPool,
    TransferRoute,
)
from shadowspill.pytorch.accelerator import accelerator_device
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.bootstrap import DEFAULT_BACKGROUND_WINDOW_BYTES
from shadowspill.runtime.core import Runtime as NeutralRuntime


class Runtime(NeutralRuntime):
    """The neutral runtime, driven by PyTorch.

    Accelerator allocator selection is process-global and irreversible after
    PyTorch initializes the accelerator, so construct exactly one of these
    before any accelerator tensor allocation, then pass it to every
    ``plan_step`` or ``plan_forward`` call.

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
        control_group: dist.ProcessGroup | None = None,
        host_headroom_bytes: int = 2 << 30,
        preparation_timeout: float = 1800.0,
    ) -> None:
        self.control_group = control_group
        self.preparation_timeout = preparation_timeout
        self._preparation_index = 0
        self._distributed_models: weakref.WeakKeyDictionary[
            nn.Module, BoundDistributed
        ] = weakref.WeakKeyDictionary()
        self._distributed_bindings: list[BoundDistributed] = []
        if control_group is not None:
            from .distributed._bootstrap import preflight

            devices = [pool for pool in pools.values() if isinstance(pool, DevicePool)]
            if len(devices) != 1:
                raise ValueError(
                    "distributed runtime needs one execution device per process"
                )
            preflight(
                control_group,
                accelerator_device(devices[0].device),
                sum(
                    pool.capacity
                    for pool in pools.values()
                    if isinstance(pool, PinnedHostPool)
                ),
                staging_bytes=host_headroom_bytes,
                timeout=preparation_timeout,
            )
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

    def _distributed_for(
        self, model: nn.Module, specification: Distributed | None
    ) -> BoundDistributed | None:
        bound = self._distributed_models.get(model)
        if bound is not None:
            if specification is not None and specification is not bound.specification:
                raise ValueError(
                    "this model is already bound to different distributed settings"
                )
            return bound
        if specification is None:
            return None
        if self.control_group is None:
            raise ValueError(
                "construct Runtime with a Gloo control_group "
                "before distributed planning"
            )
        namespace = f"model/{self._preparation_index}"
        self._preparation_index += 1
        bound = specification._bind(model, self.control_group, namespace=namespace)
        self._distributed_models[model] = bound
        self._distributed_bindings.append(bound)
        return bound

    def close(self) -> None:
        super().close()
        for bound in reversed(self._distributed_bindings):
            bound.close()
        self._distributed_bindings.clear()
        self._distributed_models.clear()


__all__ = ["Runtime"]
