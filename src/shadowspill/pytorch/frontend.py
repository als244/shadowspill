"""PyTorch, as the one `RuntimeFrontend` a runtime is opened with.

The twelve methods the neutral runtime asks for, on one object, because the
runtime holds one. The device questions are answered here; the two walks over
live tensors are `bindings`, which is a subject of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from shadowspill.pytorch.accelerator import accelerator_device, is_accelerator
from shadowspill.pytorch.bindings import dematerialize, occupants, retainers
from shadowspill.runtime.bootstrap import RuntimeInstallError
from shadowspill.runtime.configuration import RuntimeConfigurationError
from shadowspill.runtime.occupancy import PoolAllocation

_NOT_A_DEVICE: Final = (
    "execution_device must be an accelerator device, ordinal, or None"
)

#: The storage operations the adapter library registers with PyTorch. A library
#: that registered none of them is the usual symptom of a stale build.
REQUIRED_STORAGE_OPERATIONS: Final = (
    "_import_cpu_storages",
    "_export_cpu_storages",
    "_make_runtime_cpu_storage",
    "_acquire_storages",
    "_before_task_storages",
    "_dematerialize_storages",
    "_after_task_storages",
    "_transfer_acquired_storage_to_caller",
)

_WARMUP_SHAPE: Final = (2048, 2048)


class PyTorchFrontend:
    """PyTorch's answers to everything the runtime cannot do for itself."""

    __slots__ = ("_allocator", "_provider")

    def __init__(self) -> None:
        self._allocator: Any = None
        self._provider: Any = None

    # The device

    def current_device_ordinal(self) -> int:
        return int(torch.cuda.current_device())

    def device_ordinal(self, value: object) -> int:
        if isinstance(value, bool):
            raise TypeError(_NOT_A_DEVICE)
        if isinstance(value, int):
            device = accelerator_device(value)
        else:
            if not isinstance(value, (str, torch.device)):
                raise TypeError(_NOT_A_DEVICE)
            try:
                device = torch.device(value)
            except (TypeError, RuntimeError) as exc:
                raise TypeError(_NOT_A_DEVICE) from exc
        if not is_accelerator(device):
            raise RuntimeConfigurationError(
                "the installed PyTorch adapter currently requires an accelerator "
                "execution device"
            )
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)

    def select_device(self, ordinal: int) -> None:
        torch.cuda.set_device(ordinal)

    def synchronize(self, ordinal: int) -> None:
        torch.cuda.synchronize(ordinal)

    # The process allocator

    def refuse_unusable_build(self) -> None:
        if torch.version.cuda is None:
            raise RuntimeInstallError("a CUDA-enabled PyTorch build is required")
        provider: Any = torch.cuda
        if provider.is_initialized():
            raise RuntimeInstallError(
                "PyTorch CUDA was initialized before ShadowSpill allocator installation"
            )
        self._provider = provider

    def missing_operations(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in REQUIRED_STORAGE_OPERATIONS
            if not hasattr(torch.ops.shadowspill, name)
        )

    def prepare_allocator(self, library_path: Path, record_stream_pointer: int) -> None:
        allocator = self._provider.memory.CUDAPluggableAllocator(
            str(library_path),
            "shadowspill_pytorch_backend_malloc",
            "shadowspill_pytorch_backend_free",
        )
        set_record_stream = getattr(allocator.allocator(), "set_record_stream_fn", None)
        if set_record_stream is None or not callable(set_record_stream):
            raise RuntimeInstallError(
                "this PyTorch build lacks the required record-stream callback"
            )
        set_record_stream(record_stream_pointer)
        self._allocator = allocator

    def activate_allocator(self) -> None:
        self._provider.memory.change_current_allocator(self._allocator)

    def initialize_provider_workspaces(self, device_ordinal: int) -> None:
        """Create cuBLAS's retained state now, while the pool is empty.

        PyTorch obtains its cuBLAS handle lazily, and obtaining it does not force
        cuBLAS to create its retained workspace. A first GEMM later, while a
        large profiling input is live, can split an otherwise empty slab and
        prevent a large fixed-layout arena from being reserved despite ample
        aggregate capacity. Exercising the provider here gives its retained state
        deterministic low-address placement.
        """

        get_handle = getattr(torch._C, "_cuda_getCurrentBlasHandle", None)
        if get_handle is None or not callable(get_handle):
            raise RuntimeInstallError(
                "this PyTorch build lacks the required CUDA provider initializer"
            )
        torch.cuda.set_device(device_ordinal)
        get_handle()

        device = accelerator_device(device_ordinal)
        left = torch.empty(_WARMUP_SHAPE, dtype=torch.bfloat16, device=device)
        right = torch.empty(_WARMUP_SHAPE, dtype=torch.bfloat16, device=device)
        output = torch.empty(_WARMUP_SHAPE, dtype=torch.bfloat16, device=device)
        try:
            torch.mm(left, right, out=output)
            torch.cuda.current_stream(device_ordinal).synchronize()
        finally:
            del output
            del right
            del left
            torch.cuda.current_stream(device_ordinal).synchronize()

    # The objects standing on a lease

    def occupants(
        self,
        allocations: Sequence[PoolAllocation],
        locate: Callable[[int], int | None],
    ) -> dict[int, tuple[object, ...]]:
        return occupants(allocations, locate)

    def retainers(
        self, held: Sequence[object], *, ignore: Sequence[object] = ()
    ) -> dict[int, tuple[str, ...]]:
        return retainers(held, ignore=ignore)

    def dematerialize(self, bindings: Sequence[object]) -> int:
        return dematerialize(bindings)


__all__ = ["REQUIRED_STORAGE_OPERATIONS", "PyTorchFrontend"]
