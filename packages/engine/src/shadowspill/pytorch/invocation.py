"""Explicit completion ownership for asynchronously dispatched callables."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.accelerator import is_accelerator
from shadowspill.runtime.timing import Marker

if TYPE_CHECKING:
    from shadowspill.diagnostics.step import StepDiagnostics


class InvocationResult[T]:
    """The result of one `Executor.run` call, available once its work completes.

    `Executor.run` returns as soon as the call's work is dispatched; this
    object waits for it. The first `result` call synchronizes, and every later
    call returns the same value or raises the same error. ``wait`` is another
    name for `result`.
    """

    __slots__ = (
        "_diagnostics",
        "_error",
        "_lock",
        "_on_failure",
        "_on_resolved",
        "_payload",
        "_resolved",
        "_synchronize",
    )

    def __init__(
        self,
        payload: T,
        synchronize: Callable[[], None],
        *,
        on_resolved: Callable[[InvocationResult[T]], None],
        on_failure: Callable[[BaseException], None],
        diagnostics: Callable[[], StepDiagnostics | None] | None = None,
    ) -> None:
        self._payload = payload
        self._synchronize = synchronize
        self._on_resolved = on_resolved
        self._on_failure = on_failure
        self._lock = threading.RLock()
        self._resolved = False
        self._error: BaseException | None = None
        self._diagnostics = diagnostics

    @property
    def diagnostics(self) -> StepDiagnostics | None:
        """The call's runtime trace, or None when it ran without ``runtime_trace``.

        A `StepDiagnostics` when traced. Reading it waits for a traced call, and raises
        the call's error if the call failed.
        """
        if self._diagnostics is None:
            return None
        self.result()
        return self._diagnostics()

    @property
    def resolved(self) -> bool:
        """Whether the call's work has been waited for, successfully or not."""

        with self._lock:
            return self._resolved

    def result(self) -> T:
        """Wait for the call's work and return its result.

        Returns
        -------
        object
            For a training Plan, a `TrainingResult`; for a forward Plan, its
            outputs.

        Raises
        ------
        Exception
            If the call's work failed: its own error. Its outputs are then
            released, and every later call raises the same error.
        """

        with self._lock:
            if self._resolved:
                if self._error is not None:
                    raise self._error
                return self._payload
            try:
                self._synchronize()
            except BaseException as error:
                self._error = error
                self._resolved = True
                # A failed call has no result to return. Holding its outputs
                # would keep device storage alive that no caller can reach or
                # release, and the runtime refuses to close while it lives.
                del self._payload
                self._on_resolved(self)
                self._on_failure(error)
                raise
            self._resolved = True
            self._on_resolved(self)
            return self._payload

    wait = result


class ReusableCompletionEvent:
    """One marker a single-outstanding caller records completion on.

    The marker is the runtime's, taken once and recorded after each dispatch,
    so a callable allocates nothing per invocation and the caller waits for the
    instant its own work finished rather than for the whole stream.
    """

    __slots__ = ("_device", "_marker")

    def __init__(self, runtime_handle: int, device: torch.device) -> None:
        if not is_accelerator(device):
            raise ValueError("callable completion requires a CUDA execution device")
        self._device = device
        self._marker = Marker(runtime_handle)

    def record(self) -> Callable[[], None]:
        """Record completion after the caller's current compute-stream work."""

        self._marker.record(int(torch.cuda.current_stream(self._device).cuda_stream))
        return self._marker.wait

    def release(self) -> None:
        """Give the marker back; the callable that held it is closing."""

        self._marker.release()


__all__ = ["InvocationResult"]
