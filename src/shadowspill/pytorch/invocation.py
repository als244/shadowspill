"""Explicit completion ownership for asynchronously dispatched callables."""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch

from shadowspill.pytorch.accelerator import is_accelerator
from shadowspill.runtime.timing import Marker


class InvocationResult[T]:
    """One dispatched callable result with an explicit synchronization point."""

    __slots__ = (
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
    ) -> None:
        self._payload = payload
        self._synchronize = synchronize
        self._on_resolved = on_resolved
        self._on_failure = on_failure
        self._lock = threading.RLock()
        self._resolved = False
        self._error: BaseException | None = None

    @property
    def resolved(self) -> bool:
        """Whether this handle has crossed its explicit completion boundary."""

        with self._lock:
            return self._resolved

    def result(self) -> T:
        """Synchronize this invocation once and return its public result."""

        return self._resolve(report_failure=True)

    wait = result

    def _resolve_for_close(self) -> T:
        """Synchronize during callable teardown without recursive cleanup."""

        return self._resolve(report_failure=False)

    def _resolve(self, *, report_failure: bool) -> T:
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
                self._on_resolved(self)
                if report_failure:
                    self._on_failure(error)
                raise
            self._resolved = True
            self._on_resolved(self)
            return self._payload


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
