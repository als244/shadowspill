"""Bounded connection work; completed futures belong to their callers."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from ._records import integer
from .buffers import BufferBudget

TASK_BYTES = 32 << 20


class ConnectionClosed(ConnectionError):
    """Local connection ended; this says nothing about accepted server work."""


class CallFuture[T]:
    """Waiting never cancels or repeats a remote operation."""

    def __init__(self) -> None:
        self._future: Future[T] = Future()

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return False

    def _set_result(self, value: T) -> None:
        self._future.set_result(value)

    def _set_exception(self, error: BaseException) -> None:
        self._future.set_exception(error)


class ConnectionWork:
    def __init__(self, max_pending: int, workers: int) -> None:
        integer(max_pending, "max_pending_requests", minimum=1)
        integer(workers, "max_connections", minimum=1)
        self.stopped = Event()
        self.pending = BufferBudget(max_pending * TASK_BYTES, max_pending)
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="shadowspill-client"
        )

    def check(self) -> None:
        if self.stopped.is_set():
            raise ConnectionClosed(
                "client connection is closed; accepted work continues"
            )

    def pause(self, seconds: float = 0.05) -> None:
        self.stopped.wait(seconds)
        self.check()

    def submit[T](
        self,
        action: Callable[[], T],
        *,
        future: CallFuture[T] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> CallFuture[T]:
        self.check()
        lease = self.pending.reserve(TASK_BYTES)
        result = CallFuture[T]() if future is None else future

        def run() -> None:
            nonlocal action, cleanup
            value: T
            error: BaseException | None = None
            try:
                try:
                    self.check()
                    value = action()
                finally:
                    if cleanup is not None:
                        cleanup()
            except BaseException as caught:
                # Retaining a failed future must not retain its task's inputs.
                caught.__traceback__ = None
                caught.__cause__ = caught.__context__ = None
                error = caught
            finally:
                del action
                cleanup = None
                lease.close()
            if error is None:
                result._set_result(value)
            else:
                result._set_exception(error)

        try:
            self.executor.submit(run)
        except BaseException:
            lease.close()
            raise
        return result

    def stop(self) -> None:
        self.stopped.set()
        self.pending.close()

    def close(self) -> None:
        self.stop()
        self.executor.shutdown(wait=True)
