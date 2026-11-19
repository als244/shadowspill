"""Bounded connection work; completed futures belong to their callers."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Self

from ._records import integer
from .buffers import BufferBudget

TASK_BYTES = 32 << 20


class ConnectionClosed(ConnectionError):
    """The client connection closed before a call finished.

    It says nothing about work the service accepted, which continues.
    """


class CallFuture[T]:
    """The pending value of a client call.

    The client's calls to the service return one, or one of its subclasses
    such as `OperationFuture`. Waiting, with or without a timeout, never
    cancels or repeats the remote work. If the client connection closes first,
    the future fails with `ConnectionClosed`, a `ConnectionError`; work the
    service has accepted continues.
    """

    def __init__(self) -> None:
        self._future: Future[T] = Future()

    def result(self, timeout: float | None = None) -> T:
        """Wait for the call's value.

        Parameters
        ----------
        timeout : float, optional
            Longest wait, in seconds. By default the wait lasts until the call
            finishes.

        Returns
        -------
        object
            The call's value: a `TrainingResult` for a
            ``CallFuture of TrainingResult``.

        Raises
        ------
        TimeoutError
            If the call has not finished within ``timeout``. The call goes on.
        Exception
            The call's own error, if it failed.
        """
        return self._future.result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Wait for the call and return its error.

        Parameters
        ----------
        timeout : float, optional
            Longest wait, in seconds. By default the wait lasts until the call
            finishes.

        Returns
        -------
        BaseException or None
            The call's error, or None if it succeeded.

        Raises
        ------
        TimeoutError
            If the call has not finished within ``timeout``. The call goes on.
        """
        return self._future.exception(timeout)

    def done(self) -> bool:
        """Return whether the call has finished.

        Returns
        -------
        bool
            True once the call has a value or an error.
        """
        return self._future.done()

    def add_done_callback(self, callback: Callable[[Self], None]) -> None:
        """Run ``callback`` when the call finishes, without waiting for it.

        The callback runs once, on success or failure. If the call has already
        finished, it runs immediately. It should return promptly.

        Parameters
        ----------
        callback : callable
            Function that takes this future as its only argument.
        """
        self._future.add_done_callback(lambda _: callback(self))

    def cancel(self) -> bool:
        """Return False: a client call is never cancelled.

        Returns
        -------
        bool
            Always False.
        """
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
