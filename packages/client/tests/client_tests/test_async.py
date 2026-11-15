"""Connection futures own bounded pending work, never server cancellation."""

import threading
import weakref
from concurrent.futures import TimeoutError

import pytest

from shadowspill_client._async import TASK_BYTES, ConnectionClosed, ConnectionWork


def test_timeout_and_cancel_do_not_repeat_or_cancel_work():
    work = ConnectionWork(1, 1)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def action():
        calls.append(1)
        entered.set()
        release.wait(5)
        return 17

    try:
        future = work.submit(action)
        assert entered.wait(5)
        assert not future.cancel() and not future.done()
        with pytest.raises(TimeoutError):
            future.result(timeout=0)
        assert work.pending.used == (TASK_BYTES, 1)
        release.set()
        assert future.result(5) == 17 and future.result(0) == 17
        assert work.pending.used == (0, 0) and calls == [1]
    finally:
        release.set()
        work.close()


def test_close_wakes_capacity_waiters_and_releases_queued_captures():
    work = ConnectionWork(2, 1)
    entered, release, blocked = threading.Event(), threading.Event(), threading.Event()

    class Input:
        pass

    value = Input()
    reference = weakref.ref(value)

    def occupy():
        entered.set()
        release.wait(5)

    first = work.submit(occupy)
    assert entered.wait(5)
    second = work.submit(
        lambda keep=value: pytest.fail("queued work started after close")
    )
    del value
    errors = []

    def submit():
        blocked.set()
        try:
            work.submit(lambda: None)
        except RuntimeError as error:
            errors.append(str(error))

    caller = threading.Thread(target=submit)
    closer = threading.Thread(target=work.close)
    caller.start()
    assert blocked.wait(5)
    closer.start()
    assert work.stopped.wait(5)
    caller.join(5)
    release.set()
    closer.join(5)
    assert not caller.is_alive() and not closer.is_alive()
    assert errors and first.result(0) is None
    with pytest.raises(ConnectionClosed):
        second.result(0)
    assert reference() is None and work.pending.used == (0, 0)


def test_failed_future_does_not_retain_its_action_input():
    work = ConnectionWork(1, 1)

    class Input:
        pass

    value = Input()
    reference = weakref.ref(value)

    def fail(keep=value):
        raise ValueError("fixture failure")

    future = work.submit(fail)
    del value, fail
    with pytest.raises(ValueError, match="fixture failure"):
        future.result(5)
    work.close()
    assert reference() is None and work.pending.used == (0, 0)
