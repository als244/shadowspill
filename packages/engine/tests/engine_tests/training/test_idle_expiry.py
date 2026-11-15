"""Only an explicitly idle controller may expire; durable work is never replayed."""

import io
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from engine_tests.server.test_jobs import until
from engine_tests.server.test_request_records import request
from engine_tests.training.test_controller import configured, submit
from engine_tests.training.test_prepared_requests import upload
from shadowspill.training.session import controller as module


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "monotonic", lambda: value.now)
    return value


def test_explicit_idle_timer_retires_without_client_polling(tmp_path):
    with configured(tmp_path, idle_timeout=0.05) as ctx:
        until(lambda: ctx.controller.closed)
        assert ctx.controller.expired and ctx.workers.closed
        assert ctx.controller.boundary.closed and not ctx.controller.boundary.valid
        assert ctx.controller.boundary.completed_updates == 0
        assert ctx.journal.last_sequence == 0 and not ctx.workers.calls
        assert ctx.budget.used == (0, 0)


def test_default_has_no_idle_timer_even_after_large_elapsed_time(tmp_path, clock):
    with configured(tmp_path) as ctx:
        clock.now = 1e12
        ctx.controller.attach("writer")
        assert ctx.controller._idle_wait() is None
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.controller.wait("a", timeout=5).result.completed_updates == 1
        assert not ctx.controller.expired and not ctx.workers.closed


def test_large_valid_timeout_cannot_overflow_the_platform_wait(tmp_path):
    with configured(tmp_path, idle_timeout=1e30) as ctx:
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.controller.wait("a", timeout=5).result.completed_updates == 1
        assert not ctx.controller.expired


@pytest.mark.parametrize("stage", ["upload", "execution", "publication"])
def test_pending_work_suspends_expiry_and_completion_starts_a_full_interval(
    tmp_path, monkeypatch, clock, stage
):
    with configured(tmp_path, blocked=stage == "execution", idle_timeout=60) as ctx:
        controller = ctx.controller
        command, wire = upload()
        entered, release = Event(), Event()
        received = []
        producer = None
        if stage == "publication":
            original = controller._publish

            def publish(*args):
                entered.set()
                assert release.wait(5)
                return original(*args)

            monkeypatch.setattr(controller, "_publish", publish)
        try:
            if stage == "upload":

                class SlowBody(io.BytesIO):
                    def readinto(self, buffer):
                        entered.set()
                        assert release.wait(5)
                        return super().readinto(buffer)

                controller.register(command)
                producer = Thread(
                    target=lambda: received.append(
                        controller.receive("a", SlowBody(wire))
                    )
                )
                producer.start()
            else:
                submit(controller, command, wire)
            assert (ctx.workers.entered if stage == "execution" else entered).wait(5)
            clock.now = 1000
            controller._notify()
            assert controller._idle_wait() is None
            assert not controller.expired and not ctx.workers.closed
        finally:
            release.set()
            ctx.workers.release.set()
            if producer is not None:
                producer.join(5)
                assert not producer.is_alive()
        result = controller.wait("a", timeout=5)
        controller.drain(timeout=5)
        assert result.result.completed_updates == 1
        assert controller._idle_wait() == 60
        clock.now += 59
        assert controller.status("a") == result
        assert controller.register(command) == result
        controller.attach("reattached")
        assert controller._idle_wait() == 1
        clock.now += 2
        controller._notify()
        until(lambda: controller.closed)
        assert controller.expired and ctx.workers.calls == ["a"]
        assert controller.boundary.completed_updates == 1
        assert controller.status("a") == result and result.result.state_valid
        assert controller.register(command) == controller.receive("a") == result


def test_expiry_records_lost_accumulation_without_implicit_optimizer_or_discard(
    tmp_path, clock
):
    with configured(tmp_path, idle_timeout=60) as ctx:
        command, wire = upload(operation="forward_backward")
        submit(ctx.controller, command, wire)
        result = ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        assert ctx.controller._idle_wait() == 60
        clock.now += 61
        ctx.controller._notify()
        until(lambda: ctx.controller.closed)
        boundary = ctx.controller.boundary
        assert boundary.accumulation == result.result.accumulation
        assert boundary.accumulation.open and boundary.accumulation.contributions == (
            "a",
        )
        assert boundary.completed_updates == 0 and not boundary.valid
        assert ctx.workers.calls == ["a"] and ctx.controller.expired
        assert ctx.controller.status("a") == result
        with pytest.raises(RuntimeError, match="closed"):
            ctx.controller.register(request("late", 2))


def test_idle_record_failure_is_reported_and_cleanup_remains_reachable(
    tmp_path, monkeypatch, clock
):
    with configured(tmp_path, idle_timeout=60) as ctx:

        def failed_wait():
            raise OSError("idle journal lookup failed")

        monkeypatch.setattr(ctx.controller, "_idle_wait", failed_wait)
        ctx.controller._notify()
        until(lambda: ctx.controller.failure is not None)
        assert "journal lookup failed" in ctx.controller.failure
        assert not ctx.controller.expired
        assert not ctx.controller.boundary.valid
        ctx.controller.close()
        assert ctx.workers.closed and ctx.budget.used == (0, 0)
