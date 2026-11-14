"""Allocation shutdown gates dispatch without rewriting known execution effects."""

import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic

import pytest

from engine_tests.orchestration.test_connected_process import launch, startup
from engine_tests.server.test_request_records import request
from engine_tests.training.test_controller import configured, submit
from engine_tests.training.test_prepared_requests import upload
from shadowspill_client.protocol.requests import OperationRequest


def test_allocation_stop_reaps_an_actual_owned_child_blocked_after_execute(
    tmp_path, monkeypatch
):
    with (
        launch(tmp_path, "stall_execute") as child,
        startup(child) as group,
        configured(tmp_path) as ctx,
    ):
        child.ready()

        def execute(call, transition):
            return group.execute(call, transition)

        monkeypatch.setattr(ctx.workers, "execute", execute)
        monkeypatch.setattr(ctx.workers, "close", group.close)
        command, wire = upload()
        submit(ctx.controller, command, wire)
        marker = tmp_path / "effects"
        deadline = monotonic() + 5
        try:
            while "1\n" not in marker.read_text():
                assert monotonic() < deadline
                Event().wait(0.01)
            assert ctx.controller.status("a").state == "running"
        finally:
            ctx.controller.stop(
                lambda: child.close(grace_seconds=0.05, terminate_seconds=0.05),
                timeout=5,
            )
        assert child.closed and child.process.returncode is not None
        assert ctx.controller.closed and ctx.budget.used == (0, 0)
        failed = ctx.controller.status("a")
        assert (
            failed.failure.execution_effect == "unknown"
            and not failed.failure.state_valid
        )
        assert ctx.controller.register(command) == failed
        assert marker.read_text() == "initialized\n1\n"


def test_allocation_end_preserves_last_known_accumulation_without_an_implicit_update(
    tmp_path,
):
    with configured(tmp_path) as ctx:
        command, wire = upload(operation="forward_backward")
        submit(ctx.controller, command, wire)
        original = ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        with pytest.raises(RuntimeError, match="discard gradients"):
            ctx.controller.close()
        ctx.controller.stop(ctx.workers.close, timeout=5)
        assert ctx.controller.closed and ctx.workers.closed
        assert not ctx.controller.boundary.valid and ctx.controller.boundary.closed
        assert ctx.controller.boundary.accumulation == original.result.accumulation
        assert ctx.controller.boundary.completed_updates == 0
        assert ctx.controller.register(command) == original
        assert ctx.workers.calls == ["a"] and ctx.budget.used == (0, 0)
        ctx.controller.stop(lambda: pytest.fail("closed run stopped twice"))


def test_completion_racing_with_stop_keeps_its_receipt_and_fails_the_queued_suffix(
    tmp_path,
):
    with configured(tmp_path, blocked=True) as ctx:
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.workers.entered.wait(5)
        queued, queued_wire = upload(name="queued", sequence=2)
        submit(ctx.controller, queued, queued_wire)

        def stop():
            ctx.workers.close()
            # This adapter confirms its in-flight computation despite the stop.
            ctx.workers.release.set()

        ctx.controller.stop(stop, timeout=5)
        assert ctx.controller.closed and not ctx.controller.boundary.valid
        assert ctx.controller.status("a").result.completed_updates == 1
        assert ctx.journal.execution("a").completed_updates == 1
        failed = ctx.controller.status("queued").failure
        assert (
            failed.code == "allocation_stopped"
            and failed.execution_effect == "not_started"
        )
        assert not failed.state_valid and ctx.workers.calls == ["a"]
        assert ctx.controller.register(command) == ctx.controller.status("a")


def test_lost_in_flight_worker_is_unknown_and_never_replayed(tmp_path, monkeypatch):
    with configured(tmp_path, blocked=True) as ctx:

        def interrupted(call, transition):
            ctx.workers.calls.append(call.request.identity.request_id)
            ctx.workers.entered.set()
            assert ctx.workers.release.wait(5)
            raise OSError("owned worker exited without a completion")

        monkeypatch.setattr(ctx.workers, "execute", interrupted)
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.workers.entered.wait(5)

        def stop():
            ctx.workers.close()
            ctx.workers.release.set()

        ctx.controller.stop(stop, timeout=5)
        failed = ctx.controller.status("a")
        assert (
            failed.failure.execution_effect == "unknown"
            and not failed.failure.state_valid
        )
        assert ctx.journal.execution("a") is None
        assert ctx.controller.register(command) == failed and ctx.workers.calls == ["a"]
        assert ctx.budget.used == (0, 0) and ctx.controller.closed


def test_failed_stop_keeps_dispatch_gated_and_owners_for_explicit_retry(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.workers.entered.wait(5)

        def failed_stop():
            raise OSError("child termination was not confirmed")

        try:
            with pytest.raises(OSError):
                ctx.controller.stop(failed_stop, timeout=5)
            assert not ctx.controller.closed and ctx.budget.used[0] > 0
            assert ctx.controller.status("a").state == "running"
            with pytest.raises(RuntimeError, match="closing"):
                ctx.controller.register(upload(name="later", sequence=2)[0])
            with pytest.raises(RuntimeError, match="retry allocation stop"):
                ctx.controller.close()
        finally:
            ctx.workers.release.set()
            ctx.controller.stop(ctx.workers.close, timeout=5)
        assert ctx.controller.closed and ctx.budget.used == (0, 0)


def test_stop_join_timeout_retains_the_confirmed_result_and_its_publication_owner(
    tmp_path,
):
    with configured(tmp_path) as ctx:
        entered, release = Event(), Event()
        publish = ctx.controller._publish

        def blocked(call, receipt, work):
            entered.set()
            assert release.wait(5)
            return publish(call, receipt, work)

        ctx.controller._publish = blocked
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert entered.wait(5)
        try:
            with pytest.raises(TimeoutError, match="in-flight result"):
                ctx.controller.stop(ctx.workers.close, timeout=0.01)
            assert ctx.journal.execution("a").completed_updates == 1
            assert ctx.controller.status("a").state == "running"
            assert not ctx.workers.works[0].closed and ctx.budget.used[0] > 0
        finally:
            release.set()
            ctx.controller.stop(ctx.workers.close, timeout=5)
        assert ctx.controller.status("a").result.completed_updates == 1
        assert ctx.workers.works[0].closed and ctx.budget.used == (0, 0)


def test_allocation_stop_retains_a_blocked_upload_until_its_io_returns(tmp_path):
    with configured(tmp_path) as ctx, ThreadPoolExecutor(1) as threads:
        entered, release = Event(), Event()
        command, wire = upload()
        ctx.controller.register(command)

        class Body(io.BytesIO):
            def readinto(self, buffer):
                entered.set()
                assert release.wait(5)
                return super().readinto(buffer)

        def receive():
            with Body(wire) as stream:
                return ctx.controller.receive("a", stream)

        future = threads.submit(receive)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="input producers"):
                ctx.controller.stop(ctx.workers.close, timeout=5)
            assert not ctx.controller.closed and ctx.budget.used[0] > 0
            assert ctx.controller.status("a").failure.execution_effect == "not_started"
            assert not ctx.workers.calls
        finally:
            release.set()
            future.result(5)
            ctx.controller.stop(ctx.workers.close, timeout=5)
        assert ctx.controller.closed and ctx.budget.used == (0, 0)


def test_ordered_close_finishing_during_stop_keeps_its_original_valid_boundary(
    tmp_path,
):
    with configured(tmp_path, blocked=True) as ctx:
        command = OperationRequest(request("close").identity, "close")
        submit(ctx.controller, command)
        assert ctx.workers.entered.wait(5)

        def stop():
            ctx.workers.close()
            ctx.workers.release.set()

        ctx.controller.stop(stop, timeout=5)
        assert ctx.controller.closed and ctx.controller.boundary.valid
        assert ctx.controller.status("close").result.state_valid
        assert ctx.controller.register(command) == ctx.controller.status("close")


@pytest.mark.parametrize("timeout", [-1, True, float("inf")])
def test_invalid_stop_timeout_cannot_disable_a_usable_controller(tmp_path, timeout):
    with configured(tmp_path) as ctx:
        with pytest.raises((TypeError, ValueError)):
            ctx.controller.stop(lambda: pytest.fail("workers stopped"), timeout=timeout)
        command, wire = upload()
        submit(ctx.controller, command, wire)
        assert ctx.controller.wait("a", timeout=5).result.completed_updates == 1
