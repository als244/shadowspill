"""Accepted work belongs to the server queue, not a request handler or waiter."""

import io
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from engine_tests.server.test_request_records import request, store
from engine_tests.training.test_prepared_requests import family, upload
from shadowspill.training.session.controller import RunController
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.protocol.results import OperationResult


@dataclass
class Work:
    closed: bool = False

    def close(self):
        self.closed = True


class Workers:
    def __init__(self, blocked=False):
        self.entered, self.release = Event(), Event()
        if not blocked:
            self.release.set()
        self.calls, self.works = [], []
        self.closed = False

    def execute(self, call, transition):
        assert not self.closed
        self.calls.append(call.request.identity.request_id)
        assert call.worker_plans(transition)
        self.entered.set()
        assert self.release.wait(5)
        if call.request.operation == "close":
            self.close()
        work = Work()
        self.works.append(work)
        return work

    def close(self):
        self.closed = True


@contextmanager
def configured(tmp_path, *, blocked=False, fail_publication=False):
    with store(tmp_path) as journal:
        prepared = PreparedRequests(family(), cache_root=tmp_path)
        execution = prepared.setup.config.execution
        budget = BufferBudget(
            execution.max_staging_bytes, execution.max_pending_requests
        )
        workers = Workers(blocked)

        def publish(call, receipt, work):
            assert journal.execution(receipt.identity.request_id) == receipt
            assert not work.closed
            if fail_publication and receipt.identity.request_id == "a":
                raise OSError("required artifact unavailable")
            return OperationResult(
                receipt.identity,
                receipt.operation,
                receipt.history_id,
                receipt.outcome,
                receipt.completed_updates,
                receipt.accumulation,
                optim=call.controls.optim,
                loss_params=call.controls.loss_params,
                grad_scale=call.controls.grad_scale,
                state_valid=receipt.state_valid,
            )

        controller = RunController(
            journal,
            prepared,
            workers,
            publish,
            server_incarnation="server",
            budget=budget,
        )
        try:
            yield SimpleNamespace(
                controller=controller, journal=journal, workers=workers, budget=budget
            )
        finally:
            workers.release.set()
            try:
                controller.drain(timeout=5)
            except RuntimeError:
                controller._thread.join(timeout=5)
            # Emergency test teardown after a failed assertion; the test for the
            # close guard explicitly resolves accumulation through optim_step.
            if controller.boundary.accumulation.open:
                with controller._dispatcher.order.synchronized():
                    controller._state.invalidate()
            controller.close()
            assert not controller._thread.is_alive()
            assert budget.used == (0, 0)


def submit(controller, command, data=None):
    controller.register(command)
    with io.BytesIO(data or b"") as body:
        return controller.receive(
            command.identity.request_id, body if data is not None else None
        )


def test_discard_is_ordered_deduplicated_and_resolves_only_the_old_generation(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        controller = ctx.controller
        backward, body = upload(operation="forward_backward")
        submit(controller, backward, body)
        assert ctx.workers.entered.wait(5)
        discard = OperationRequest(request("discard", 2).identity, "discard")
        assert submit(controller, discard).state == "accepted"
        assert ctx.workers.calls == ["a"]
        ctx.workers.release.set()
        result = controller.wait("discard", timeout=5).result
        assert result.completed_updates == 0 and not result.accumulation.open
        assert result.accumulation.generation == 1
        controller.drain(timeout=5)
        assert controller.register(discard).result == result
        assert controller.receive("discard").result == result
        assert ctx.workers.calls == ["a", "discard"]
        backward, body = upload(operation="forward_backward", name="fresh", sequence=3)
        submit(controller, backward, body)
        next_result = controller.wait("fresh", timeout=5).result
        assert next_result.accumulation.generation == 2
        assert next_result.accumulation.contributions == ("fresh",)
        discard = OperationRequest(request("resolve", 4).identity, "discard")
        submit(controller, discard)
        assert controller.wait("resolve", timeout=5).state == "completed"


def test_accepted_work_survives_closed_body_timed_out_waiter_and_duplicate_request(
    tmp_path,
):
    with configured(tmp_path, blocked=True) as ctx:
        controller, workers = ctx.controller, ctx.workers
        command, wire = upload()
        assert submit(controller, command, wire).state == "accepted"
        assert workers.entered.wait(5)
        accepted = controller.wait("a", accepted=True, timeout=0)
        assert accepted.acceptance is not None
        with pytest.raises(TimeoutError):
            controller.wait("a", timeout=0.01)
        assert controller.register(command).state == "running"
        assert controller.receive("a").state == "running"
        assert workers.calls == ["a"]
        with pytest.raises(RuntimeError, match="drain"):
            controller.close()
        workers.release.set()
        completed = controller.wait("a", timeout=5)
        assert (
            completed.state == "completed" and completed.result.completed_updates == 1
        )
        assert completed.acceptance == accepted.acceptance
        controller.drain(timeout=5)
        assert all(work.closed for work in workers.works)
        controller.close()
        assert controller.status("a") == controller.register(command) == completed
        assert controller.receive("a") == completed
        assert workers.calls == ["a"]
        with pytest.raises(RuntimeError, match="closed"):
            controller.attach("another")


def test_out_of_order_upload_completion_still_executes_in_submission_order(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        controller = ctx.controller
        a, a_wire = upload(name="a", sequence=1)
        b, b_wire = upload(name="b", sequence=2)
        controller.register(b)
        controller.register(a)
        assert controller.receive("b", io.BytesIO(b_wire)).state == "uploading"
        assert not ctx.workers.calls
        assert controller.receive("a", io.BytesIO(a_wire)).state == "accepted"
        assert ctx.workers.entered.wait(5)
        assert controller.wait("b", accepted=True, timeout=5).acceptance is not None
        assert ctx.workers.calls == ["a"]
        ctx.workers.release.set()
        assert controller.wait("b", timeout=5).result.completed_updates == 2
        controller.drain(timeout=5)
        assert ctx.workers.calls == ["a", "b"]


def test_writer_takeover_fences_old_upload_without_canceling_accepted_work(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        controller = ctx.controller
        a, a_wire = upload(name="a", sequence=1)
        b, _ = upload(name="b", sequence=2)
        submit(controller, a, a_wire)
        assert ctx.workers.entered.wait(5)
        controller.register(b)
        replacement = controller.attach("replacement")
        assert not replacement.ready and replacement.lease.writer_epoch == 2
        assert replacement.lease.next_sequence == 3
        ctx.workers.release.set()
        assert controller.wait("a", timeout=5).state == "completed"
        controller.drain(timeout=5)
        assert controller.status("b").failure.code == "writer_fenced"
        assert controller.attach("replacement").ready
        c, c_wire = upload(name="c", sequence=3)
        c = replace(
            c, identity=replace(c.identity, writer_id="replacement", writer_epoch=2)
        )
        submit(controller, c, c_wire)
        assert controller.wait("c", timeout=5).result.completed_updates == 2
        assert ctx.workers.calls == ["a", "c"]


def test_publication_failure_retains_completed_effect_and_continued_live_queue(
    tmp_path,
):
    with configured(tmp_path, fail_publication=True) as ctx:
        command, wire = upload()
        submit(ctx.controller, command, wire)
        failure = ctx.controller.wait("a", timeout=5)
        assert failure.failure.execution_effect == "completed"
        assert failure.failure.completed_updates == 1 and failure.failure.state_valid
        assert ctx.controller.register(command) == failure
        b, b_wire = upload(name="b", sequence=2)
        submit(ctx.controller, b, b_wire)
        assert ctx.controller.wait("b", timeout=5).result.completed_updates == 2
        assert ctx.workers.calls == ["a", "b"]


def test_close_refuses_open_accumulation_without_disabling_corrected_work(tmp_path):
    with configured(tmp_path) as ctx:
        command, wire = upload(operation="forward_backward")
        submit(ctx.controller, command, wire)
        ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        with pytest.raises(RuntimeError, match="discard gradients"):
            ctx.controller.close()
        assert not ctx.workers.closed and ctx.controller.boundary.accumulation.open
        optim = OperationRequest(request("optim", 2).identity, "optim_step")
        submit(ctx.controller, optim)
        result = ctx.controller.wait("optim", timeout=5)
        assert result.result.completed_updates == 1
        assert not result.result.accumulation.open


def test_ordered_close_drains_prior_work_and_keeps_its_result_queryable(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        controller = ctx.controller
        command, body = upload()
        submit(controller, command, body)
        assert ctx.workers.entered.wait(5)
        close = OperationRequest(request("close", 2).identity, "close")
        assert submit(controller, close).state == "accepted"
        assert not ctx.workers.closed
        ctx.workers.release.set()
        result = controller.wait("close", timeout=5)
        controller.drain(timeout=5)
        assert result.result.completed_updates == 1 and result.result.state_valid
        assert controller.closed and ctx.workers.closed
        with pytest.raises(RuntimeError, match="closed"):
            ctx.budget.reserve(1)
        assert ctx.workers.calls == ["a", "close"]
        assert controller.register(close) == controller.receive("close") == result
        with pytest.raises(RuntimeError, match="closed"):
            controller.register(upload(name="late", sequence=3)[0])


def test_close_rejects_already_accepted_suffix_without_worker_execution(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        close = OperationRequest(request("close", 1).identity, "close")
        submit(ctx.controller, close)
        assert ctx.workers.entered.wait(5)
        late, body = upload(name="late", sequence=2)
        assert submit(ctx.controller, late, body).state == "accepted"
        ctx.workers.release.set()
        assert ctx.controller.wait("close", timeout=5).state == "completed"
        failure = ctx.controller.wait("late", timeout=5).failure
        assert (
            failure.code == "state_closed" and failure.execution_effect == "not_started"
        )
        ctx.controller.drain(timeout=5)
        assert ctx.workers.calls == ["close"] and ctx.controller.closed


def test_close_retires_incomplete_upload_without_waiting_for_body(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        close = OperationRequest(request("close", 1).identity, "close")
        submit(ctx.controller, close)
        assert ctx.workers.entered.wait(5)
        # A later registered upload is incomplete; no receive call is required
        # for close to classify and release its metadata/pipeline ownership.
        late, _ = upload(name="late", sequence=2)
        ctx.controller.register(late)
        ctx.workers.release.set()
        assert ctx.controller.wait("late", timeout=5).failure.code == "state_closed"
        ctx.controller.drain(timeout=5)
        assert ctx.controller.closed and ctx.budget.used == (0, 0)


def test_close_keeps_active_cancelled_upload_storage_until_io_returns(tmp_path):
    entered, release = Event(), Event()
    received = []
    with configured(tmp_path, blocked=True) as ctx:
        close = OperationRequest(request("close", 1).identity, "close")
        submit(ctx.controller, close)
        assert ctx.workers.entered.wait(5)
        late, wire = upload(name="late", sequence=2)
        ctx.controller.register(late)

        class Body(io.BytesIO):
            def readinto(self, buffer):
                entered.set()
                assert release.wait(5)
                return super().readinto(buffer)

        def receive():
            with Body(wire) as body:
                received.append(ctx.controller.receive("late", body))

        producer = Thread(target=receive)
        producer.start()
        try:
            assert entered.wait(5)
            ctx.workers.release.set()
            assert ctx.controller.wait("close", timeout=5).state == "completed"
            assert ctx.controller.wait("late", timeout=5).failure.code == "state_closed"
            assert ctx.budget.used[0] > 0
            with pytest.raises(TimeoutError):
                ctx.controller.drain(timeout=0.01)
        finally:
            release.set()
            producer.join(timeout=5)
        assert not producer.is_alive() and received[0].failure.code == "state_closed"
        ctx.controller.drain(timeout=5)
        assert ctx.controller.closed and ctx.budget.used == (0, 0)


def test_ordered_close_refuses_gradients_then_accepts_explicit_discard(tmp_path):
    with configured(tmp_path) as ctx:
        backward, wire = upload(operation="forward_backward")
        submit(ctx.controller, backward, wire)
        ctx.controller.wait("a", timeout=5)
        close = OperationRequest(request("premature", 2).identity, "close")
        submit(ctx.controller, close)
        rejected = ctx.controller.wait("premature", timeout=5)
        assert rejected.failure.code == "operation_conflict"
        assert rejected.failure.accumulation.contributions == ("a",)
        assert not ctx.workers.closed
        discard = OperationRequest(request("discard", 3).identity, "discard")
        submit(ctx.controller, discard)
        ctx.controller.wait("discard", timeout=5)
        close = OperationRequest(request("close", 4).identity, "close")
        submit(ctx.controller, close)
        assert ctx.controller.wait("close", timeout=5).state == "completed"
        ctx.controller.drain(timeout=5)
        assert ctx.workers.calls == ["a", "discard", "close"]


def test_close_of_invalid_state_retains_uncertainty_and_last_known_history(tmp_path):
    with configured(tmp_path) as ctx:
        backward, wire = upload(operation="forward_backward")
        submit(ctx.controller, backward, wire)
        ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        with ctx.controller._dispatcher.order.synchronized():
            ctx.controller._state.invalidate()
        close = OperationRequest(request("close", 2).identity, "close")
        submit(ctx.controller, close)
        result = ctx.controller.wait("close", timeout=5)
        ctx.controller.drain(timeout=5)
        assert result.state == "completed" and not result.result.state_valid
        assert result.result.accumulation.contributions == ("a",)
        assert not ctx.journal.execution("close").state_valid
        assert ctx.controller.boundary.closed and not ctx.controller.boundary.valid


def test_close_publication_failure_cannot_reopen_or_replay_workers(tmp_path):
    with configured(tmp_path, fail_publication=True) as ctx:
        close = OperationRequest(request().identity, "close")
        submit(ctx.controller, close)
        status = ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        assert status.failure.execution_effect == "completed"
        assert ctx.controller.closed and ctx.workers.closed
        assert ctx.controller.register(close) == status
        assert ctx.workers.calls == ["a"]


def test_fatal_completion_storage_failure_never_replays_confirmed_effects(
    tmp_path, monkeypatch
):
    with configured(tmp_path) as ctx:

        def fail(_):
            raise OSError("completion storage unavailable")

        monkeypatch.setattr(ctx.journal, "complete", fail)
        command, wire = upload()
        submit(ctx.controller, command, wire)
        ctx.controller._thread.join(timeout=5)
        assert not ctx.controller._thread.is_alive()
        status = ctx.controller.status("a")
        assert status.failure.code == "controller_failed"
        assert status.failure.execution_effect == "completed"
        assert status.failure.completed_updates == 1 and not status.failure.state_valid
        assert ctx.workers.closed and ctx.workers.calls == ["a"]
        assert ctx.controller.register(command) == status
        with pytest.raises(RuntimeError, match="controller failed"):
            ctx.controller.register(upload(name="b", sequence=2)[0])


def test_nonempty_journal_is_not_a_new_live_controller_or_replay_source(tmp_path):
    with configured(tmp_path) as ctx:
        command, wire = upload()
        submit(ctx.controller, command, wire)
        ctx.controller.wait("a", timeout=5)
        ctx.controller.drain(timeout=5)
        with pytest.raises(ValueError, match="fresh records"):
            RunController(
                ctx.journal,
                ctx.controller._prepared,
                Workers(),
                lambda *args: None,
                server_incarnation="other",
                budget=ctx.budget,
            )
