"""Durable dispatch/receipt/publication faults around the real operation state."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Event

import pytest

from engine_tests.server.test_request_records import request, store
from shadowspill.training.session import dispatch
from shadowspill.training.session.dispatch import ExecutionNotStarted, RequestDispatcher
from shadowspill.training.session.state import OperationState
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.results import (
    ArtifactIssue,
    Failure,
    OperationResult,
    RequestStatus,
)


@dataclass
class HostWork:
    closed: bool = False

    def close(self):
        self.closed = True


def publish(receipt, work):
    assert not work.closed
    return OperationResult(
        receipt.identity,
        receipt.operation,
        receipt.history_id,
        receipt.outcome,
        receipt.completed_updates,
        receipt.accumulation,
        metrics={"loss": 2.0},
        grad_scale=1 if receipt.operation in ("step", "optim_step") else None,
    )


def submit(dispatcher, records, name, sequence, operation="step", dependencies=()):
    command = replace(
        request(name, sequence),
        operation=operation,
        mode=None
        if operation == "optim_step"
        else "evaluation"
        if operation == "forward"
        else "training",
        dependencies=dependencies,
    )
    records.submit(command)
    return dispatcher.order.validated(name)


def test_dispatch_precedes_worker_command_and_receipt_precedes_publication(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        work = HostWork()
        calls = []

        def execute(transition):
            calls.append(transition.request_id)
            assert records.get("a")[1].state == "running"
            assert records.execution("a") is None
            assert state.boundary.completed_updates == 0
            return work

        def output(receipt, work):
            assert records.execution("a") == receipt
            assert state.boundary.completed_updates == 1
            return publish(receipt, work)

        result = dispatcher.run("a", execute, output, participates=True)
        assert result.result.completed_updates == 1 and work.closed
        assert dispatcher.run("a", execute, output, participates=True) == result
        assert calls == ["a"]
        assert [row.kind for row in records._journal.records()][-3:] == [
            "request.status",
            "request.executed",
            "request.status",
        ]


def test_dispatch_cannot_overtake_an_upload_or_an_accepted_predecessor(tmp_path):
    with store(tmp_path) as records:
        dispatcher = RequestDispatcher(records, OperationState(), max_pending=2)
        records.submit(request("a", 1))
        records.submit(request("b", 2))
        dispatcher.order.validated("b")
        calls = []

        def execute(transition):
            calls.append(transition.request_id)
            return HostWork()

        with pytest.raises(RuntimeError, match="queue head"):
            dispatcher.run("b", execute, publish)
        dispatcher.order.validated("a")
        with pytest.raises(RuntimeError, match="queue head"):
            dispatcher.run("b", execute, publish)
        dispatcher.run("a", execute, publish)
        dispatcher.run("b", execute, publish)
        assert calls == ["a", "b"]


def test_accumulation_conflict_rejects_before_dispatch_and_keeps_successful_gradients(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=3)
        submit(dispatcher, records, "a", 1, "forward_backward")
        dispatcher.run("a", lambda t: HostWork(), publish, participates=True)
        submit(dispatcher, records, "bad_step", 2)
        submit(dispatcher, records, "queued", 3, "optim_step")

        def forbidden(transition):
            pytest.fail("invalid step must never reach workers")

        status = dispatcher.run("bad_step", forbidden, publish, participates=True)
        assert status.failure.execution_effect == "not_started"
        assert status.failure.accumulation.contributions == ("a",)
        assert records.get("queued")[1].failure.failed_dependencies == ("bad_step",)
        assert state.boundary.valid and state.boundary.completed_updates == 0
        submit(dispatcher, records, "corrected", 4, "optim_step", ("a",))
        result = dispatcher.run("corrected", lambda t: HostWork(), publish)
        assert result.result.completed_updates == 1


def test_worker_failure_is_unknown_invalidates_state_and_never_replays(tmp_path):
    with store(tmp_path) as records:
        state = OperationState(completed_updates=3)
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        submit(dispatcher, records, "b", 2, dependencies=("a",))
        partial_effects = []

        def partial(transition):
            partial_effects.append("mutated")
            raise OSError("worker lost after partial mutation")

        failed = dispatcher.run("a", partial, publish, participates=True)
        assert failed.failure.execution_effect == "unknown"
        assert failed.failure.completed_updates is None and not state.boundary.valid
        following = records.get("b")[1].failure
        assert following.execution_effect == "not_started" and not following.state_valid
        assert following.completed_updates is None and following.accumulation is None
        assert following.failed_dependencies == ("a",)
        assert dispatcher.run("a", partial, publish, participates=True) == failed
        assert partial_effects == ["mutated"]


def test_confirmed_preexecution_worker_abort_preserves_prior_gradients(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=3)
        submit(dispatcher, records, "a", 1, "forward_backward")
        dispatcher.run("a", lambda t: HostWork(), publish, participates=True)
        before = state.boundary
        submit(dispatcher, records, "b", 2, "forward_backward")
        submit(dispatcher, records, "optimizer", 3, "optim_step")
        attempts = []

        def aborted(transition):
            assert records.get("b")[1].state == "running"
            attempts.append(transition.request_id)
            raise ExecutionNotStarted(
                "All workers acknowledged unstarted input staging"
            )

        failed = dispatcher.run("b", aborted, publish, participates=True)
        assert failed.failure.code == "worker_preflight_failed"
        assert failed.failure.execution_effect == "not_started"
        assert failed.failure.state_valid and state.boundary == before
        assert failed.failure.accumulation.contributions == ("a",)
        assert records.get("optimizer")[1].failure.failed_dependencies == ("b",)
        assert dispatcher.run("b", aborted, publish, participates=True) == failed
        assert attempts == ["b"]
        submit(dispatcher, records, "corrected", 4, "forward_backward", ("a",))
        dispatcher.run("corrected", lambda t: HostWork(), publish, participates=True)
        submit(dispatcher, records, "apply", 5, "optim_step")
        completed = dispatcher.run("apply", lambda t: HostWork(), publish)
        assert completed.result.completed_updates == 1


def test_required_publication_failure_preserves_known_update_and_closes_host_outputs(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        submit(dispatcher, records, "b", 2)
        work = HostWork()
        calls = []

        def execute(transition):
            calls.append(transition.request_id)
            return work

        def failed_publication(receipt, work):
            raise OSError("required output file unavailable")

        failed = dispatcher.run("a", execute, failed_publication, participates=True)
        assert failed.failure.execution_effect == "completed"
        assert failed.failure.completed_updates == 1 and failed.failure.state_valid
        assert (
            failed.failure.history_id
            == records.execution("a").history_id
            == state.boundary.history_id
        )
        assert (
            records.execution("a").completed_updates
            == state.boundary.completed_updates
            == 1
        )
        assert records.get("b")[1].failure.failed_dependencies == ("a",)
        assert work.closed
        assert dispatcher.run("a", execute, publish, participates=True) == failed
        assert calls == ["a"]


def test_publisher_cannot_change_the_confirmed_history(tmp_path):
    with store(tmp_path) as records:
        state = OperationState(history_id="confirmed-history")
        dispatcher = RequestDispatcher(records, state, max_pending=1)
        submit(dispatcher, records, "a", 1)
        failed = dispatcher.run(
            "a",
            lambda _: HostWork(),
            lambda receipt, work: replace(
                publish(receipt, work), history_id="invented-history"
            ),
            participates=True,
        )
        assert failed.failure.code == "publication_failed"
        assert failed.failure.history_id == "confirmed-history"
        assert state.boundary.valid and state.boundary.completed_updates == 1


def test_optional_artifact_issue_leaves_training_successful(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=1)
        submit(dispatcher, records, "a", 1)

        def optional_issue(receipt, work):
            return replace(
                publish(receipt, work),
                artifact_issues=(
                    ArtifactIssue("tracker_offline", "Optional tracker is offline."),
                ),
            )

        done = dispatcher.run(
            "a", lambda t: HostWork(), optional_issue, participates=True
        )
        assert done.state == "completed" and done.result.metrics["loss"] == 2.0
        assert done.result.artifact_issues[0].code == "tracker_offline"
        assert state.boundary.valid and state.boundary.completed_updates == 1


@pytest.mark.parametrize("boundary", ["dispatch", "receipt", "result"])
def test_authoritative_failure_stops_dispatch_and_releases_host_owners(
    tmp_path, monkeypatch, boundary
):
    records = store(tmp_path)
    state = OperationState()
    dispatcher = RequestDispatcher(records, state, max_pending=1)
    submit(dispatcher, records, "a", 1)
    work, calls = HostWork(), []
    update, executed = records._index.update, records._index.executed

    def fail_status(status):
        if status.state == ("running" if boundary == "dispatch" else "completed"):
            raise OSError("index unavailable after authoritative commit")
        update(status)

    def fail_receipt(receipt):
        if boundary == "receipt":
            raise OSError("index unavailable after authoritative commit")
        executed(receipt)

    monkeypatch.setattr(records._index, "update", fail_status)
    monkeypatch.setattr(records._index, "executed", fail_receipt)

    def execute(transition):
        calls.append(transition.request_id)
        return work

    with pytest.raises(OSError, match="authoritative commit"):
        dispatcher.run("a", execute, publish, participates=True)
    assert not state.boundary.valid
    assert calls == ([] if boundary == "dispatch" else ["a"])
    assert work.closed == (boundary != "dispatch")
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.get("a")[1].state == (
            "completed" if boundary == "result" else "running"
        )
        assert (recovered.execution("a") is None) == (boundary == "dispatch")


def test_parallel_validation_waits_while_second_dispatch_is_refused(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        records.submit(request("b", 2))
        running, release = Event(), Event()

        def execute(transition):
            running.set()
            assert release.wait(timeout=3)
            return HostWork()

        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(
                dispatcher.run, "a", execute, publish, participates=True
            )
            assert running.wait(timeout=3)
            rejected = dispatcher.order.reject(
                "b", code="bad_input", message="Bad input."
            )
            assert rejected.state == "uploading"
            with pytest.raises(RuntimeError, match="another request"):
                dispatcher.run("b", execute, publish)
            release.set()
            assert pending.result(timeout=3).state == "completed"
        assert records.get("b")[1].failure.completed_updates == 1
        assert state.boundary.valid


def test_dispatch_rechecks_an_accepted_dependency_that_later_failed(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=2)
        submit(dispatcher, records, "a", 1)
        submit(dispatcher, records, "b", 2, dependencies=("a",))
        records.fail(
            "a", Failure("invalid", "Rejected externally", "not_started", True)
        )

        def forbidden(transition):
            pytest.fail("failed dependency must never dispatch")

        failed = dispatcher.run("b", forbidden, publish, participates=True)
        assert failed.failure.code == "dependency_failed"
        assert failed.failure.failed_dependencies == ("a",)
        assert state.boundary.completed_updates == 0 and state.boundary.valid


def test_masked_backward_then_empty_optimizer_preserves_update_count(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=1)
        submit(dispatcher, records, "a", 1, "forward_backward")
        observed = []

        def output_only(transition):
            observed.append(transition)
            assert transition.skipped and not transition.participates
            return HostWork()

        skipped = dispatcher.run("a", output_only, publish, participates=False)
        assert skipped.state == "skipped"
        assert skipped.result.accumulation.contributions == ("a",)
        submit(dispatcher, records, "optim", 2, "optim_step")
        skipped = dispatcher.run("optim", output_only, publish)
        assert skipped.state == "skipped" and skipped.result.completed_updates == 0
        assert not skipped.result.accumulation.open and len(observed) == 2


def test_growing_completion_history_is_rejected_before_gpu_work_and_can_be_optimized(
    tmp_path, monkeypatch
):
    with store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=1)
        first, second = "a" * 128, "b" * 128
        submit(dispatcher, records, first, 1, "forward_backward")
        completed = dispatcher.run(
            first, lambda t: HostWork(), publish, participates=True
        )
        maximum = (
            len(
                encode(
                    RequestStatus(
                        completed.identity,
                        completed.state,
                        completed.acceptance,
                        result=replace(completed.result, history_id="h" * 128),
                    )
                )
            )
            + 16
        )
        monkeypatch.setattr(
            dispatch, "encode", lambda value: encode(value, max_bytes=maximum)
        )
        before = state.boundary
        submit(dispatcher, records, second, 2, "forward_backward")

        def forbidden(transition):
            pytest.fail("oversized completion metadata reached workers")

        failed = dispatcher.run(second, forbidden, publish, participates=True)
        assert failed.failure.code == "metadata_limit"
        assert failed.failure.execution_effect == "not_started"
        assert state.boundary == before and state.boundary.valid
        assert records.execution(second) is None
        submit(dispatcher, records, "optim", 3, "optim_step")
        completed = dispatcher.run("optim", lambda t: HostWork(), publish)
        assert completed.result.completed_updates == 1
