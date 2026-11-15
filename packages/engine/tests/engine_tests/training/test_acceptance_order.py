"""Actual durable acceptance follows submission order and settled failure state."""

from dataclasses import replace

import pytest

from engine_tests.server.test_request_records import request, store
from shadowspill.training.session.acceptance import AcceptanceOrder
from shadowspill.training.session.state import OperationState
from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.protocol.results import ExecutionReceipt, OperationResult


def finish(records, state, key, *, participates=None):
    command = records.get(key)[0]
    transition = state.begin(command.operation, key, participates=participates)
    records.dispatch(key)
    boundary = state.complete(transition)
    records.executed(
        ExecutionReceipt(
            command.identity,
            command.operation,
            boundary.history_id,
            "skipped" if transition.skipped else "completed",
            boundary.completed_updates,
            boundary.accumulation,
        )
    )
    return records.complete(
        OperationResult(
            command.identity,
            command.operation,
            boundary.history_id,
            "skipped" if transition.skipped else "completed",
            boundary.completed_updates,
            boundary.accumulation,
            grad_scale=command.grad_scale,
        )
    )


def test_later_complete_upload_waits_for_missing_metadata_and_validation(tmp_path):
    with store(tmp_path) as records:
        order = AcceptanceOrder(records, OperationState(), max_pending=3)
        records.submit(request("b", 2))
        assert order.validated("b").state == "uploading"
        records.submit(request("a", 1))
        order.advance()
        assert records.get("b")[1].acceptance is None
        a = order.validated("a")
        b = records.get("b")[1]
        assert a.state == b.state == "accepted"
        assert a.acceptance.journal_position < b.acceptance.journal_position
        position = records._journal.position
        # A lost response and repeated completion callback return the originals.
        assert order.validated("a") == a
        assert order.validated("b") == b
        assert records._journal.position == position
    with store(tmp_path) as reopened:
        assert reopened.at_sequence(1)[1] == a
        assert reopened.at_sequence(2)[1] == b
        assert reopened.at_sequence(3) is None


def test_closed_run_rejects_later_upload_despite_a_missing_sequence(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        order = AcceptanceOrder(records, state, max_pending=3)
        close = OperationRequest(request("close", 1).identity, "close")
        records.submit(close)
        records.submit(request("late", 3))
        order.validated("close")
        assert order.validated("late").state == "uploading"
        finish(records, state, "close")
        order.advance()
        failure = records.get("late")[1].failure
        assert (
            failure.code == "state_closed" and failure.execution_effect == "not_started"
        )
        assert not records.pending() and not order._ready


def test_malformed_b_fails_queued_optimizer_but_preserves_a_for_corrected_work(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        order = AcceptanceOrder(records, state, max_pending=4)
        a = replace(request("a", 1), operation="forward_backward", mode="training")
        b = replace(a, identity=replace(a.identity, request_id="b", sequence=2))
        c = replace(
            request("c", 3), operation="optim_step", mode=None, dependencies=("a", "b")
        )
        for command in (a, b, c):
            records.submit(command)
        order.validated("a")
        finish(records, state, "a", participates=True)
        surviving = state.boundary
        assert order.validated("c").state == "uploading"
        rejected = order.reject("b", code="bad_media", message="Invalid media shape.")
        dependent = records.get("c")[1]
        assert rejected.failure.execution_effect == "not_started"
        assert rejected.failure.accumulation == surviving.accumulation
        assert dependent.failure.failed_dependencies == ("b",)
        assert dependent.failure.accumulation.contributions == ("a",)
        assert state.boundary == surviving
        corrected = replace(
            b, identity=replace(b.identity, request_id="fixed", sequence=4)
        )
        records.submit(corrected)
        assert order.validated("fixed").state == "accepted"
        finish(records, state, "fixed", participates=True)
        assert state.boundary.accumulation.contributions == ("a", "fixed")
        optim = replace(
            c,
            identity=replace(c.identity, request_id="optim", sequence=5),
            dependencies=("a", "fixed"),
        )
        records.submit(optim)
        order.validated("optim")
        completed = finish(records, state, "optim")
        assert completed.result.completed_updates == 1
        assert not completed.result.accumulation.open


def test_rejection_waits_for_earlier_execution_before_reporting_surviving_boundary(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        order = AcceptanceOrder(records, state, max_pending=3)
        a = replace(request("a", 1), operation="step", mode="training")
        records.submit(a)
        records.submit(request("b", 2))
        records.submit(request("c", 3))
        order.validated("a")
        pending = state.begin("step", "a", participates=True)
        records.dispatch("a")
        assert (
            order.reject("b", code="invalid", message="Invalid input.").state
            == "uploading"
        )
        assert order.validated("c").state == "uploading"
        boundary = state.complete(pending)
        records.executed(
            ExecutionReceipt(
                a.identity,
                "step",
                boundary.history_id,
                "completed",
                boundary.completed_updates,
            )
        )
        records.complete(
            OperationResult(
                a.identity,
                "step",
                boundary.history_id,
                "completed",
                boundary.completed_updates,
                grad_scale=1,
            )
        )
        order.advance()
        for key in ("b", "c"):
            failure = records.get(key)[1].failure
            assert failure.completed_updates == 1
            assert failure.state_valid and failure.execution_effect == "not_started"


def test_out_of_order_validation_failure_cannot_skip_an_unresolved_predecessor(
    tmp_path,
):
    with store(tmp_path) as records:
        state = OperationState()
        order = AcceptanceOrder(records, state, max_pending=2)
        records.submit(request("b", 2))
        assert (
            order.reject("b", code="invalid", message="Invalid input.").state
            == "uploading"
        )
        records.submit(request("a", 1))
        assert order.validated("a").state == "accepted"
        assert records.get("b")[1].state == "uploading"
        finish(records, state, "a")
        order.advance()
        assert records.get("b")[1].failure.code == "invalid"


def test_delayed_metadata_keeps_failed_dependency_instead_of_silently_optimizing(
    tmp_path,
):
    with store(tmp_path) as records:
        order = AcceptanceOrder(records, OperationState(), max_pending=3)
        records.submit(request("a", 1))
        order.reject("a", code="invalid", message="Invalid input.")
        # This metadata arrives after the original failure; its SDK snapshot
        # still names the earlier call on which it depended at submission.
        records.submit(replace(request("b", 2), dependencies=("a",)))
        rejected = order.validated("b")
        assert rejected.failure.code == "dependency_failed"
        assert rejected.failure.failed_dependencies == ("a",)
        records.submit(request("new", 3))
        assert order.validated("new").state == "accepted"


@pytest.mark.parametrize("dependency", ["absent", "later"])
def test_dependencies_must_name_an_existing_earlier_request(tmp_path, dependency):
    with store(tmp_path) as records:
        order = AcceptanceOrder(records, OperationState(), max_pending=2)
        records.submit(replace(request(), dependencies=(dependency,)))
        records.submit(request("later", 2))
        status = order.validated("a")
        assert status.failure.code == "invalid_dependency"
        assert status.failure.failed_dependencies == (dependency,)
        assert records.get("later")[1].failure.code == "dependency_failed"


@pytest.mark.parametrize("closed", [False, True])
def test_unavailable_operation_state_rejects_without_creating_acceptance(
    tmp_path, closed
):
    with store(tmp_path) as records:
        state = OperationState()
        if closed:
            state.complete(state.begin("close", "close"))
        else:
            state.invalidate()
        order = AcceptanceOrder(records, state, max_pending=1)
        records.submit(request())
        status = order.validated("a")
        assert status.acceptance is None
        assert status.failure.code == (
            "state_closed" if closed else "state_unavailable"
        )
        assert status.failure.state_valid == closed


def test_fenced_upload_rejection_preserves_the_actual_live_boundary(tmp_path):
    with store(tmp_path) as records:
        state = OperationState(completed_updates=7)
        order = AcceptanceOrder(records, state, max_pending=1)
        records.submit(request())
        lease = records.claim_writer("new", "server")
        failed = order.validated("a")
        assert failed.failure.code == "writer_fenced"
        assert failed.failure.completed_updates == 7 and state.boundary.valid
        new = request("new", 2)
        new = replace(
            new,
            identity=replace(
                new.identity, writer_id=lease.writer_id, writer_epoch=lease.writer_epoch
            ),
        )
        records.submit(new)
        assert order.validated("new").state == "accepted"


@pytest.mark.parametrize("reject", [False, True])
def test_authoritative_failure_invalidates_state_and_requires_reconciliation(
    tmp_path, monkeypatch, reject
):
    records = store(tmp_path)
    state = OperationState()
    order = AcceptanceOrder(records, state, max_pending=1)
    records.submit(request())

    def fail_index(status):
        raise OSError("index failure after durable commit")

    monkeypatch.setattr(records._index, "update", fail_index)
    with pytest.raises(OSError, match="after durable commit"):
        if reject:
            order.reject("a", code="invalid", message="Invalid input.")
        else:
            order.validated("a")
    assert not state.boundary.valid
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.get("a")[1].state == ("failed" if reject else "accepted")


def test_validation_metadata_window_preserves_room_for_earlier_calls(tmp_path):
    with store(tmp_path) as records:
        state = OperationState()
        order = AcceptanceOrder(records, state, max_pending=2)
        for sequence in (3, 2, 1):
            records.submit(request(str(sequence), sequence))
        order.validated("2")
        with pytest.raises(TimeoutError, match="window is full"):
            order.validated("3")
        assert state.boundary.valid
        with pytest.raises(KeyError, match="unknown request"):
            order.validated("unknown")
        order.validated("1")
        assert order.validated("3").state == "accepted"


def test_rejection_does_not_capture_corrected_work_submitted_after_its_outcome(
    tmp_path, monkeypatch
):
    with store(tmp_path) as records:
        order = AcceptanceOrder(records, OperationState(), max_pending=3)
        records.submit(request("bad", 1))
        records.submit(request("dependent", 2))
        fail = records.fail

        def commit_failure(key, failure):
            status = fail(key, failure)
            if key == "bad":
                records.submit(request("corrected", 3))
            return status

        monkeypatch.setattr(records, "fail", commit_failure)
        order.reject("bad", code="invalid", message="Invalid input.")
        assert records.get("dependent")[1].failure.failed_dependencies == ("bad",)
        assert order.validated("corrected").state == "accepted"


def test_sequence_lookup_rejects_nonportable_values(tmp_path):
    with store(tmp_path) as records:
        for sequence in (True, 0, -1, 1 << 63, "1"):
            with pytest.raises(ValueError, match="positive 63-bit"):
                records.at_sequence(sequence)
