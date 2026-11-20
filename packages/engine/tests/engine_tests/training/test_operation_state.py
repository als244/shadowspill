"""Operation boundaries preserve contributions without guessing from gradients."""

from dataclasses import replace

import pytest

from shadowspill.training.session.state import OperationState


def test_completion_projection_is_the_same_boundary_without_committing_it():
    state = OperationState()
    before = state.boundary
    pending = state.begin("forward_backward", "a", participates=True)
    projected = state.completion(pending)
    assert state.boundary is before
    assert state.completion(pending) == projected
    assert projected.accumulation.contributions == ("a",)
    assert state.complete(pending) == projected
    before = state.boundary
    pending = state.begin("optim_step", "optim")
    projected = state.completion(pending)
    assert state.boundary is before
    assert projected.completed_updates == 1 and not projected.accumulation.open
    assert state.complete(pending) == projected


def test_restore_history_changes_only_at_confirmation_and_never_rewinds_identity():
    state = OperationState(completed_updates=10, history_id="initial")
    before = state.boundary
    rejected = state.begin("restore", "rejected")
    projected = state.completion(rejected, restored_updates=3)
    assert projected.history_id != before.history_id
    assert state.completion(rejected, restored_updates=3) == projected
    assert state.boundary == before
    assert state.fail(rejected, execution_started=False) == before

    restore = state.begin("restore", "restore-1")
    expected = state.completion(restore, restored_updates=3)
    completed = state.complete(restore, restored_updates=3)
    assert completed == expected and completed.completed_updates == 3
    assert completed.history_id not in (before.history_id, projected.history_id)
    assert state.complete(state.begin("forward", "next")) == completed
    second = state.complete(state.begin("restore", "restore-2"), restored_updates=0)
    assert second.history_id not in (before.history_id, completed.history_id)
    assert second.completed_updates == 0


def finish(state, command, request_id, **kwargs):
    pending = state.begin(command, request_id, **kwargs)
    state.complete(pending)
    return pending


def test_rejected_empty_parts_preserve_first_write_and_generation():
    state = OperationState()
    before = state.boundary
    with pytest.raises(ValueError, match="no loss participation"):
        state.begin("forward_backward", "empty", participates=False)
    assert state.boundary == before
    first = finish(state, "forward_backward", "first", participates=True)
    assert not first.accumulate_gradients
    assert state.boundary.accumulation.generation == 1
    before = state.boundary
    finish(state, "forward", "evaluate")
    with pytest.raises(ValueError, match="no loss participation"):
        state.begin("forward_backward", "later_empty", participates=False)
    assert state.boundary == before
    last = finish(state, "forward_backward", "last", participates=True)
    assert last.accumulate_gradients
    assert state.boundary.accumulation.contributions == ("first", "last")
    assert state.boundary.accumulation.participating_contributions == 2
    finish(state, "optim_step", "update")
    assert state.boundary.completed_updates == 1
    assert not state.boundary.accumulation.open
    next_part = finish(state, "forward_backward", "next_part", participates=True)
    assert not next_part.accumulate_gradients
    assert state.boundary.accumulation.generation == 2


def test_empty_step_rejects_and_explicit_discard_does_not_advance_optimizer():
    state = OperationState()
    before = state.boundary
    with pytest.raises(ValueError, match="no loss participation"):
        state.begin("step", "empty", participates=False)
    assert state.boundary == before
    with pytest.raises(RuntimeError, match="open accumulation"):
        state.begin("optim_step", "empty_update")
    finish(state, "forward_backward", "valid", participates=True)
    finish(state, "discard", "discard")
    assert state.boundary.completed_updates == 0
    assert state.boundary.accumulation.generation == 1
    assert not finish(
        state, "forward_backward", "new", participates=True
    ).accumulate_gradients


@pytest.mark.parametrize("command", ["step", "save", "restore", "close"])
def test_pending_accumulation_rejects_boundary_changes_without_poisoning(command):
    state = OperationState()
    finish(state, "forward_backward", "part", participates=True)
    before = state.boundary
    with pytest.raises(RuntimeError, match="discard gradients"):
        state.begin(command, "rejected")
    assert state.boundary == before
    finish(state, "discard", "explicit_discard")
    if command == "restore":
        state.complete(state.begin(command, "retry"), restored_updates=0)
    else:
        finish(
            state,
            command,
            "retry",
            **({"participates": True} if command == "step" else {}),
        )


def test_validation_failure_preserves_completed_gradients_and_explicit_correction():
    state = OperationState()
    finish(state, "forward_backward", "a", participates=True)
    before = state.boundary
    rejected = state.begin("forward_backward", "b", participates=True)
    assert state.boundary == before
    assert state.fail(rejected, execution_started=False) == before
    corrected = finish(state, "forward_backward", "corrected_b", participates=True)
    assert corrected.accumulate_gradients
    finish(state, "optim_step", "new_optimizer_request")
    assert state.boundary.completed_updates == 1


@pytest.mark.parametrize(
    "command", ["step", "forward_backward", "forward", "optim_step", "restore"]
)
def test_partial_execution_invalidates_state_and_leaves_only_close(command):
    state = OperationState(completed_updates=10)
    if command == "optim_step":
        finish(state, "forward_backward", "part", participates=True)
    failed = state.begin(
        command,
        "failed",
        **({"participates": True} if command in ("step", "forward_backward") else {}),
    )
    state.fail(failed, execution_started=True)
    assert not state.boundary.valid
    assert state.boundary.completed_updates == 10
    for blocked in ("forward", "optim_step", "discard", "save", "step", "restore"):
        with pytest.raises(RuntimeError, match="invalid"):
            state.begin(blocked, "blocked")
    finish(state, "close", "close")
    assert state.boundary.closed and not state.boundary.valid
    assert state.boundary.completed_updates == 10


def test_completion_is_once_and_publication_errors_cannot_roll_back_an_update():
    state = OperationState()
    pending = state.begin("step", "update", participates=True)
    with pytest.raises(RuntimeError, match="in flight"):
        state.begin("close", "close")
    with pytest.raises(RuntimeError, match="current session"):
        state.complete(replace(pending))
    state.complete(pending)
    for action in (
        lambda: state.complete(pending),
        lambda: state.fail(pending, execution_started=True),
    ):
        with pytest.raises(RuntimeError, match="current session"):
            action()
    assert state.boundary.valid and state.boundary.completed_updates == 1
    # A lost authoritative journal commit stops further work even with a known
    # local completion. Publishing an optional report does not call invalidate.
    state.invalidate()
    assert not state.boundary.valid
    finish(state, "close", "release_failed_workers")
    assert state.boundary.closed


def test_close_rechecks_state_after_prior_queued_optimizer_or_discard():
    state = OperationState()
    finish(state, "forward_backward", "part", participates=True)
    with pytest.raises(RuntimeError, match="discard"):
        state.begin("close", "premature")
    finish(state, "optim_step", "queued_update")
    finish(state, "close", "queued_close")
    assert state.boundary.closed and state.boundary.completed_updates == 1
    with pytest.raises(RuntimeError, match="closed"):
        state.begin("forward", "after_close")


def test_duplicate_contribution_cannot_be_replayed_and_participation_is_explicit():
    state = OperationState()
    with pytest.raises(TypeError, match="global participation"):
        state.begin("step", "missing")
    with pytest.raises(ValueError, match="only step/backward"):
        state.begin("forward", "invalid", participates=True)
    with pytest.raises(RuntimeError, match="open accumulation"):
        state.begin("optim_step", "empty")
    finish(state, "forward_backward", "part", participates=True)
    with pytest.raises(ValueError, match="do not replay"):
        state.begin("forward_backward", "part", participates=True)
