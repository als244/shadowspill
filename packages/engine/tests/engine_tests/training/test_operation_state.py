"""Operation boundaries preserve contributions without guessing from gradients."""

from dataclasses import replace

import pytest

from shadowspill.training.session.state import OperationState


def finish(state, command, request_id, **kwargs):
    pending = state.begin(command, request_id, **kwargs)
    state.complete(pending)
    return pending


def test_masked_first_and_later_parts_preserve_first_write_and_generation():
    state = OperationState()
    masked = finish(state, "forward_backward", "masked", participates=False)
    assert masked.skipped and not masked.accumulate_gradients
    assert state.boundary.accumulation.open
    assert state.boundary.accumulation.generation == 1
    first = finish(state, "forward_backward", "first", participates=True)
    assert not first.skipped and not first.accumulate_gradients
    before = state.boundary
    finish(state, "forward", "evaluate")
    assert state.boundary == before
    later_mask = finish(state, "forward_backward", "later_mask", participates=False)
    assert later_mask.skipped and not later_mask.accumulate_gradients
    last = finish(state, "forward_backward", "last", participates=True)
    assert last.accumulate_gradients
    assert state.boundary.accumulation.contributions == (
        "masked",
        "first",
        "later_mask",
        "last",
    )
    assert state.boundary.accumulation.participating_contributions == 2
    update = finish(state, "optim_step", "update")
    assert not update.skipped
    assert state.boundary.completed_updates == 1
    assert not state.boundary.accumulation.open
    next_part = finish(state, "forward_backward", "next_part", participates=True)
    assert not next_part.accumulate_gradients
    assert state.boundary.accumulation.generation == 2


def test_all_masked_skip_and_explicit_discard_do_not_advance_optimizer():
    state = OperationState()
    assert finish(state, "step", "masked_step", participates=False).skipped
    finish(state, "forward_backward", "masked_part", participates=False)
    assert finish(state, "optim_step", "empty_update").skipped
    assert state.boundary.completed_updates == 0
    assert not state.boundary.accumulation.open
    finish(state, "forward_backward", "valid", participates=True)
    finish(state, "discard", "discard")
    assert state.boundary.completed_updates == 0
    assert state.boundary.accumulation.generation == 2
    assert not finish(
        state, "forward_backward", "new", participates=True
    ).accumulate_gradients


@pytest.mark.parametrize("command", ["step", "save", "restore", "close"])
def test_pending_accumulation_rejects_boundary_changes_without_poisoning(command):
    state = OperationState()
    finish(state, "forward_backward", "part", participates=False)
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
def test_partial_execution_invalidates_state_and_explicit_restore_recovers(command):
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
    for blocked in ("forward", "optim_step", "discard", "save", "step"):
        with pytest.raises(RuntimeError, match="invalid"):
            state.begin(blocked, "blocked")
    generation = state.boundary.accumulation.generation
    # The coordinator must recreate failed workers and restore actual tensors
    # before confirming this recovery; changing metadata alone cannot recover.
    restored = state.begin("restore", "restore_checkpoint")
    state.complete(restored, restored_updates=3)
    assert state.boundary.valid and state.boundary.completed_updates == 3
    assert state.boundary.accumulation.generation == generation
    assert not state.boundary.accumulation.open


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
