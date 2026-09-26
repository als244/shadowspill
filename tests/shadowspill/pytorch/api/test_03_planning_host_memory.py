"""Where profiling keeps what it measures on, and that a plan keeps none of it.

Each backward is measured on what its forward saved, copied to the host. Those
copies are kept in the spill pool while the backwards are measured, so they
spend none of the host memory the pool was sized to leave free, and they are
all released before the plan's first step.
"""

from __future__ import annotations

import gc
from functools import partial

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import PlanningError
from shadowspill.pytorch import import_model_state, plan_step
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.profiling.profiler import TaskProfiler, _SavedValues
from shadowspill.pytorch.state.registry import registry_for
from shadowspill.runtime import Runtime

from ..runtime_test_support import public_test_runtime
from .test_01_public_training import _require_adapter


def _objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.mse_loss(model(value), target)


def _plan(model: nn.Module, runtime: Runtime, tmp_path: object):
    return plan_step(
        model,
        objective=_objective,
        optimizer=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=[[torch.randn(3, 8), torch.randn(3, 4)]],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_plan_keeps_no_host_copy_of_what_its_forwards_saved(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved values are measured on in the spill pool, and all released.

    Planning keeps the artifacts the copies are attached to, so they are
    emptied, and the pool given back what they occupied, once every task is
    measured and warmed.
    """

    _require_adapter()
    in_pool: list[int] = []
    release = TaskProfiler.release_host_memory

    def recording_release(self: TaskProfiler) -> None:
        in_pool.append(self.saved_value_bytes_in_pool)
        release(self)

    monkeypatch.setattr(TaskProfiler, "release_host_memory", recording_release)
    torch.manual_seed(79)
    runtime = public_test_runtime()
    model = import_model_state(
        nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        runtime=runtime,
        pool="spill",
    )
    training = _plan(model, runtime, tmp_path)
    assert in_pool and in_pool[0] > 0, "the saved values were kept in the pool"
    # By exact type: isinstance would reach through any dead weak proxy the
    # collector is tracking and raise.
    saved = [
        item.representative_value
        for item in gc.get_objects()
        if type(item) is TaskInputProvenance
        and item.produced_device_type is not None
        and item.representative_value is not None
    ]
    assert saved, "the backward is measured on what its forward saved"
    assert all(value.untyped_storage().nbytes() == 0 for value in saved)
    # No saved value is left in the pool for the plan's steps to find.
    assert not [
        state
        for state in registry_for(runtime).values()
        if type(state.target) is _SavedValues
    ]
    training.close()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_planning_fails_when_the_spill_pool_has_no_room_for_saved_values(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding them beside the pool instead would spend the host memory the
    pool was sized to leave free, so a pool without room fails planning."""

    _require_adapter()
    statistics = Runtime.pool_statistics

    def full_spill_pool(self: Runtime, pool: str = "execution"):
        result = statistics(self, pool)
        if pool == "spill":
            result.free_bytes = 0
        return result

    monkeypatch.setattr(Runtime, "pool_statistics", full_spill_pool)
    runtime = public_test_runtime()
    model = import_model_state(
        nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        runtime=runtime,
        pool="spill",
    )
    with pytest.raises(PlanningError, match="has no room for the values a forward"):
        _plan(model, runtime, tmp_path)
    # The failed plan left nothing of its own in the pool.
    assert {type(state.target) for state in registry_for(runtime).values()} == {
        type(model)
    }
