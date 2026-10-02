"""Where profiling keeps what it measures on, and that a plan keeps none of it.

Each backward is measured on what its forward saved, copied to the host. Those
copies live in the spill pool for just one backward measurement or warmup.
Their total must not accumulate across stages, variants, or cached plans.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest
import torch
import torch.nn as nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.errors import PlanningError
from shadowspill.pytorch import import_model_state, plan_step
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
        profiling_options=CORRECTNESS_PROFILING,
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
    """Both fresh profiles and cache-only warmups retain one snapshot at a time."""

    _require_adapter()
    snapshots: list[tuple[torch.Tensor, ...]] = []
    sizes: list[int] = []
    completed: list[tuple[int, int]] = []
    keep = TaskProfiler.keep_saved_values
    release = TaskProfiler.release_host_memory

    def recording_keep(
        self: TaskProfiler, copies: tuple[torch.Tensor, ...], fill: Callable[[], None]
    ) -> None:
        assert self.saved_value_bytes_in_pool == 0, "previous backward still retained"
        assert all(
            value.untyped_storage().nbytes() == 0
            for snapshot in snapshots
            for value in snapshot
        )
        keep(self, copies, fill)
        snapshots.append(copies)
        sizes.append(self.saved_value_bytes_in_pool)

    def recording_release(self: TaskProfiler) -> None:
        completed.append(
            (self.saved_value_bytes_in_pool, self.peak_saved_value_bytes_in_pool)
        )
        release(self)

    monkeypatch.setattr(TaskProfiler, "keep_saved_values", recording_keep)
    monkeypatch.setattr(TaskProfiler, "release_host_memory", recording_release)
    torch.manual_seed(79)
    runtime = public_test_runtime()
    model = import_model_state(
        nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)),
        runtime=runtime,
        pool="spill",
    )
    for cached in (False, True):
        first_snapshot = len(snapshots)
        training = _plan(model, runtime, tmp_path)
        assert len(snapshots) > first_snapshot, "backward inputs must come from replay"
        if cached:
            assert training.plan_report.profile_cache_misses == 0
            assert training.plan_report.profile_cache_hits > 0
        else:
            assert sum(sizes) > max(sizes), "exercise multiple forward snapshots"
        assert completed[-1] == (0, max(sizes[first_snapshot:]))
        assert all(
            value.untyped_storage().nbytes() == 0
            for snapshot in snapshots
            for value in snapshot
        )
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
