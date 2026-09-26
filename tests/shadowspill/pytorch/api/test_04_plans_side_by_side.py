"""Plans admitted side by side on one runtime.

A plan holds its fixed layout in the execution pool for as long as it is
admitted, so a plan made beside it shares the pool with that layout, and what
the pool holds for the first is nothing the second leaked.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import AdmissionError
from shadowspill.pytorch import import_model_state, plan_forward, plan_step
from shadowspill.runtime.abi import runtime_library
from shadowspill.runtime.occupancy import plan_slices

from ..runtime_test_support import public_test_runtime
from .test_01_public_training import _require_adapter


def _network() -> nn.Module:
    torch.manual_seed(80)
    return nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 2048))


def _objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.mse_loss(model(value), target)


def _batch(seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(512, 2048, generator=generator),
        torch.randn(512, 2048, generator=generator),
    ]


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_forward_is_planned_beside_a_step_holding_more_than_it_reserves(
    tmp_path: object,
) -> None:
    """The forward that evaluates a training step, planned while the step is admitted.

    The step's layout is larger than what planning the forward reserves for
    itself, so counting the step's bytes as the forward's refuses the forward.
    """

    _require_adapter()
    runtime = public_test_runtime()
    model = import_model_state(_network(), runtime=runtime, pool="spill")
    training = plan_step(
        model,
        objective=_objective,
        optimizer=partial(torch.optim.SGD, lr=0.01, foreach=False),
        example_inputs=[_batch(0)],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    training([_batch(1)])
    assert sum(runtime._installed.admitted_layout_bytes.values()) > 128 << 20

    forward = plan_forward(
        model,
        example_inputs=[_batch(0)[0]],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    value = _batch(99)[0]
    reference = _network()
    reference.load_state_dict(training.state_dict()["model"])
    torch.testing.assert_close(forward([value]).cpu(), reference(value).detach())
    forward.close()
    training.close()
    assert not runtime._installed.admitted_layout_bytes


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_forward_shares_the_slab_of_the_step_it_evaluates(tmp_path: object) -> None:
    """The forward that evaluates a training step, admitted into the step's slab.

    Its layout lies at the step's offset and reserves nothing; the two run in
    turn, each computing what it would alone; and the step, whose bytes they
    are, closes last. The step trains on wide batches, so its slab has room
    for the smallest workspace any plan is allowed.
    """

    def wide(seed: int) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return [
            torch.randn(8192, 2048, generator=generator),
            torch.randn(8192, 2048, generator=generator),
        ]

    _require_adapter()
    runtime = public_test_runtime()
    model = import_model_state(_network(), runtime=runtime, pool="spill")
    training = plan_step(
        model,
        objective=_objective,
        optimizer=partial(torch.optim.SGD, lr=0.01, foreach=False),
        example_inputs=[wide(0)],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    training([wide(1)])
    held = sum(runtime._installed.admitted_layout_bytes.values())

    def forward_within(budget: int | None) -> object:
        return plan_forward(
            model,
            example_inputs=[_batch(0)[0]],
            runtime=runtime,
            execution="execution",
            spill="spill",
            execution_budget=budget,
            share_slab_with=training,
            artifact_store=tmp_path,
        )

    with pytest.raises(AdmissionError, match="slab this plan shares"):
        forward_within(held + (1 << 20))
    forward = forward_within(None)
    identity = runtime_library().shadowspill_plan_id
    step_id = int(identity(training._plan_handle))
    forward_id = int(identity(forward._plan_handle))
    slices = {item.plan_id: item for item in plan_slices(runtime, "execution")}
    assert slices[forward_id].offset == slices[step_id].offset
    assert slices[forward_id].slab_plan_id == step_id
    assert slices[forward_id].bytes <= slices[step_id].bytes
    assert sum(runtime._installed.admitted_layout_bytes.values()) == held

    def evaluated(value: torch.Tensor) -> torch.Tensor:
        reference = _network()
        reference.load_state_dict(training.state_dict()["model"])
        return reference(value).detach()

    value = _batch(99)[0]
    torch.testing.assert_close(forward([value]).cpu(), evaluated(value))
    training([wide(2)])
    torch.testing.assert_close(forward([value]).cpu(), evaluated(value))

    with pytest.raises(RuntimeError, match="share this plan's slab"):
        training.close()
    forward.close()
    training.close()
    assert not runtime._installed.admitted_layout_bytes
    assert not plan_slices(runtime, "execution")
