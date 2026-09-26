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

from shadowspill.pytorch import import_model_state, plan_forward, plan_step

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
