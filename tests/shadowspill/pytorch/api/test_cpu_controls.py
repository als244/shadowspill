"""CPU-valued task outputs and SSD-backed producer slices use normal APIs."""

import copy
from contextlib import ExitStack

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    plan_forward,
    release_model_state,
)
from shadowspill.pytorch.state.initialization import pool_values
from shadowspill.ssd import ssd
from tests.shadowspill.pytorch.state.representations import ScaledWeight

pytestmark = [pytest.mark.cuda, pytest.mark.fresh_process]


@torch.library.custom_op("shadowspill_test::cpu_controls", mutates_args=())
def _controls(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert value.device.type == "cuda"
    return (value.sum(1) > 0).to(device="cpu", dtype=torch.int64), value.sin()


@_controls.register_fake
def _controls_fake(value):
    return torch.empty(value.shape[0], dtype=torch.int64, device="cpu"), (
        torch.empty_like(value)
    )


@torch.library.custom_op("shadowspill_test::use_cpu_controls", mutates_args=())
def _use_controls(value: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    assert control.device.type == "cpu"
    assert value.device.type == "cuda"
    return value * (control.to(value.device)[:, None] + 1)


@_use_controls.register_fake
def _use_controls_fake(value, control):
    assert control.device.type == "cpu"
    return torch.empty_like(value)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32) * 0.1)

    def forward(self, value, control):
        return _use_controls(value * self.weight, control)


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            ScaledWeight(
                torch.randint(-8, 8, (32, 32), dtype=torch.int8),
                torch.tensor(0.025),
            ),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, value):
        control, value = _controls(value @ self.weight.T)
        for block in self.blocks:
            value = block(value, control)
        return control, value


def test_cpu_outputs_and_mixed_device_controls_from_ssd(tmp_path):
    torch.manual_seed(901)
    model = _Network().eval()
    reference = copy.deepcopy(model)
    inputs = torch.randn(3, 32)
    with ExitStack() as stack:
        runtime = stack.enter_context(
            Runtime(
                pools={
                    "execution": device(physical_capacity=2 << 30),
                    "spill": ssd(capacity=512 << 20, directory=tmp_path),
                },
                routes={
                    "fetch": transfer_route(source="spill", destination="execution"),
                    "evict": transfer_route(source="execution", destination="spill"),
                },
                calibrate=False,
            )
        )
        runtime.calibrate_transfer_capabilities(
            large_copy_bytes=4 << 20, warmup_copies=1, measured_copies=2
        )
        reference.cuda()
        model = import_model_state(
            model, runtime=runtime, pool="spill", release_source=True
        )
        stack.callback(release_model_state, model, runtime=runtime)
        # Physical wrapper leaves are read, mutated, and restored to metadata
        # views by the same setup context used by authentic control derivation.
        replacement = torch.randn(32, 32) * 0.1
        with torch.no_grad(), pool_values(runtime):
            returned = model.weight.copy_(replacement)
            assert returned is model.weight
            actual = model.weight.to("cuda")
        with torch.no_grad():
            reference.weight.copy_(replacement.cuda())
        torch.testing.assert_close(actual.dense(), reference.weight.dense())
        del actual

        with plan_forward(
            model,
            example_inputs=[inputs],
            runtime=runtime,
            execution="execution",
            spill="spill",
            artifact_store=tmp_path / "artifacts",
            profiling_options=CORRECTNESS_PROFILING,
        ) as forward:
            assert forward.plan_report.captured_stage_count == 3
            for multiplier in (1, -1, 2):
                sample = inputs * multiplier
                expected = reference(sample.cuda())
                actual = forward([sample])
                assert actual[0].device.type == "cpu"
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])
                del actual
