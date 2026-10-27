"""Saved forward tasks read packed SSD weights and pass data-dependent CPU controls."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward
from tests.shadowspill.execution._host_controls import host_control, read_host_control
from tests.shadowspill.execution.helpers import run_worker, save_plan
from tests.shadowspill.pytorch.representation_fixtures import ScaledWeight


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32) * 0.1)

    def forward(self, value, control):
        return read_host_control(value * self.weight, control)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            ScaledWeight(
                torch.randint(-8, 8, (32, 32), dtype=torch.int8),
                torch.tensor(0.025),
            ),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, value):
        value = value @ self.weight.T
        control, value = host_control(value), value.sin()
        for block in self.blocks:
            value = block(value, control)
        return control, value


@pytest.mark.cuda
def test_packed_weights_and_host_controls_from_ssd(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.delenv("SHADOWSPILL_TEST_SPILL_PEER", raising=False)
    monkeypatch.setenv("SHADOWSPILL_TEST_SSD_DIRECTORY", str(tmp_path))
    torch.manual_seed(901)
    reference, inputs = Network().eval(), torch.randn(3, 32)
    with torch.device("meta"):
        template = Network().eval()
    program = build_forward(template, example_args=(inputs,))
    assert len(program.bindings["entrypoints"]) == 3
    save_plan(program, tmp_path)
    state = {
        "weight.payload": reference.weight.payload.detach().clone(),
        "weight.scale": reference.weight.scale.detach().clone(),
        **{
            name: value.detach().clone()
            for name, value in reference.named_parameters()
            if name.startswith("blocks.")
        },
    }
    calls = [[((inputs * multiplier,), {})] for multiplier in (1, -1, 2)]
    with torch.no_grad():
        expected = [(reference(*call[0][0]), state) for call in calls]
    assert not torch.equal(expected[0][0][0], expected[1][0][0])
    torch.save(
        dict(
            initial=state,
            examples=calls[0],
            invocation_examples=calls,
            expected=expected,
            queued_outputs=[expected[0][0]] * 2,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
