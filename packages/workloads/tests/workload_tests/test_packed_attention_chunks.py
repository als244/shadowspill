"""Packed attention outputs keep token axes when head width matches capacity."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.task.inputs import TaskInputRole
from shadowspill_workloads.models.shared.execution_policies import chunks


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_packed_attention_head_width_does_not_hide_token_coordinates(batch, save):
    torch.manual_seed(137)
    tokens = width = 17
    heads = 3
    inputs = (
        torch.randn(
            batch * tokens, heads, width, dtype=torch.float64, requires_grad=True
        ),
        torch.randn(heads * width, 13, dtype=torch.float64, requires_grad=True),
        torch.randn(batch, tokens, 13, dtype=torch.float64, requires_grad=True),
    )

    def region(attention, weight, residual):
        return (
            attention.reshape(batch, tokens, heads * width) @ weight + residual
        ).tanh()

    expected = region(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(expected, inputs, tangent)
    pair = capture_graph_pair(
        make_fx(region)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION),
            TaskInputProvenance(TaskInputRole.PARAMETER),
            TaskInputProvenance(TaskInputRole.ACTIVATION),
        ),
    )
    policy = chunks(
        sequence_length=tokens, batch_size=batch, chunk_size=7, expert_chunk_size=7
    )
    variant = "save" if save else "recompute"
    forward = region_artifact(
        pair.forward, policy, stage_id="attention", variant=variant
    )
    assert any(child.chunking for child in forward.regions.children)
    saved = forward.regions.output_domains[-pair.saved_value_count :]
    backward = region_artifact(
        pair.backward,
        policy,
        stage_id="attention",
        variant=variant,
        input_domains=(*saved, forward.regions.output_domains[0]),
        domain_rules=forward.regions.domain_rules,
        gradient_leaves=(1,),
    )
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        derivatives = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(derivatives, gradients, rtol=1e-11, atol=1e-11)
