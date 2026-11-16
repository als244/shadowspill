"""Router statistics keep their physical microbatch across execution chunks."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.task.inputs import TaskInputRole
from shadowspill_workloads.models.shared.execution_policies import chunks
from shadowspill_workloads.models.shared.router_statistics import masked_statistics


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("experts", [4, 17])
@pytest.mark.parametrize("real_tokens", [0, 13])
def test_masked_router_counts_probabilities_and_vjp_keep_complete_microbatch(
    save, experts, real_tokens
):
    torch.manual_seed(137)
    inputs = (
        torch.randn(1, 17, 5, requires_grad=True),
        torch.randn(experts, 5, requires_grad=True),
        torch.arange(17)[None, :] < real_tokens,
    )

    def region(hidden, weight, mask):
        return masked_statistics(
            hidden, weight, mask, top_k=2, routing_mode="topk_then_softmax"
        )

    expected = region(*inputs)
    tangent = torch.randn_like(expected[1])
    gradients = torch.autograd.grad(expected[1], inputs[:2], tangent)
    pair = capture_graph_pair(
        make_fx(region)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION),
            TaskInputProvenance(TaskInputRole.PARAMETER),
            TaskInputProvenance(TaskInputRole.CONTROL),
        ),
    )
    policy = chunks(sequence_length=17, batch_size=1, chunk_size=7, expert_chunk_size=7)
    forward = region_artifact(
        pair.forward, policy, stage_id="router", variant="save" if save else "recompute"
    )
    assert any(child.chunking for child in forward.regions.children)
    saved = forward.regions.output_domains[-pair.saved_value_count :]
    backward = region_artifact(
        pair.backward,
        policy,
        stage_id="router",
        variant="save" if save else "recompute",
        input_domains=(*saved, forward.regions.output_domains[1]),
        domain_rules=forward.regions.domain_rules,
        gradient_leaves=(1,),
    )
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        derivatives = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[:2], expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(derivatives, (*gradients, None), rtol=3e-5, atol=3e-6)
    assert actual[0].sum() == 2 * real_tokens
