"""Frozen embedding prologues preserve flattened normalization coordinates."""

import mlops
import pytest
import torch

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.task.inputs import TaskInputRole
from shadowspill_workloads.policies.text import chunks


@pytest.mark.cuda
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("heads", [2, 4])
def test_frozen_embedding_normalization_vjp_preserves_token_head_statistics(
    save, batch, heads
):
    torch.manual_seed(137)
    tokens, width, rank = 17, 8, 3
    inputs = (
        torch.randint(31, (batch, tokens), device="cuda"),
        torch.randn(31, heads * width, device="cuda"),
        torch.randn(heads * width, rank, device="cuda", requires_grad=True),
        torch.randn(rank, heads * width, device="cuda", requires_grad=True),
        torch.ones(width, device="cuda"),
        torch.arange(batch + 1, device="cuda", dtype=torch.int32) * tokens,
    )

    def region(ids, embedding, a, b, scale, lengths):
        hidden = torch.nn.functional.embedding(ids, embedding)
        projected = (hidden + (hidden @ a) @ b).view(batch, tokens, heads, width)
        normalized = mlops.rms_norm(projected, scale, 1e-6)
        # This complete-sequence operation separates the normalization from
        # its backward. The provider stores inverse RMS as flat token/head rows.
        packed = normalized.to(torch.bfloat16).reshape(batch * tokens, heads, width)
        return (
            mlops.flash_attention(packed, packed, packed, lengths, tokens)
            .float()
            .tanh()
        )

    expected = region(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(expected, inputs[2:4], tangent)

    class Model(torch.nn.Module):
        def forward(self, *values):
            return region(*values)

    graph = torch.export.export(Model(), inputs, strict=True).module()
    pair = capture_graph_pair(
        graph,
        inputs,
        original_output=expected,
        memory_budget=float(save),
        input_provenance=tuple(
            TaskInputProvenance(role)
            for role in (
                TaskInputRole.CONTROL,
                *(TaskInputRole.PARAMETER,) * 4,
                TaskInputRole.CONTROL,
            )
        ),
    )
    policy = chunks(
        sequence_length=tokens, batch_size=batch, chunk_size=7, expert_chunk_size=7
    )
    variant = "save" if save else "recompute"
    forward = region_artifact(pair.forward, policy, stage_id="first", variant=variant)
    saved = forward.regions.output_domains[-pair.saved_value_count :]
    backward = region_artifact(
        pair.backward,
        policy,
        stage_id="first",
        variant=variant,
        input_domains=(*saved, forward.regions.output_domains[0]),
        domain_rules=forward.regions.domain_rules,
        gradient_leaves=(2, 3),
    )
    assert any(child.chunking for child in backward.regions.children)
    with torch.no_grad():
        actual = executable(forward)(*inputs)
        derivatives = executable(backward)(*actual[-pair.saved_value_count :], tangent)
    torch.testing.assert_close(actual[0], expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        derivatives, (None, None, *gradients, None, None), rtol=2e-4, atol=2e-5
    )
