"""Packed sequence boundaries stay whole while token-local work is chunked."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill_workloads.models.shared.decoder import packed_metadata
from shadowspill_workloads.models.shared.execution import text_execution_chunks


@pytest.mark.parametrize("lengths", [(7, 0, 10), (1, 15, 1), (17, 0, 0)])
def test_variable_packing_keeps_complete_boundaries_and_token_positions(lengths):
    inputs = (
        torch.arange(17).reshape(1, 17),
        torch.tensor(lengths, dtype=torch.int32),
        torch.arange(17).reshape(1, 17) % 3 != 0,
    )

    def region(tokens, lengths, mask):
        active = torch.zeros_like(mask) | mask
        packing = packed_metadata(tokens, lengths, capacity=17)
        return torch.where(active, tokens, 0), packing.positions, packing.cu_seqlens

    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(region)(*inputs),
        example_inputs=inputs,
    )
    policy = text_execution_chunks(
        sequence_length=17,
        batch_size=1,
        chunk_size=7,
        expert_chunk_size=7,
        rules=(),
        expert_targets=(),
    )
    captured = region_artifact(artifact, policy, stage_id="metadata", variant=None)
    assert any(child.chunking for child in captured.regions.children)
    for child in captured.regions.children:
        if {"aten.cumsum.default", "aten.searchsorted.Tensor"} & set(
            child.operator_targets
        ):
            assert child.chunking is None
    # Reuse the same graph over empty and changed boundary slots.
    runner = executable(captured)
    for geometry in ((7, 0, 10), (1, 15, 1), (17, 0, 0)):
        values = inputs[0], torch.tensor(geometry, dtype=torch.int32), inputs[2]
        torch.testing.assert_close(runner(*values), region(*values), rtol=0, atol=0)
