"""GLM uses the common packed metadata as live tensors, including empty slots."""

from importlib import import_module
from itertools import accumulate

import pytest
import torch
from torch.export.graph_signature import InputKind

from shadowspill.pytorch.capture.export import capture_forward
from shadowspill_workloads.models.glm53.implementations.mlops.attention import (
    positions_for,
)
from shadowspill_workloads.models.shared.glm.inputs import packed_inputs, token_features

GEOMETRIES = ((65, 0, 64, 0), (1, 127, 1, 0), (129, 0, 0, 0), (0, 65, 0, 64))


class Metadata(torch.nn.Module):
    def forward(self, tokens, lengths):
        flat, cumulative, chunks = packed_inputs(tokens, lengths)
        return positions_for(cumulative, flat), cumulative, chunks


def _arguments(lengths, device="cpu"):
    return (
        torch.arange(sum(lengths), device=device).reshape(1, -1),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )


def _expected(lengths):
    boundaries = [0, *accumulate(lengths)]
    pairs = [
        (sequence, chunk)
        for sequence, length in enumerate(lengths)
        for chunk in range((length + 63) // 64)
    ]
    bound = (sum(lengths) + 63) // 64 + len(lengths)
    pairs.extend((len(lengths), index) for index in range(bound - len(pairs)))
    return (
        torch.tensor([p for length in lengths for p in range(length)]),
        torch.tensor([*boundaries, boundaries[-1]], dtype=torch.int64),
        torch.tensor(pairs, dtype=torch.int64),
    )


def test_glm_export_keeps_lengths_as_input_and_shares_live_boundaries():
    exported = capture_forward(Metadata(), _arguments(GEOMETRIES[0])).exported_program
    assert [spec.kind for spec in exported.graph_signature.input_specs] == [
        InputKind.USER_INPUT,
        InputKind.USER_INPUT,
    ]
    assert not exported.constants
    for lengths in GEOMETRIES:
        torch.testing.assert_close(
            exported.module()(*_arguments(lengths)), _expected(lengths), rtol=0, atol=0
        )


def test_token_features_preserves_head_and_shapes_and_one_metadata_description():
    class Backbone(torch.nn.Module):
        token_features = token_features

        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(3, 8)

        def hidden(self, flat, boundaries, cumulative, chunks):
            assert boundaries is cumulative
            assert flat.ndim == 1
            assert chunks.shape[1] == 2
            return flat[:, None].float().expand(-1, 3)

    model = Backbone()
    tokens, lengths = _arguments(GEOMETRIES[0])
    features = model.token_features(tokens, lengths)
    assert features["head"] is model.lm_head
    torch.testing.assert_close(
        features["head_hidden"], tokens[..., None].float().expand(-1, -1, 3)
    )
    # Ordinary rectangular batches also get explicit metadata, even one sequence.
    flat, cumulative, chunks = packed_inputs(torch.zeros(2, 17, dtype=torch.int64))
    assert flat.shape == (34,)
    assert cumulative.tolist() == [0, 17, 34, 34]
    assert chunks.shape == (3, 2)
    assert packed_inputs(torch.zeros(1, 17, dtype=torch.int64))[1].tolist() == [
        0,
        17,
        17,
    ]


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
def test_both_glm_families_expose_shared_head_features(family):
    module = import_module(
        f"shadowspill_workloads.models.{family}.implementations.mlops"
    )
    assert module.Model.token_features is token_features


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
def test_glm_runtime_packing_matches_compiled_host_metadata_loss_and_lora_gradients(
    family,
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = import_module(f"shadowspill_workloads.models.{family}.config").Config
    model_class = import_module(
        f"shadowspill_workloads.models.{family}.implementations.mlops"
    ).Model
    torch.manual_seed(20261117)
    model = model_class(
        config.tiny(
            hidden_size=128,
            lora_rank=16,
            lora_factor_dtype="bfloat16",
            index_topk=16,
            num_experts_per_tok=4,
        ),
        device="cuda",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                parameter.normal_(std=0.01)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    tokens = torch.randint(model.config.vocab_size, (1, 128), device="cuda")
    targets = torch.randint(model.config.vocab_size, tokens.shape, device="cuda")

    def loss(tokens, lengths, targets):
        features = model.token_features(tokens, lengths)
        return features["head"].loss(features["head_hidden"], targets, chunk_size=32)

    compilations = []

    def backend(graph, values):
        compilations.append(graph)
        return torch._inductor.compile(graph, values)

    from torch._inductor import config as inductor_config

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        inductor_config.patch(emulate_precision_casts=True),
        torch.cuda.stream(stream),
    ):
        compiled = torch.compile(loss, fullgraph=True, dynamic=False, backend=backend)
        reference = torch.compile(model.loss, fullgraph=True, dynamic=False)
        for lengths in ((0, 32, 96, 0), (64, 0, 32, 32), (128, 0, 0, 0)):
            metadata = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            _, cumulative, chunks = packed_inputs(tokens, metadata)
            # The independent existing host-boundary indexer is the reference;
            # omit empty slots there, but preserve KDA's full runtime metadata.
            boundaries = tuple([0, *accumulate(n for n in lengths if n)])
            expected = reference(
                tokens.reshape(-1),
                targets.reshape(-1),
                boundaries,
                cumulative,
                chunks,
                chunk_size=32,
            )
            expected_gradients = torch.autograd.grad(expected, parameters)
            actual = compiled(tokens, metadata, targets)
            gradients = torch.autograd.grad(actual, parameters)
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
            for got, want in zip(gradients, expected_gradients, strict=True):
                assert torch.isfinite(got).all()
                relative = (
                    got.float() - want.float()
                ).norm() / want.float().norm().clamp_min(1e-8)
                assert float(relative) < 0.05, float(relative)
        # Capture the complete loss/backward after warmup. Any provider reading
        # sequence metadata back to the host would fail CUDA graph capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay_loss = compiled(tokens, metadata, targets)
            replay_gradients = torch.autograd.grad(replay_loss, parameters)
        changed = (64, 0, 32, 32)
        metadata.copy_(torch.tensor(changed, device="cuda", dtype=torch.int32))
        _, cumulative, chunks = packed_inputs(tokens, metadata)
        expected = reference(
            tokens.reshape(-1),
            targets.reshape(-1),
            (0, 64, 96, 128),
            cumulative,
            chunks,
            chunk_size=32,
        )
        expected_gradients = torch.autograd.grad(expected, parameters)
        graph.replay()
        torch.testing.assert_close(replay_loss, expected, rtol=0.01, atol=0.01)
        for got, want in zip(replay_gradients, expected_gradients, strict=True):
            assert torch.isfinite(got).all()
            relative = (
                got.float() - want.float()
            ).norm() / want.float().norm().clamp_min(1e-8)
            assert float(relative) < 0.05, float(relative)
    torch.cuda.current_stream().wait_stream(stream)
    assert len(compilations) == 1
