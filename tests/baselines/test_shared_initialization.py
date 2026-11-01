"""FSDP2 and DeepSpeed start from the values ShadowSpill's engine fills."""

import math

import pytest
import torch

from baselines.common.initialization import (
    SharedInitialization,
    fill_parameters,
    fresh,
    shard_rows,
)
from baselines.common.training import pinned_parameters
from baselines.engines.shadowspill.initialize import Initialize
from baselines.engines.shadowspill.settings import Settings
from shadowspill.artifacts.tensors import TensorChunk


def filled_by_shadowspill(fill, name, shape, dtype):
    total = math.prod(shape)
    values = torch.empty(total, dtype=dtype)
    fill(TensorChunk("model." + name, shape, dtype, 0, total), values)
    return values.view(shape)


@pytest.mark.parametrize("family", ["llama3", "qwen35", "qwen3moe"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_every_parameter_starts_as_shadowspill_stores_it(family, mode):
    settings = Settings(model=f"{family}-tiny", mode=mode, seed=3)
    request = settings.request()
    # FSDP2 builds at the stepped dtype: float32 masters over bfloat16
    # weights, which ShadowSpill widens from the weights it stores.
    model = request.build(dtype="float32", device="cpu")
    shared = SharedInitialization(request, settings.seed, model_dtype="bfloat16")
    written = fill_parameters(model, shared)
    assert written == [name for name, _ in model.named_parameters()]
    with Initialize(settings) as fill:
        for name, parameter in model.named_parameters():
            stored = filled_by_shadowspill(
                fill, name, tuple(parameter.shape), shared.stored[name]
            )
            torch.testing.assert_close(
                parameter, stored.to(parameter.dtype), rtol=0, atol=0
            )
    # Parameters the model pins at float32 are drawn at float32.
    assert all(
        shared.stored[name] == torch.float32 for name in pinned_parameters(request)
    )


def test_a_checkpointed_run_draws_only_its_fresh_factors():
    settings = Settings(model="llama3-tiny", mode="lora", seed=5)
    request = settings.request()
    model = request.build(dtype="bfloat16", device="cpu")
    before = {name: value.clone() for name, value in model.named_parameters()}
    shared = SharedInitialization(request, settings.seed, model_dtype="bfloat16")
    written = fill_parameters(model, shared, only=fresh)
    assert written and all(fresh(name) for name in written)
    for name, value in model.named_parameters():
        if name not in written:
            assert torch.equal(value, before[name])


def test_a_rank_draws_its_rows_alone():
    settings = Settings(model="llama3-tiny", seed=7)
    request = settings.request()
    shared = SharedInitialization(request, settings.seed, model_dtype="bfloat16")
    name, parameter = next(
        (name, p) for name, p in request.build(dtype="bfloat16").named_parameters()
    )
    shape = tuple(parameter.shape)
    whole = shared.values(name, shape)
    row = math.prod(shape[1:])
    for world in (1, 2, 3):
        pieces = torch.arange(shape[0]).chunk(world)
        for rank in range(world):
            first, count = shard_rows(shape[0], world, rank)
            if rank < len(pieces):
                assert (first, count) == (int(pieces[rank][0]), len(pieces[rank]))
            else:
                assert count == 0
            part = shared.values(name, shape, first * row, (first + count) * row)
            assert torch.equal(part, whole[first * row : (first + count) * row])


def test_shard_rows_leaves_the_last_ranks_empty_as_torch_chunk_does():
    assert [shard_rows(5, 4, rank) for rank in range(4)] == [
        (0, 2),
        (2, 2),
        (4, 1),
        (5, 0),
    ]
