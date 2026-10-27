"""An optional storage cutoff preserves compute ownership and gradient SUMs."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from shadowspill import Parallelism, ParameterStorage, Sharded
from shadowspill.build.parallelism import model_parallelism
from shadowspill.parallelism.reductions import gradient_reduction


def test_cutoff_uses_physical_dtype_and_unpadded_bytes():
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(1024, dtype=torch.bfloat16, device="meta"))
    model.bias = nn.Parameter(torch.empty(32, device="meta"))
    spec = Parallelism.dp(ranks_per_replica=4, num_replicas=2, min_shard_bytes=2048)
    _, (weight, bias) = model_parallelism(model, spec)
    assert weight.weights.capacity == 256  # equality still shards
    assert (
        bias.weights.capacity
        == bias.gradients.capacity
        == bias.optimizer.capacity
        == 32
    )
    reduction = gradient_reduction(bias)
    assert [(p.operation, p.groups) for p in reduction.phases] == [
        ("all_reduce", (tuple(range(8)),))
    ]
    assert bias.copies(0) == tuple(range(8))
    loaded = Parallelism.from_dict(spec.to_dict())
    assert model_parallelism(model, loaded)[1] == (weight, bias)


def test_cutoff_is_opt_in_and_exact_override_can_disable_it():
    model = nn.Linear(8, 4, device="meta")
    spec = Parallelism.dp(ranks_per_replica=4)
    _, (weight, bias) = model_parallelism(model, spec)
    assert weight.weights.capacity == 8 and bias.weights.capacity == 1
    spec = replace(
        Parallelism.dp(ranks_per_replica=4, min_shard_bytes=2 << 20),
        parameter_storage={"weight": ParameterStorage(min_shard_bytes=0)},
    )
    _, (weight, bias) = model_parallelism(model, spec)
    assert weight.weights.capacity == 8 and bias.weights.capacity == 4
    assert weight.optimizer.capacity == 8 and bias.optimizer.capacity == 4


def test_cutoff_does_not_replicate_different_experts():
    model = nn.Linear(8, 4, bias=False, device="meta")
    spec = Parallelism(
        ranks_per_replica=4,
        num_replicas=2,
        groups={"exchange": ((0, 1), (2, 3)), "copies": ((0, 2), (1, 3))},
        default_parameter_placement=Sharded("exchange", 0),
        default_parameter_storage=ParameterStorage(
            group="copies",
            weights="sharded",
            gradients="sharded",
            optimizer="sharded",
            min_shard_bytes=2 << 20,
        ),
    )
    _, (weight,) = model_parallelism(model, spec)
    assert weight.local_shape == (4, 8) and weight.global_shape == (8, 8)
    assert weight.weights.capacity == 32
    assert weight.copy_groups == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert gradient_reduction(weight).phases[0].groups == weight.copy_groups


@pytest.mark.parametrize("invalid", [-1, 2.5, True])
def test_cutoff_rejects_invalid_byte_counts(invalid):
    with pytest.raises(ValueError, match="min_shard_bytes"):
        Parallelism.dp(min_shard_bytes=invalid)
