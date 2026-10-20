"""EP quickstart geometry, replica ownership, precision, and objective scaling."""

from types import SimpleNamespace

import pytest
import torch

from workloads.recipes.text import expert_parallel as recipe


@pytest.fixture
def environment(monkeypatch):
    state = SimpleNamespace(rank=0, groups=[], destroyed=[], models=[])
    monkeypatch.setattr(recipe.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(recipe.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(recipe.dist, "get_rank", lambda: state.rank)

    def new_group(**kwargs):
        group = object()
        state.groups.append((group, kwargs))
        return group

    monkeypatch.setattr(recipe.dist, "new_group", new_group)
    monkeypatch.setattr(recipe.dist, "destroy_process_group", state.destroyed.append)
    monkeypatch.setattr(recipe.Precision, "apply", lambda self: None)

    class Model(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.expert = torch.nn.Parameter(torch.randn(2, 3))
            self.replica = torch.nn.Parameter(torch.randn(3, 3))
            self.closed = False
            state.models.append((config, kwargs, self))

        def expert_parameters(self):
            yield self.expert

        def close(self):
            self.closed = True

    from workloads.models.qwen3moe.implementations import mlops as qwen3moe

    monkeypatch.setattr(qwen3moe, "Model", Model)
    return state


def factory(**kwargs):
    return recipe.experiment(
        device="cuda:0",
        sequence_length=16,
        sequences_per_step=8,
        sequences_per_microbatch=2,
        config_overrides={"vocab_size": 512, "n_experts": 16},
        **kwargs,
    )


def test_exact_geometry_and_only_experts_are_unique(environment):
    experiment = factory()
    assert list(experiment["candidates"]) == ["2"]
    batches = experiment["candidates"]["2"]
    assert len(batches) == 4
    assert all(x.shape == (1, 32) and lengths == (16, 16) for x, _, lengths in batches)
    with experiment["context"]():
        model = experiment["model_factory"]()
        second = experiment["model_factory"]()
        torch.testing.assert_close(model.expert, second.expert)
        torch.testing.assert_close(model.replica, second.replica)
        _, options, _ = environment.models[0]
        assert options["token_capacity"] == 32
        assert options["parameter_device"] == "cpu"
        assert options["weight_grad_dtype"] == torch.float32
        assert options["dtype"] == torch.bfloat16
        ownership = experiment["distributed"](model)
        assert ownership.replica_group is environment.groups[0][0]
        assert ownership.replica_overrides == (((model.expert,), None),)
        assert ownership.transfer_sharding is None and not ownership.shard_gradients
        assert experiment["plan_options"]["master_dtype"] == torch.float32
        assert experiment["plan_options"]["grad_dtype"] == torch.float32
        assert experiment["plan_options"]["shard_optimizer"] is True
        assert experiment["optimizer"].keywords["opt_state_dtype"] == torch.float32
        experiment["cleanup_model"](model)
        experiment["cleanup_model"](second)
        assert model.closed and second.closed
    assert environment.destroyed == [environment.groups[0][0]]


def test_data_is_shared_between_geometries_but_distinct_between_ranks(environment):
    first = factory()
    second = recipe.experiment(
        device="cuda:0",
        sequence_length=16,
        sequences_per_step=8,
        sequences_per_microbatch=4,
        config_overrides={"vocab_size": 512, "n_experts": 16},
    )
    environment.rank = 1
    other = factory()

    def joined(exp):
        batches = next(iter(exp["candidates"].values()))
        return torch.cat([x for x, _, _ in batches], dim=1)

    assert torch.equal(joined(first), joined(second))
    assert not torch.equal(joined(first), joined(other))


def test_ep_objective_normalizes_global_tokens_once(environment, monkeypatch):
    experiment = factory()
    from workloads.recipes.text import quickstart

    parameter = torch.tensor(2.0, requires_grad=True)

    def objective(manifest, model, tokens, *_):
        return parameter.square() * tokens.numel() / manifest.tokens_per_step

    monkeypatch.setattr(quickstart, "full_model_objective", objective)
    for batch in experiment["candidates"]["2"]:
        loss, _ = experiment["objective"](None, *batch)
        loss.backward()
    # Each of two replicas contributes half of the global gradient sum.
    torch.testing.assert_close(parameter.grad, torch.tensor(2.0))
    assert experiment["metadata"]["global_tokens_per_step"] == 256
    assert experiment["units_per_step"] == 128


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model": "llama3"}, "model must be"),
        ({"model_dtype": "float32"}, "BF16"),
        ({"grad_dtype": "float16"}, "gradients"),
        ({"preparation_timeout": 0}, "positive"),
    ],
)
def test_bad_requests_fail_before_group_creation(environment, kwargs, match):
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)
    assert not environment.groups
