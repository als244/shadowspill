"""Portable expert declarations retain the existing provider and group contract."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from shadowspill import Parallelism, Resource
from shadowspill_workloads.definitions.expert_parallel import describe, ownership
from shadowspill_workloads.models.shared.expert_parallel import resources


def dimensions(**changes):
    return SimpleNamespace(
        **{
            **dict(
                n_layers=2,
                n_experts=8,
                top_k=2,
                d_model=128,
                d_ff_expert=256,
                initializer_range=0.017,
            ),
            **changes,
        }
    )


@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe", "olmoe"])
@pytest.mark.parametrize("grad_dtype", ["float32", "bfloat16"])
def test_description_and_runtime_factory_preserve_provider_defaults_and_groups(
    family, grad_dtype, monkeypatch
):
    from mlops.expert_parallel import QuackMoEConfig
    from mlops.expert_parallel.quack import resources as provider
    from mlops.expert_parallel.quack.config import _configuration

    parallelism = ownership(2, 1)
    resource = describe(dimensions(), family, parallelism, 64, grad_dtype=grad_dtype)
    replay = Resource.from_dict(json.loads(json.dumps(resource.to_dict())))
    assert replay == resource
    assert resource.groups == {"exchange": parallelism.group("all_ranks")}
    expected = QuackMoEConfig(
        ep_size=2,
        num_experts=8,
        top_k=2,
        model_dim=128,
        expert_hidden_dim=256,
        tokens_per_rank=64,
        router_dtype=torch.float32 if family == "olmoe" else torch.bfloat16,
        router_weight_grad_dtype=torch.float32
        if family == "olmoe"
        else getattr(torch, grad_dtype),
        weight_grad_dtype=getattr(torch, grad_dtype),
        renormalize_topk=family != "olmoe",
        share_expert_banks=True,
        init_std=0.017,
        gemm_tuned=False,
    )
    assert resources.configuration(resource.options["config"]) == expected
    result, group = object(), object()
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(provider, "initialize", initialize)
    assert (
        resources.initialize(
            name="experts",
            device="cuda:1",
            groups={"exchange": group},
            options=resource.options,
        )
        is result
    )
    assert len(calls) == 1
    actual = calls[0]
    assert actual["name"] == "experts" and actual["device"] == "cuda:1"
    assert actual["groups"] == {"exchange": group}
    assert [r["name"] for r in actual["options"]["layers"]] == [
        "blocks.0.moe.experts",
        "blocks.1.moe.experts",
    ]
    assert all(
        _configuration(row["config"]) == expected for row in actual["options"]["layers"]
    )


def test_description_needs_no_framework_provider_or_recipe_imports():
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, json, sys
from types import SimpleNamespace
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if (name.split('.')[0] in ('torch','mlops','triton','pynvml')
                or 'recipes' in name):
            raise AssertionError('description imported ' + name)
sys.meta_path.insert(0, Block())
from shadowspill import Parallelism, Resource
from shadowspill_workloads.definitions.expert_parallel import describe, ownership
config = SimpleNamespace(n_layers=2, n_experts=8, top_k=2, d_model=128,
                         d_ff_expert=256, initializer_range=0.017)
value = describe(config, 'qwen3moe', ownership(2, 1),
                 64, grad_dtype='float32')
assert Resource.from_dict(json.loads(json.dumps(value.to_dict()))) == value
assert not any(name in sys.modules for name in ('torch','mlops','triton'))
""",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe", "olmoe"])
def test_existing_recipe_constructs_meta_model_with_portable_resources(
    family, monkeypatch
):
    from benchmarks.factories.expert_parallel import experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("declarative model construction touched a live group")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "get_world_size", forbidden)
    setup = experiment(
        device="meta",
        model=family + "-tiny",
        ranks=2,
        sequence_length=4,
        sequences_per_step=2,
        sequences_per_example=2,
        config_overrides=dict(
            n_layers=1,
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            n_experts=4,
            top_k=2,
            d_ff_expert=128,
        ),
    )
    resource = setup["build_options"]["resources"]["experts"]
    assert resource.options["config"]["tokens_per_rank"] == 8
    model = setup["model_factory"]()
    try:
        assert all(parameter.is_meta for parameter in model.parameters())
        experts = list(model.expert_parameters())
        assert len(experts) == 2 and all(value.shape[0] == 2 for value in experts)
    finally:
        model.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"n_layers": 0},
        {"n_experts": 7},
        {"top_k": 9},
        {"d_model": 64},
        {"d_ff_expert": 129},
        {"initializer_range": float("nan")},
    ],
)
def test_invalid_shapes_fail_before_worker_construction(changes):
    with pytest.raises(ValueError):
        describe(
            dimensions(**changes),
            "qwen3moe",
            Parallelism.dp(ranks_per_replica=2),
            64,
            grad_dtype="float32",
        )


def test_provider_failure_is_propagated_without_another_resource(monkeypatch):
    from mlops.expert_parallel.quack import resources as provider

    resource = describe(
        dimensions(), "qwen3moe", ownership(2, 1), 64, grad_dtype="float32"
    )
    failure = RuntimeError("provider could not create the shared buffer")
    calls = []

    def failed(**kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setattr(provider, "initialize", failed)
    with pytest.raises(RuntimeError) as error:
        resources.initialize(
            name="experts", device="cuda:0", groups={}, options=resource.options
        )
    assert error.value is failure and len(calls) == 1
