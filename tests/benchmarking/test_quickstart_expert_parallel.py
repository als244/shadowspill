"""EP recipe geometry, portable resources, ownership and bounded initialization."""

import pytest
import torch

from shadowspill import Resource
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.build.parallelism import model_parallelism
from workloads.recipes.text.expert_parallel import experiment


def factory(**kwargs):
    return experiment(
        device="cuda:0",
        sequence_length=16,
        sequences_per_step=8,
        sequences_per_example=2,
        ranks=2,
        config_overrides={
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "n_kv_heads": 2,
            "head_dim": 32,
            "d_ff_expert": 128,
            "vocab_size": 512,
            "n_experts": 16,
        },
        **kwargs,
    )


@pytest.fixture(autouse=True)
def no_live_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recipe construction opened a live GPU/group")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "get_world_size", forbidden)


def test_portable_recipe_preserves_geometry_and_ownership():
    setup = factory()
    batches = setup["candidates"]["2"]["args"]
    assert len(batches) == 4
    assert all(x.shape == (1, 32) and lengths == (16, 16) for x, _, lengths in batches)
    module = setup["model_factory"]()
    assert all(p.is_meta for p in module.parameters())
    options = setup["build_options"]
    resource = options["resources"]["experts"]
    assert Resource.from_dict(resource.to_dict()) == resource
    assert resource.groups["exchange"] == options["parallelism"].group("all_ranks")
    assert len(resource.options["layers"]) == 2
    _, ownership = model_parallelism(module, options["parallelism"])
    local = {id(p) for p in module.expert_parameters()}
    buffers = {name for name, _ in module.named_buffers(remove_duplicate=False)}
    for row in ownership:
        if row.name in buffers:
            # A buffer is stored whole on every rank.
            assert row.copy_groups == ((0, 1),)
            assert row.weights.groups == ((0,), (1,))
            continue
        expert = id(module.get_parameter(row.name)) in local if row.trainable else False
        assert row.copy_groups == (((0,), (1,)) if expert else ((0, 1),))
        assert row.optimizer.groups == (((0,), (1,)) if expert else ((0, 1),))
    assert options["master_dtype"] == options["grad_dtype"] == torch.float32
    assert (
        setup["units_per_step"] == 256 and setup["metadata"]["units_scope"] == "global"
    )
    module.to_empty(device="cpu")
    from tests.shadowspill.execution.expert_parallel.reference import payloads

    for name, tensor in payloads(module):
        setup["fill_tensor"](
            TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, tensor.numel()),
            tensor.reshape(-1),
        )
        assert torch.isfinite(tensor).all(), name
    module.close()


def test_rank_local_data_and_shared_dense_initialization():
    first, second = factory(), factory(logical_rank=1)
    assert not torch.equal(
        first["candidates"]["2"]["args"][0][0],
        second["candidates"]["2"]["args"][0][0],
    )
    for name, equal in (
        ("blocks.0.moe.experts.router_weight._data", True),
        ("blocks.0.moe.experts.gate_up_weight._data", False),
        ("embed.weight", True),
    ):
        request = TensorChunk(name, (16, 128), torch.bfloat16, 0, 2048)
        values = []
        for setup in (first, second):
            data = torch.empty(2048, dtype=torch.bfloat16)
            setup["fill_tensor"](request, data)
            assert torch.isfinite(data).all()
            values.append(data)
        assert torch.equal(*values) == equal


def test_ep_objective_normalizes_global_tokens_once(monkeypatch):
    from workloads.recipes.text.benchmark import factory as common

    setup = factory()
    parameter = torch.tensor(2.0, requires_grad=True)

    def objective(manifest, model, tokens, *_):
        return parameter.square() * tokens.numel() / manifest.tokens_per_step

    monkeypatch.setattr(common, "full_model_objective", objective)
    for batch in setup["candidates"]["2"]["args"]:
        loss, _ = setup["objective"](None, *batch)
        loss.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(2.0))


@pytest.mark.parametrize("logical_rank", [0, 1])
def test_explicit_masters_match_packed_payloads_across_chunk_sizes(logical_rank):
    loader = factory(logical_rank=logical_rank)["fill_tensor"]
    for leaf in ("router_weight", "gate_up_weight", "down_weight"):
        name = "blocks.0.moe.experts." + leaf
        payload = torch.empty(35, dtype=torch.bfloat16)
        master = torch.empty(35, dtype=torch.float32)
        loader(TensorChunk(name + "._data", (5, 7), payload.dtype, 0, 35), payload)
        for start in range(0, 35, 9):
            chunk = master[start : start + 9]
            loader(
                TensorChunk("master." + name, (5, 7), master.dtype, start, 35), chunk
            )
        torch.testing.assert_close(master, payload.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model": "llama3-8b"}, "model must be"),
        ({"model_dtype": "float32"}, "BF16"),
        ({"grad_dtype": "float16"}, "gradients"),
    ],
)
def test_bad_requests_fail_before_live_resources(kwargs, match):
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)
