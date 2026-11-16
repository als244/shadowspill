"""Installed EP setup reuses the provider, parameter ownership and bounded loader."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from shadowspill import Parallelism, ParameterStorage, Replicated
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from shadowspill.training.definitions import (
    InitializationContext,
    InitializationRuntime,
    load_registry,
)
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_workloads.definitions.expert_parallel import ownership

FAMILIES = ("qwen3moe", "qwen35moe", "olmoe")


def settings(family, *, grad_dtype="float32"):
    return Model(
        f"shadowspill/{family}@1",
        implementation="mlops_ep",
        precision={"grad_dtype": grad_dtype},
        config={
            "architecture": dict(
                n_layers=2,
                d_model=128,
                n_heads=4,
                n_kv_heads=2,
                head_dim=32,
                n_experts=4,
                top_k=2,
                d_ff_expert=128,
                vocab_size=97,
                max_seq_len=8,
            )
        },
        initialize=Weights.random(seed=17),
    )


def prepared(family, *, parallelism=None, grad_dtype="float32"):
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    parallelism = ownership(2, 1) if parallelism is None else parallelism
    parallelism = replace(
        parallelism,
        inputs=InputPlacement.independent(
            parallelism.world_size,
            fields=[
                InputFieldPlacement(path)
                for path in (
                    ("inputs", "tokens"),
                    ("loss_inputs", "target_tokens"),
                    ("loss_inputs", "loss_mask"),
                    ("loss_inputs", "weights"),
                    ("loss_inputs", "router_mask"),
                )
            ],
        ),
    )
    setup = resolve_training(
        registry,
        model=settings(family, grad_dtype=grad_dtype),
        objective=Objective.sum(
            {
                "tokens": Objective("shadowspill/cross_entropy@1"),
                "balance": Objective("shadowspill/moe_balance@1"),
            },
            weights={"balance": 0.001},
        ),
        batch=BatchConfig(
            limits={"token_capacity": 32, "max_sequence_length": 4, "max_sequences": 8}
        ),
        memory=Memory(1, 1),
        parallelism=parallelism,
        optimizer=Optimizer("mlops/adamw@1", precision={"gradient_dtype": grad_dtype}),
        operations={
            "step": ["metrics"],
            "forward_backward": ["metrics"],
            "forward": ["metrics", "logprobs"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[8, 16]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    return setup, capture_inventory(prepare_layouts(setup))


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_shapes_have_distinct_resources_and_operations_reuse_them(family):
    setup, inventory = prepared(family)
    resources, uses = {}, {}
    for template in inventory.templates.values():
        build = template.model_build
        assert build.grad_dtype == "float32"
        assert not build.operator_preparation  # No single-device Sonic preparation.
        assert len(build.resources) == 1
        name, resource = next(iter(build.resources.items()))
        size = resource.options["config"]["tokens_per_rank"]
        assert size in (8, 16)
        assert build.config["expert_options"] == resource.options
        assert resource.groups == {
            "exchange": setup.context.parallelism.group("all_ranks")
        }
        if size in resources:
            assert resources[size] == resource
        resources[size] = resource
        uses[size] = uses.get(size, 0) + 1
        assert name == f"experts.tokens-{size}"
    assert set(resources) == {8, 16} and all(count > 1 for count in uses.values())
    assert set(resources[8].options["layers"]).isdisjoint(
        resources[16].options["layers"]
    )
    assert not any("fp8" in name for name in build.allocation_invariants)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("grad_dtype", ["float32", "bfloat16"])
def test_model_construction_keeps_state_names_and_uses_matching_resource(
    family, grad_dtype, monkeypatch
):
    from mlops.expert_parallel.quack.registry import resource_handle

    from shadowspill.pytorch.representations import tensor_components

    setup, inventory = prepared(family, grad_dtype=grad_dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("meta construction requested a live GPU/group")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "get_world_size", forbidden)
    before, geometries, handles = torch.get_default_dtype(), [], {}
    for template in inventory.templates.values():
        build = template.model_build
        model = setup.model.create(build=build)
        try:
            assert torch.get_default_dtype() == before
            assert all(p.is_meta for p in model.parameters())
            assert model.embed.weight.dtype == torch.bfloat16
            assert model.rotary.cosine.dtype == torch.float32
            geometries.append(
                {n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()}
            )
            components = {
                (name, path): (parameter, leaf)
                for name, parameter in model.named_parameters()
                for path, leaf in tensor_components(parameter)
                if path
            }
            assert {
                (layout.parameter, layout.component_path)
                for layout in build.component_layouts
            } == components.keys()
            for layout in build.component_layouts:
                parameter, leaf = components[(layout.parameter, layout.component_path)]
                assert leaf.dtype == torch.bfloat16
                assert parameter.dtype == torch.float32
                assert tuple(leaf.shape) == tuple(
                    parameter.shape[a] for a in layout.axes
                )
                assert layout.encoding == "mlops/bf16-compute@1"
            options = build.config["expert_options"]
            size = options["config"]["tokens_per_rank"]
            handles[size] = tuple(block.moe.experts._handle for block in model.blocks)
            assert handles[size] == tuple(resource_handle(n) for n in options["layers"])
            for block in model.blocks:
                assert block.moe.experts.config.tokens_per_rank == size
                assert block.moe.experts.gate_up_weight.shape == (2, 256, 128)
                assert block.moe.experts.down_weight.shape == (2, 128, 128)
        finally:
            model.close()
    assert all(geometry == geometries[0] for geometry in geometries)
    assert set(handles[8]).isdisjoint(handles[16])


def fill(loader, name, shape, dtype, chunk):
    value = torch.empty(shape, dtype=dtype)
    flat = value.flatten()
    for start in range(0, flat.numel(), chunk):
        loader(
            TensorChunk(name, shape, dtype, start, flat.numel()),
            flat[start : start + chunk],
        )
    return value


@pytest.mark.parametrize("family", FAMILIES)
def test_initialization_preserves_replicas_expert_slices_and_payload_masters(
    tmp_path, family
):
    from shadowspill_workloads.initialization.decoder import Initialize as Catalog
    from shadowspill_workloads.initialization.expert_parallel import Initialize
    from shadowspill_workloads.models.catalog import entry

    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    requested = settings(family)
    resolved = registry.resolve_model(requested.definition, settings=requested)
    architecture = entry(family + "-tiny").config(**requested.config["architecture"])
    runtime = InitializationRuntime(
        tmp_path, lambda _: pytest.fail("unexpected artifact")
    )
    values = []
    for rank in range(4):
        context = InitializationContext(
            resolved.settings, resolved.settings.initialize, ownership(2, 2), rank
        )
        described = resolved.describe_initialization(context)
        assert described.chunk_bytes <= described.peak_bytes <= 33 << 20
        expected = Initialize(
            Catalog(family, architecture, seed=17),
            expert_rank=rank % 2,
            std=architecture.initializer_range,
            router_dtype=torch.float32 if family == "olmoe" else torch.bfloat16,
        )
        actual = {}
        with described.open(runtime) as loader:
            for name, shape in (
                ("embed.weight", (97, 128)),
                ("blocks.0.moe.experts.router_weight", (4, 128)),
                ("blocks.0.moe.experts.gate_up_weight", (2, 256, 128)),
                ("blocks.0.moe.experts.down_weight", (2, 128, 128)),
            ):
                # Vary chunk boundaries, including through one expert's rows.
                a = fill(loader, name, shape, torch.float32, 509)
                b = fill(expected, name, shape, torch.float32, 773)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                if name.endswith(("gate_up_weight", "down_weight")):
                    master = fill(loader, "master." + name, shape, torch.float32, 877)
                    payload = fill(loader, name + "._data", shape, torch.bfloat16, 1291)
                    torch.testing.assert_close(a, master, rtol=0, atol=0)
                    torch.testing.assert_close(a, payload.float(), rtol=0, atol=0)
                actual[name] = a
        values.append(actual)
    for name in values[0]:
        torch.testing.assert_close(values[0][name], values[2][name], rtol=0, atol=0)
        torch.testing.assert_close(values[1][name], values[3][name], rtol=0, atol=0)
        if name.endswith(("gate_up_weight", "down_weight")):
            assert not torch.equal(values[0][name], values[1][name])
        else:
            torch.testing.assert_close(values[0][name], values[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_unsupported_ep_modes_fail_before_construction(family):
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    requested = settings(family)
    for model in (
        replace(requested, precision={"compute_dtype": "float32"}),
        replace(requested, lora=LoRA(2)),
    ):
        with pytest.raises(ValueError, match="EP"):
            registry.resolve_model(model.definition, settings=model)
    resolved = registry.resolve_model(requested.definition, settings=requested)
    with pytest.raises(ValueError, match="ModelBuild"):
        resolved.create()


@pytest.mark.parametrize(
    "parallelism",
    [
        Parallelism.dp(ranks_per_replica=2),
        replace(ownership(2, 1), parameter_storage={}),
        replace(
            ownership(2, 1),
            parameter_placements={
                "blocks.*.moe.experts.*weight": Replicated("all_ranks")
            },
        ),
        replace(
            ownership(2, 1),
            parameter_storage={
                "blocks.*.moe.experts.*weight": ParameterStorage(group="all_ranks")
            },
        ),
    ],
)
def test_wrong_expert_ownership_fails_during_cpu_description(parallelism):
    with pytest.raises(ValueError, match=r"Quack EP|expert .*storage"):
        prepared("qwen3moe", parallelism=parallelism)


def test_ep_inventory_does_not_import_torch_providers_or_recipes():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if 'recipes' in name or '.implementations.' in name:
            raise AssertionError('CPU description imported '+name)
sys.meta_path.insert(0, Guard())
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import *
from shadowspill.parallelism import InputFieldPlacement, InputPlacement
from dataclasses import replace
from shadowspill_workloads.definitions.expert_parallel import ownership
registry = load_registry(['shadowspill', 'shadowspill-workloads'])
for family in ('qwen3moe', 'qwen35moe', 'olmoe'):
    setup = resolve_training(registry,
        model=Model('shadowspill/'+family+'@1', implementation='mlops_ep',
                    initialize=Weights.random(seed=17)),
        objective=Objective('shadowspill/cross_entropy@1'),
        parallelism=replace(ownership(2, 1), inputs=InputPlacement.independent(2,
            fields=[InputFieldPlacement(p) for p in (
                ('inputs', 'tokens'), ('loss_inputs', 'target_tokens'),
                ('loss_inputs', 'loss_mask'), ('loss_inputs', 'weights'))])),
        memory=Memory(1, 1), optimizer=Optimizer('mlops/adamw@1'),
        operations={'step': ['metrics'], 'forward_backward': ['metrics']},
        batch=BatchConfig(limits={'token_capacity': 32, 'max_sequence_length': 4,
                                  'max_sequences': 8}),
        prepare=Preparation(candidates=BuildCandidates(microbatch_sizes=[8, 16]),
                            planning=PlanningOptions(microbatch_counts=[1])))
    assert capture_inventory(prepare_layouts(setup)).templates
assert not any(n in sys.modules for n in
    ('torch', 'mlops', 'triton', 'moonep', 'quack', 'sonicmoe', 'fla', 'pynvml'))
print('portable')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "portable"
