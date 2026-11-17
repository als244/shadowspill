"""Public GLM EP description and rank-local native checkpoint initialization."""

from dataclasses import replace
from importlib import import_module

import pytest
import torch

from shadowspill import Parallelism
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
    Preparation,
    Weights,
)
from shadowspill_client.parallelism import expert_parallel
from shadowspill_workloads.models.shared.glm.checkpoint import physical_state
from workload_tests.test_glm53_checkpoint import _write
from workload_tests.test_glm_definitions import mark_family
from workload_tests.test_pretrained_initialization import fill


def prepared(model, *, grad_dtype="float32"):
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=model,
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw", precision={"gradient_dtype": grad_dtype}),
        memory=Memory(1, 1),
        batch=BatchConfig(
            limits={
                "token_capacity": 128,
                "max_sequence_length": 32,
                "max_sequences": 8,
            }
        ),
        parallelism=Parallelism.from_dict(
            expert_parallel(2, experts=["layers.*.mlp.experts.*"])
        ),
        prepare=Preparation(candidates=BuildCandidates(microbatch_sizes=[32, 64])),
    )
    return setup, capture_inventory(prepare_layouts(setup))


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
@pytest.mark.parametrize("storage", ["bf16", "fp8_block"])
@pytest.mark.parametrize("grad_dtype", ["float32", "bfloat16"])
def test_ep_geometry_resources_and_component_ownership_without_live_runtime(
    family, storage, grad_dtype, monkeypatch
):
    from mlops.expert_parallel.lora import parse_signature

    from shadowspill.pytorch.representations import tensor_components

    setup, inventory = prepared(
        Model(
            family,
            implementation="mlops_ep",
            config={
                "preset": "tiny",
                "architecture": {"hidden_size": 128, "expert_storage": storage},
            },
            lora=LoRA(16),
            initialize=Weights.random(seed=17),
        ),
        grad_dtype=grad_dtype,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("description opened a live device or communicator")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "get_world_size", forbidden)
    sizes, handles = set(), {}
    for template in inventory.templates.values():
        build = template.model_build
        assert len(build.resources) == 1
        resource = next(iter(build.resources.values()))
        size = resource.options["config"]["tokens_per_rank"]
        sizes.add(size)
        with torch.device("meta"):
            model = setup.model.create(build=build)
        assert all(p.is_meta for p in model.parameters())
        components = {
            (name, path): (parameter, leaf)
            for name, parameter in model.named_parameters()
            for path, leaf in tensor_components(parameter)
            if path
        }
        assert {
            (x.parameter, x.component_path) for x in build.component_layouts
        } == components.keys()
        for layout in build.component_layouts:
            parameter, leaf = components[(layout.parameter, layout.component_path)]
            assert leaf.shape == tuple(
                (parameter.shape[a] + b - 1) // b
                for a, b in zip(layout.axes, layout.blocks, strict=True)
            )
        current = []
        for layer in model.layers:
            if not hasattr(layer.mlp, "experts"):
                continue
            experts = layer.mlp.experts
            assert experts is not None and experts.router_weight is None
            assert not hasattr(layer.mlp, "w13")
            assert experts.gate_up_weight.shape == (2, 256, 128)
            assert experts.config.swiglu_limit == model.config.swiglu_limit
            _, lora = parse_signature(experts.resource_spec)
            assert lora.rank == 16 and lora.gradient_dtype == getattr(torch, grad_dtype)
            current.append(experts._handle)
        handles[size] = set(current)
        assert all(
            "lora_" in name for name, p in model.named_parameters() if p.requires_grad
        )
    assert sizes == {32, 64}
    assert handles[32].isdisjoint(handles[64])


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
@pytest.mark.parametrize("fp8", [False, True])
def test_native_checkpoint_expert_slices_and_lora_seeds_match_global_loader(
    tmp_path, family, fp8
):
    module = import_module(f"shadowspill_workloads.models.{family}.checkpoint")
    config = import_module(f"shadowspill_workloads.models.{family}.config").Config.tiny(
        hidden_size=128
    )
    _write(
        tmp_path,
        fp8_experts=fp8,
        config=config,
        model_type=module.Model,
        name_for=module.source_name,
        prefix=module.GLMCheckpoint.source_prefix.removesuffix("."),
        nested=family == "glm53_flash",
    )
    mark_family(tmp_path, family)
    setup, inventory = prepared(
        Model.from_pretrained(str(tmp_path), implementation="mlops_ep", lora=LoRA(16))
    )
    registry = load_registry(["shadowspill-workloads"])
    dense_settings = replace(
        setup.config.model,
        implementation="mlops",
        config={**setup.config.model.config, "implementation": "mlops"},
    )
    dense = registry.resolve_model(dense_settings.definition, settings=dense_settings)
    with torch.device("meta"):
        global_model = dense.create()
        local_model = setup.model.create(
            build=next(iter(inventory.templates.values())).model_build
        )
    global_values = {}
    description = dense.describe_initialization(
        InitializationContext(
            dense.settings, dense.settings.initialize, setup.context.parallelism, 0
        )
    )
    root = tmp_path / "private"
    root.mkdir()
    runtime = InitializationRuntime(
        root, lambda _: pytest.fail("artifact request"), description.disk_bytes
    )
    with description.open(runtime) as loader:
        for name, tensor in physical_state(global_model).items():
            value = torch.empty(tensor.shape, dtype=tensor.dtype)
            fill(loader, name, value, 4093)
            global_values[name] = value
    for rank in range(2):
        description = setup.model.describe_initialization(
            InitializationContext(
                setup.config.model,
                setup.config.model.initialize,
                setup.context.parallelism,
                rank,
            )
        )
        with description.open(runtime) as loader:
            for name, tensor in physical_state(local_model).items():
                value = torch.empty(tensor.shape, dtype=tensor.dtype)
                fill(loader, name, value, 8191)
                mapped = (
                    name.replace(".mlp.experts.", ".mlp.")
                    .replace(".gate_up_weight", ".w13")
                    .replace(".down_weight", ".w2")
                )
                if mapped not in global_values:
                    mapped = mapped.removesuffix("._data")
                expected = global_values[mapped]
                if ".mlp.experts." in name:
                    expected = expected.chunk(2)[rank]
                assert torch.equal(
                    value.view(torch.uint8), expected.view(torch.uint8)
                ), (rank, name)
    assert not list(root.iterdir())
