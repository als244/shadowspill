"""GLM uses public setup, native checkpoint storage and bounded initialization."""

import importlib
import json
import subprocess
import sys

import pytest
import torch

from shadowspill.parallelism import Parallelism
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
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    Weights,
)
from shadowspill_workloads.models.shared.glm.checkpoint import physical_state
from workload_tests.test_glm53_checkpoint import _materialized, _write
from workload_tests.test_glm53_initialization import (
    mixed_checkpoint as mixed_checkpoint,
)
from workload_tests.test_pretrained_initialization import fill


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
def test_public_random_setup_prepares_runtime_packing(family):
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=Model(
            family,
            config={"preset": "tiny"},
            lora=LoRA(4),
            initialize=Weights.random(seed=17),
        ),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer("adamw"),
        batch=BatchConfig(
            limits={"token_capacity": 64, "max_sequence_length": 32, "max_sequences": 8}
        ),
        memory=Memory(1, 1),
    )
    inventory = capture_inventory(prepare_layouts(setup))
    assert inventory.templates
    assert resolve_training(registry, setup.config).config == setup.config
    with torch.device("meta"):
        model = setup.model.create()
    assert any(p.requires_grad for p in model.parameters())
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)


def mark_family(directory, family):
    path = directory / "config.json"
    metadata = json.loads(path.read_text())
    text = metadata.get("text_config", metadata)
    text["model_type"] = "glm5_next_text" if family == "glm53_flash" else "glm_moe_dsa"
    path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("family", ["glm53", "glm53_flash"])
@pytest.mark.parametrize("fp8", [False, True])
def test_public_checkpoint_matches_existing_reader_without_base_conversion(
    tmp_path, family, fp8
):
    module = importlib.import_module(
        f"shadowspill_workloads.models.{family}.checkpoint"
    )
    config = importlib.import_module(
        f"shadowspill_workloads.models.{family}.config"
    ).Config.tiny(hidden_size=128)
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
    check_checkpoint(tmp_path, family)


def check_checkpoint(directory, family):
    registry = load_registry(["shadowspill-workloads"])
    requested = Model.from_pretrained(
        str(directory), lora=LoRA(4), precision={"lora_dtype": "bfloat16"}
    )
    pinned, _, _ = registry.resolve_model_reference(requested)
    resolved = registry.resolve_model(pinned.definition, settings=pinned)
    with torch.device("meta"):
        model = resolved.create()
    module = importlib.import_module(
        f"shadowspill_workloads.models.{family}.checkpoint"
    )
    with module.GLMCheckpoint(directory) as reader:
        reference = reader.build_model(
            lora_rank=4, lora_alpha=4, lora_factor_dtype=torch.bfloat16
        )

        def geometry(value):
            return {
                n: (tuple(t.shape), t.dtype, t.requires_grad)
                for n, t in physical_state(value).items()
            }

        assert geometry(model) == geometry(reference)
        reference = _materialized(reference)
        reader.load_into(reference)
    initial = resolved.describe_initialization(
        InitializationContext(
            resolved.settings, resolved.settings.initialize, Parallelism(), 0
        )
    )
    cache = directory / "private"
    cache.mkdir()
    runtime = InitializationRuntime(
        cache, lambda _: pytest.fail("artifact request"), initial.disk_bytes
    )
    model = _materialized(model)
    with initial.open(runtime) as loader:
        for name, tensor in physical_state(model).items():
            fill(loader, name, tensor, 8191)
            if "lora_" not in name:
                expected = physical_state(reference)[name]
                assert torch.equal(
                    tensor.view(torch.uint8), expected.view(torch.uint8)
                ), name
            elif name.endswith("_b"):
                assert torch.count_nonzero(tensor) == 0
            else:
                assert torch.isfinite(tensor).all() and torch.count_nonzero(tensor)
    assert not list(cache.iterdir())
    return pinned


def test_public_mixed_projections_and_description_are_framework_free(mixed_checkpoint):
    mark_family(mixed_checkpoint, "glm53_flash")
    pinned = check_checkpoint(mixed_checkpoint, "glm53_flash")
    assert pinned.config["architecture"]["expert_storage"] == "fp8_block"
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton', 'mlops'):
        raise AssertionError('description imported runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from shadowspill.training.definitions import load_registry
from shadowspill_client import Model, LoRA
registry = load_registry(['shadowspill-workloads'])
requested = Model.from_pretrained(sys.argv[1], lora=LoRA(4))
pinned, _, _ = registry.resolve_model_reference(requested)
resolved = registry.resolve_model(pinned.definition, settings=pinned)
assert resolved.config['architecture']['expert_storage'] == 'fp8_block'
assert not any(x in sys.modules for x in ('torch', 'triton', 'mlops'))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(mixed_checkpoint)], check=True, timeout=60
    )
