"""Fresh adapters have reproducible seeds independent of their base weights."""

from dataclasses import replace

import pytest
import torch

from shadowspill.parallelism import Parallelism
from shadowspill.training.components import (
    InitializationContext,
    InitializationRuntime,
    load_registry,
)
from shadowspill_workloads.initialization.decoder import LORA_LEAVES, Initialize
from workload_tests.initialization.test_pretrained_initialization import fill
from workload_tests.initialization.test_pretrained_initialization import (
    settings as text_settings,
)
from workload_tests.initialization.test_pretrained_metadata import FAMILIES


def settings(family):
    return text_settings(family, lora=True)


@pytest.mark.parametrize("family", FAMILIES)
def test_random_base_seed_and_adapter_seed_are_independent(tmp_path, family):
    requested = settings(family)
    requested = replace(requested, config={**requested.config, "lora_seed": 45})
    registry = load_registry(["shadowspill-workloads"])
    resolved = registry.resolve_model(requested.name, settings=requested)
    model = resolved.create()
    initial = resolved.describe_initialization(
        InitializationContext(
            resolved.settings, resolved.settings.initialize, Parallelism(), 0
        )
    )
    runtime = InitializationRuntime(tmp_path, lambda _: pytest.fail("artifact"))
    base = Initialize(family, model.config, seed=requested.initialize.seed)
    adapters = Initialize(family, model.config, seed=45)
    changed = set()
    with initial.open(runtime) as loader:
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            expected, original = torch.empty_like(value), torch.empty_like(value)
            is_adapter = name.rsplit(".", 1)[-1] in LORA_LEAVES
            fill(adapters if is_adapter else base, name, expected, value.numel())
            fill(base, name, original, 113)
            fill(loader, name, value, 17)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            if not torch.equal(value, original):
                assert is_adapter
                changed.add(name)
    assert changed


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_adapter_seed_is_a_nonnegative_integer(family, seed):
    registry = load_registry(["shadowspill-workloads"])
    requested = settings(family)
    with pytest.raises((TypeError, ValueError), match="lora_seed"):
        registry.resolve_model(
            requested.name,
            settings=replace(requested, config={**requested.config, "lora_seed": seed}),
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_adapter_seed_without_adapters_is_rejected(family):
    registry = load_registry(["shadowspill-workloads"])
    requested = settings(family)
    with pytest.raises(ValueError, match="lora_seed requires LoRA"):
        registry.resolve_model(
            requested.name,
            settings=replace(
                requested, lora=None, config={**requested.config, "lora_seed": 45}
            ),
        )
