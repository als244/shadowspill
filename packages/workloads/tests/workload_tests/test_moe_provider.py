"""Installed MoE builds retain provider preparation and exact source dependencies."""

import pytest

from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_workloads.definitions import moe_provider, source
from workload_tests.test_moe_definitions import FAMILIES, prepared


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_prepared_moe_family_retains_provider_lifecycle_and_allocation_scope(
    family, dtype, monkeypatch
):
    import torch
    from mlops.operator_metadata import operator_preparations

    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *a, **kw: pytest.fail("device discovery during CPU description"),
    )
    _, _, layouts = prepared(family, dtype=dtype, lora=True)
    templates = capture_inventory(layouts).templates
    available = {
        item.factory: dict(item.options)
        for item in operator_preparations(dtype=getattr(torch, dtype))
    }
    for template in templates.values():
        build = template.model_build
        assert len(build.operator_preparation) == (dtype == "bfloat16")
        for item in build.operator_preparation:
            assert dict(item.options) == available[item.factory]
        sonic = {
            name: contract
            for name, contract in build.allocation_invariants.items()
            if "sonicmoe" in name
        }
        if dtype == "bfloat16":
            assert sonic == moe_provider.SONIC_CONTRACTS
        else:
            assert not sonic
        assert not any("accumulate" in name for name in sonic)


@pytest.mark.parametrize("provider", moe_provider.PROVIDERS)
def test_provider_changes_invalidate_each_supplied_moe_definition(
    provider, monkeypatch
):
    changed = False

    def digest(name):
        return (name + ("changed" if changed and name == provider else "")).encode()

    monkeypatch.setattr(source, "_provider_digest", digest)

    def identities():
        registry = load_registry(["shadowspill-workloads"])
        return {
            family: registry.resolve_model(f"shadowspill/{family}@1").source
            for family in FAMILIES
        }

    before = identities()
    changed = True
    after = identities()
    assert all(before[family] != after[family] for family in FAMILIES)
