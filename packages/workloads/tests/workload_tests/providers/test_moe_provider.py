"""Installed MoE builds retain provider preparation and exact source dependencies."""

import pytest

from shadowspill.artifacts.program import Program
from shadowspill.ir import DeviceSpec
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.capture.allocation_coverage import (
    allocation_coverage,
)
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill_workloads.definitions import moe_provider, source
from workload_tests.definitions.test_moe_definitions import FAMILIES, prepared


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
        # Exercise the actual bound-task check before reserving a GPU. This
        # description-only Program isolates preparation activation from kernels.
        program = Program(
            devices=(DeviceSpec("rank0", "worker", "cuda", 0),),
            alias_groups=(),
            objects=(),
            tasks=(),
            task_definitions=(),
            bindings={
                "operator_preparation": {
                    "providers": [item.to_dict() for item in build.operator_preparation]
                }
            },
        )
        proof = allocation_coverage(template, program)
        assert proof["preparations"] == {
            item.factory: build.allocation_invariants[item.factory]
            for item in build.operator_preparation
        }
        sonic = {
            name: contract
            for name, contract in build.allocation_invariants.items()
            if "sonicmoe" in name
        }
        if dtype == "bfloat16":
            assert sonic == {
                **moe_provider.SONIC_CONTRACTS,
                **moe_provider.DESTINATION_CONTRACTS,
            }
        else:
            assert not sonic


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "dtype,lora", [("bfloat16", True), ("float32", True), ("bfloat16", False)]
)
def test_destination_rules_reuse_provider_eligibility_and_exact_allocation_contracts(
    family, dtype, lora, monkeypatch
):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *a, **kw: pytest.fail("device discovery while resolving contracts"),
    )
    _, setup, layouts = prepared(family, dtype=dtype, lora=lora)
    template = next(iter(capture_inventory(layouts).templates.values()))
    rules = setup.model.gradient_destination_rules(template.model_build)
    if dtype != "bfloat16" or not lora:
        assert not rules
        return
    assert {"torch.ops." + item.replacement for item in rules} == set(
        moe_provider.DESTINATION_CONTRACTS
    )
    for item in rules:
        assert item.version == 1 and item.output_indices == (1, 2, 3)
        assert item.mapper is not None
        assert "torch.ops." + item.target in moe_provider.SONIC_CONTRACTS
        name = "torch.ops." + item.replacement
        assert (
            template.model_build.allocation_invariants[name]
            == (moe_provider.DESTINATION_CONTRACTS[name])
        )


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
