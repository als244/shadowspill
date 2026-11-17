"""Search the policy/chunk product, then qualify the exact two-rank winner."""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking._policy_search import (
    GEOMETRIES,
    experiment,
    fill_tensor,
    initial_values,
    policy_pairs,
    reference_fixture,
)


def test_distributed_policy_factory_preserves_update_and_worker_topology():
    setups = [experiment(device="cpu", logical_rank=rank) for rank in range(2)]
    assert setups[0]["metadata"] == setups[1]["metadata"]
    assert len(setups[0]["candidates"]) == 6
    masks = []
    for setup in setups:
        assert setup["build_options"]["parallelism"].world_size == 2
        first = next(iter(setup["candidates"].values()))
        for candidate in setup["candidates"].values():
            torch.testing.assert_close(candidate["args"], first["args"], atol=0, rtol=0)
            torch.testing.assert_close(
                candidate["kwargs"], first["kwargs"], atol=0, rtol=0
            )
            assert "parallelism" not in candidate["build_options"]
            options = {**setup["build_options"], **candidate["build_options"]}
            assert options["parallelism"].world_size == 2
        masks.extend(item["sample_weights"] for item in first["kwargs"])
    assert len({int(mask.count_nonzero()) for mask in masks}) == 4
    assert sum(mask.double().sum().item() * 7 for mask in masks) == pytest.approx(1)
    first, second = [next(iter(s["candidates"].values())) for s in setups]
    assert not torch.equal(first["args"][0][0], second["args"][0][0])
    model = setups[0]["model_factory"]()
    assert all(p.device.type == "meta" for p in model.parameters())
    assert model.blocks[0].out is model.blocks[1].out
    for name, parameter in model.named_parameters():
        actual = torch.empty(parameter.numel())
        for offset in range(0, actual.numel(), 13):
            fill_tensor(
                SimpleNamespace(name=name, offset=offset), actual[offset : offset + 13]
            )
        torch.testing.assert_close(
            actual, initial_values(name, 0, actual.numel()), atol=0, rtol=0
        )


def test_distributed_search_reference_global_normalization_and_updates():
    fixture = reference_fixture()
    assert fixture["reference_dtype"] == "float64"
    assert len(fixture["expected"]) == 3
    for name, value in fixture["initial"].items():
        assert value.dtype == torch.float32 and value.count_nonzero() > 0
        assert not torch.equal(value, fixture["expected"][-1]["state"][name])
    for index, expected in enumerate(fixture["expected"]):
        for name, value in expected["state"].items():
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
            if name.endswith(".step"):
                assert value.item() == index + 1


@pytest.mark.parametrize("name", tuple(policy_pairs()))
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("variant", ["save", "recompute"])
def test_search_candidate_regions_preserve_arbitrary_vjp(name, geometry, variant):
    from engine_tests.execution._distributed_regions import (
        Block,
        check_lifetime_reports,
    )
    from engine_tests.pytorch.chunking.test_composition import executable
    from shadowspill.pytorch.capture.aot import capture_graph_pair
    from shadowspill.pytorch.chunking.regions import region_artifact

    torch.manual_seed(731)
    model = Block(False, dtype=torch.float64, lifetime_mode="replay_compact")
    weights = dict(model.named_parameters())
    inputs = (
        torch.randn(2, 17, 7, dtype=torch.float64, requires_grad=True),
        *weights.values(),
    )

    def forward(x, *values):
        return torch.func.functional_call(
            model, dict(zip(weights, values, strict=True)), (x,)
        )

    output = forward(*inputs)
    tangent = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs, tangent)
    label = name + "-chunks" + "_".join(map(str, geometry))
    controls = experiment(device="cpu")["candidates"][label]["build_options"]
    pair = capture_graph_pair(
        make_fx(forward)(*inputs),
        inputs,
        original_output=output,
        memory_budget=float(variant == "save"),
        retention=controls["graph_pair_policy"].variant(variant),
    )
    fw = region_artifact(
        pair.forward, controls["chunking"], stage_id="block", variant=variant
    )
    count = pair.saved_value_count
    domains = (*fw.regions.output_domains[-count:], fw.regions.output_domains[0])
    bw = region_artifact(
        pair.backward,
        controls["chunking"],
        stage_id="block",
        variant=variant,
        input_domains=domains,
    )
    assert fw.storage_contract == pair.forward.storage_contract
    assert bw.storage_contract == pair.backward.storage_contract
    reports = [a.regions.lifetime_report for a in (fw, bw)]
    if variant == "recompute":
        check_lifetime_reports(reports, "replay_compact")
    else:
        # SAVE can protect the original backing as required checkpoint state,
        # leaving no profitable compaction. Its boundary and VJP still must hold.
        assert all(
            after <= before
            for report in reports
            for name in ("schedule", "compaction", "replay")
            for _, before, after in (report.get(name) or {}).get("barrier_bytes", ())
        )
    selector = controls["chunking"].lifetimes.select
    selector.building = False
    with torch.no_grad():
        actual = executable(fw)(*inputs)
        gradients = executable(bw)(*actual[-count:], tangent)
    torch.testing.assert_close(actual[0], output, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        tuple(g for g in gradients if g is not None), expected, atol=1e-10, rtol=1e-10
    )
