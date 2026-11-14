"""Search the policy/chunk product, then qualify the exact two-rank winner."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.benchmarks._distributed_policy_search import (
    GEOMETRIES,
    experiment,
    fill_tensor,
    initial_values,
    policy_pairs,
    reference_fixture,
)


def test_distributed_policy_factory_preserves_update_and_worker_topology():
    from benchmarks.quickstart.phases.candidates import build_options

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
            assert build_options(setup, candidate)["parallelism"].world_size == 2
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


def _chunk_sizes(value):
    sizes = set()
    if isinstance(value, Mapping):
        if "row_chunks" in value:
            sizes.add(value["row_chunks"]["chunk_size"])
        if {"domain", "rows", "chunk_size", "inputs", "outputs"} <= value.keys():
            sizes.add(value["chunk_size"])
        for child in value.values():
            sizes.update(_chunk_sizes(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            sizes.update(_chunk_sizes(child))
    return sizes


def _cold_winner(directory):
    from engine_tests.execution.test_distributed_regions import (
        _check_collective_traces,
    )

    # Locate without importing the worker, whose module body performs execution.
    worker = (
        Path(__file__).parents[2]
        / "packages/engine/tests/engine_tests/execution/_distributed_execute.py"
    )
    for phase in ("execute", "resume"):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "--start-method=spawn",
            "--no-python",
            sys.executable,
            str(worker),
            str(directory),
        ]
        if phase == "resume":
            command.append("resume")
        with (directory / (phase + ".log")).open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                start_new_session=True,
            )
            try:
                status = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                pytest.fail(f"cold selected Plan {phase} timed out")
        assert status == 0, (directory / (phase + ".log")).read_text()
        for invocation in range(3) if phase == "execute" else (1, 2):
            _check_collective_traces(directory, invocation)


@pytest.mark.cuda
def test_distributed_quickstart_searches_policies_chunks_and_checks_saved_winner(
    tmp_path,
):
    from benchmarks.quickstart import main, run
    from benchmarks.quickstart.phases.common import plan_identity
    from engine_tests.spill_pool import spill_pool
    from shadowspill import CommunicationInfo, Plan, Program, SweepResult

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct CUDA devices")
    output = tmp_path / "search"
    assert (
        run(
            experiment,
            rank_map={0: 1, 1: 0},
            search_budget_gib=[3],
            spill_gib=0.125,
            spill=spill_pool(128 << 20),
            steps=3,
            output_dir=output,
            external_headroom_gib=0.75,
            warmup_seconds=0.001,
            measure_seconds=0.002,
            plots=False,
        )
        == 0
    )
    inventory = json.loads((output / "programs/index.json").read_text())
    assert inventory["world_size"] == 2 and inventory["preparation_world_size"] == 1
    assert len(inventory["candidates"]) == 6
    structures = {}
    checkpoint_counts = set()
    for entry in inventory["candidates"]:
        assert "failure" not in entry
        program = Program.load(output / "programs" / entry["file"])
        ownership = program.bindings["parameter_ownership"]
        assert len(ownership) == 6
        assert any(
            set(item["names"]) == {"blocks.0.out", "blocks.1.out"} for item in ownership
        )
        name, chunks = entry["name"].split("-chunks")
        geometry = tuple(map(int, chunks.split("_")))
        assert set(geometry) <= _chunk_sizes(program.bindings["task_metadata"])
        checkpoint_counts.add(tuple(map(len, program.bindings["training_templates"])))
        first = program.bindings["training_templates"][0][0]
        variants = {row["variant"]: row for row in first if not row["accumulates"]}
        assert set(variants) == {"save", "recompute"}
        for variant, row in variants.items():
            key = row["backward"]["profile_key"]
            operators = program.bindings["task_metadata"][key]["operators"]
            if variant == "recompute":
                assert ("aten._softmax.default" not in operators) == (
                    name == "sequence_retained"
                )
            elif name == "sequence_only_save":
                assert "aten._softmax.default" not in operators
        structures[(name, geometry)] = tuple(
            program.bindings["manifests"][variants["save"][direction]["profile_key"]][
                "compatibility_digest"
            ]
            for direction in ("forward", "backward")
        )
    assert len(checkpoint_counts) == 1
    for geometry in GEOMETRIES:
        assert (
            structures[("inputs_only", geometry)]
            == structures[("sequence_retained", geometry)]
        )
    communication = CommunicationInfo.load(output / "communication.ssp")
    assert {r["operation"] for r in communication.measurements} >= {
        "all_gather",
        "reduce_scatter",
    }
    sweep = SweepResult.load(output / "planning/sweep.ssp")
    assert len(sweep.cases) == 12 and all(c.plan is not None for c in sweep.cases)
    winner = json.loads((output / "plans/index.json").read_text())[0]
    plan = Plan.load(output / "plans" / winner["file"])
    assert plan_identity(plan) == winner["plan_digest"]
    assert winner["predicted_seconds"] == min(
        c.plan.summary["duration_seconds"] for c in sweep.cases
    )
    result = json.loads((output / "execution/budget-0000/result.json").read_text())
    assert result["plan_digest"] == winner["plan_digest"]
    assert [r["logical_rank"] for r in result["per_rank"]] == [0, 1]
    fixture = reference_fixture()
    for actual, expected in zip(result["steps"], fixture["expected"], strict=True):
        assert actual["objective_loss"] == pytest.approx(
            sum(sum(v).item() for v in expected["outputs"]), rel=3e-5, abs=3e-6
        )
    cold = tmp_path / "cold-winner"
    cold.mkdir()
    shutil.copy2(output / "plans" / winner["file"], cold / "plan.ssp")
    torch.save(fixture, cold / "oracle.pt")
    digest = hashlib.sha256((cold / "plan.ssp").read_bytes()).hexdigest()
    _cold_winner(cold)
    assert hashlib.sha256((cold / "plan.ssp").read_bytes()).hexdigest() == digest
    artifacts = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("programs", "profiles", "profiled")
        for path in (output / name).glob("*.ssp")
    }
    assert len(artifacts) == 18
    shutil.copy2(output / "planning/sweep.ssp", output / "original-sweep.ssp")
    assert (
        main(
            [
                "--phase",
                "plan",
                "--output-dir",
                str(output),
                "--search-budget-gib",
                "3.25",
                "--no-plots",
            ]
        )
        == 0
    )
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected
        for path, expected in artifacts.items()
    )
    assert json.loads((output / "plans/index.json").read_text())[0]["budget"] == int(
        3.25 * 2**30
    )
