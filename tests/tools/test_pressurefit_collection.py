"""Counterfactual bandwidths survive collection, recording, and replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from shadowspill.diagnostics.pressurefit_trace import read_manifest
from shadowspill.ir import MutationSpec
from shadowspill.planner import plan_program
from tests.shadowspill.planner.test_annotated_plan import _pressurefit_program
from tools.pressurefit_trace.__main__ import main
from tools.pressurefit_trace.collect.budgets import transfer_bandwidths


@pytest.mark.parametrize(
    "value", ["0,450", "-1,450", "nan,450", "inf,450", "450", "a,b"]
)
def test_bandwidth_override_rejects_invalid_rates(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive finite"):
        transfer_bandwidths(value)


def test_bandwidth_override_distinguishes_independent_and_shared_lanes():
    independent = transfer_bandwidths("450,450")
    shared = transfer_bandwidths("450,225,450,225")
    assert independent.fetch_solo_bytes_per_second == 450_000_000_000
    assert independent.fetch_concurrent_bytes_per_second == 450_000_000_000
    assert independent.evict_concurrent_bytes_per_second == 450_000_000_000
    assert shared.fetch_concurrent_bytes_per_second == 225_000_000_000
    assert shared.evict_concurrent_bytes_per_second == 225_000_000_000
    assert independent.fetch_latency_ns is None
    assert independent.evict_latency_ns is None


def test_recorded_winner_uses_the_saved_bandwidth_override(tmp_path):
    source = _pressurefit_program()
    source = replace(
        source,
        program=replace(
            source.program,
            tasks=(
                replace(
                    source.program.tasks[0],
                    mutations=(MutationSpec("state_object"),),
                ),
            ),
        ),
    )
    original = plan_program(source, execution_budget=512, verbose=False)
    capture = tmp_path / "dump"
    capture.mkdir()
    (capture / "program.problem.json").write_text(source.to_json())
    (capture / "program.semantics.json").write_text("{}")
    (capture / "dump.json").write_text(
        json.dumps(
            {
                "execution_budget_bytes": 512,
                "physical_budget_bytes": 512,
                "spill_budget_bytes": 1024,
                "programs": [
                    {
                        "name": "program",
                        "problem": "program.problem.json",
                        "semantics": "program.semantics.json",
                    }
                ],
            }
        )
    )
    planning, recording = tmp_path / "plans", tmp_path / "traces"
    main(
        [
            "plan",
            str(capture),
            "--output",
            str(planning),
            "--transfer-bandwidths",
            "450,450",
            "--workers",
            "1",
        ]
    )
    plans = json.loads((planning / "plans.json").read_text())
    assert plans["programs"][0]["makespan_ns"] == 11
    assert original.result.simulation.makespan_ns == 32_010
    assert (
        plans["transfer_bandwidths_override"]["evict_solo_bytes_per_second"]
        == 450_000_000_000
    )
    # A rejected geometry has no saved plan. "all" still traces every success.
    plans["programs"].append({"name": "rejected", "error": "does not fit"})
    (planning / "plans.json").write_text(json.dumps(plans))
    main(["trace", str(planning), "--output", str(recording), "--programs", "all"])
    collection = json.loads((recording / "collection.json").read_text())
    trace = Path(collection["directories"][0])
    manifest = read_manifest(trace)
    assert manifest["answer"]["makespan_ns"] == 11
    assert (
        manifest["simulation_config"]["devices"][0][
            "evict_concurrent_bandwidth_bytes_per_second"
        ]
        == 450_000_000_000
    )
    context = json.loads((trace / "collection_context.json").read_text())
    assert (
        context["transfer_bandwidths"]["fetch_concurrent_bytes_per_second"]
        == 450_000_000_000
    )
