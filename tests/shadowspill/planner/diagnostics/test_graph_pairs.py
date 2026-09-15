"""Graph-pair outcomes and the cost table they and the plan summary share."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.planner.diagnostics import (
    AlternativeCosts,
    GraphPairOutcome,
    graph_pair_outcomes,
)

from .._examples import recomputation_program, representative_result


def test_the_cost_table_reads_every_option_and_the_floor_off_the_program() -> None:
    program = recomputation_program(recompute_workspace_bytes=1)
    dearer = replace(
        program,
        profiles=tuple(
            replace(item, runtime_ns=150)
            if item.profile_id == "recompute_profile"
            else item
            for item in program.profiles
        ),
    )
    costs = AlternativeCosts.from_program(dearer)
    # middle and consume are paid under every selection
    assert costs.fixed_ns == 1_000 + 100
    assert dict(costs.option_ns) == {
        ("activation_tradeoff", "save"): 100,
        ("activation_tradeoff", "recompute"): 150,
    }
    assert dict(costs.cheapest_ns) == {"activation_tradeoff": 100}
    assert costs.floor_ns == 1_200
    assert costs.selected_ns({"activation_tradeoff": "recompute"}) == 1_250
    assert costs.recomputing({"activation_tradeoff": "recompute"}) == 1
    assert costs.recomputing({"activation_tradeoff": "save"}) == 0


def test_one_outcome_per_resolved_program_derived_without_simulating() -> None:
    result = representative_result()
    (outcome,) = graph_pair_outcomes(result)
    assert outcome.selection_id == "fixture"
    assert (outcome.group_count, outcome.recompute_groups) == (1, 0)
    assert outcome.makespan_seconds == result.simulation.makespan_ns / 1e9
    # save and recompute share one profile in the representative program
    assert outcome.selected_compute_seconds == outcome.unconstrained_seconds
    assert outcome.recomputation_overhead_seconds == 0.0
    assert outcome.waiting_seconds == (
        outcome.makespan_seconds - outcome.selected_compute_seconds
    )
    assert (outcome.valid_candidate_count, outcome.candidate_count) == (1, 1)
    assert GraphPairOutcome.from_dict(outcome.as_dict(), "outcome") == outcome
