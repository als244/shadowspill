"""Offline search changes host-copy rates, retaining sharded collective costs."""

import json
from dataclasses import replace

import pytest

from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.search.report import (
    StepSearchGeometryBuild,
    StepSearchPoint,
    StepSearchReport,
)
from shadowspill.simulator import simulate
from shadowspill.step import StepDataOrdering
from tests.shadowspill.simulator.test_transfer_recipes import case, recipe
from tests.tools.test_pressurefit_matrix import collection
from tools.pressurefit_trace.replan import host_rates, replan
from tools.pressurefit_trace.search_capture import record_requests, write


@pytest.mark.parametrize("simultaneous", [False, True])
def test_copy_speed_changes_without_scaling_collective_or_memory(simultaneous):
    program, schedule, config = case(
        evicting=simultaneous,
        fetch_recipe=recipe(size=3000, resource="gather"),
        evict_recipe=recipe(size=3000, scatter=0, resource="scatter"),
    )
    objects = 2 if simultaneous else 1
    program = replace(
        program,
        alias_groups=tuple(replace(a, size_bytes=24000) for a in program.alias_groups),
        objects=tuple(replace(o, size_bytes=24000) for o in program.objects),
    )
    measured = TransferBandwidths(
        50_000_000_000,
        50_000_000_000,
        50_000_000_000,
        50_000_000_000,
        fetch_latency_ns=7,
        evict_latency_ns=9,
    )
    results = []
    for gbps in (50, 300):
        lanes = host_rates(gbps, measured)
        assert lanes.fetch_latency_ns == 7 and lanes.evict_latency_ns == 9
        device = replace(
            config.devices[0],
            capacity_bytes=24000 * objects,
            fetch_solo_bandwidth_bytes_per_second=lanes.fetch_solo_bytes_per_second,
            fetch_concurrent_bandwidth_bytes_per_second=lanes.fetch_concurrent_bytes_per_second,
            evict_solo_bandwidth_bytes_per_second=lanes.evict_solo_bytes_per_second,
            evict_concurrent_bandwidth_bytes_per_second=lanes.evict_concurrent_bytes_per_second,
            fetch_latency_ns=7,
            evict_latency_ns=9,
        )
        results.append(
            simulate(
                program,
                schedule,
                config=replace(
                    config, devices=(device,), spill_capacity_bytes=3000 * objects
                ),
            )
        )
    # 3000-byte shard copies take 60/10 ns. Gather remains 60 ns, compute 10 ns,
    # in the logical bandwidth-only simulation. Physical admission keeps the
    # measured launch latencies carried by the override above.
    assert [r.makespan_ns for r in results] == [130, 80]
    assert [r.spill_peak_bytes for r in results] == [3000 * objects] * 2
    assert [r.device_peaks[0].object_bytes for r in results] == [24000 * objects] * 2
    if simultaneous:
        assert [
            next(
                t.end_ns for t in r.transfer_intervals if t.alias_group_id == "gradient"
            )
            for r in results
        ] == [60, 10]


@pytest.mark.parametrize("jobs", [1, 2])
def test_cpu_replan_covers_search_and_preserves_uncaptured_geometry(tmp_path, jobs):
    planning = collection(tmp_path)
    plan = AnnotatedProgramPlan.from_json((planning / "program.plan.json").read_text())
    problem = replace(
        plan.program,
        maximum_execution_budget_bytes=plan.memory_budgets.execution_bytes + 64,
    )
    budget = (plan.memory_budgets.execution_bytes, plan.memory_budgets.spill_bytes)
    budgets = (budget, (budget[0] + 64, budget[1]))
    ordering = StepDataOrdering.depth_first(1)
    source, output = tmp_path / "baseline", tmp_path / "300GBps"
    inputs = source / "search-inputs"
    source.mkdir()
    record_requests(
        inputs,
        (problem,),
        {
            (0, pair): (
                problem,
                plan.transfer_bandwidths,
                object(),
                {"verified": True},
            )
            for pair in budgets
        },
        incumbents=True,
        options=plan.result.search_options.to_dict(),
    )
    write(
        inputs / "geometries" / "example.json", {"local_problem_digest": problem.digest}
    )
    write(inputs / "metadata.json", {"complete": True, "global_tokens_per_step": 100})
    baseline = StepSearchReport(
        budgets,
        tuple(
            StepSearchGeometryBuild(name, 1, ordering, "example", 0)
            for name in ("small", "same_cost")
        ),
        (
            *(
                StepSearchPoint(name, 1, ordering, *pair, "succeeded", 1, None, None, 0)
                for name in ("small", "same_cost")
                for pair in budgets
            ),
            StepSearchPoint(
                "too_big",
                1,
                ordering,
                *budget,
                "infeasible",
                None,
                None,
                "capture OOM",
                0,
            ),
        ),
        search_options=plan.result.search_options,
    )
    baseline.save(source / "search.json")
    result = replan(source, output, gbps=300, workers=1, jobs=jobs)
    assert len(result.points) == 5
    assert result.points[-1].error == "capture OOM"
    assert result.winners[0].candidate == "small"
    saved = AnnotatedProgramPlan.from_json(
        (output / "winners" / f"{budget[0]}-{budget[1]}.plan.json").read_text()
    )
    assert saved.result.program == problem.program
    assert saved.transfer_bandwidths.fetch_solo_bytes_per_second == 300_000_000_000
    assert saved.memory_budgets == plan.memory_budgets
    assert len(saved.result.diagnostics.resolved_programs) == 2
    assert json.loads((output / "complete.json").read_text())["executed_steps"] == 0
    # Resume from the separate plan store and reject a changed experiment.
    resumed = replan(source, output, gbps=300, workers=1, jobs=jobs)
    assert resumed.winners[0].makespan_seconds == result.winners[0].makespan_seconds
    if jobs > 1:
        serial = replan(source, tmp_path / "serial", gbps=300, workers=1)
        assert [
            (p.candidate, p.makespan_seconds, p.incumbent_budget_bytes)
            for p in result.points
        ] == [
            (p.candidate, p.makespan_seconds, p.incumbent_budget_bytes)
            for p in serial.points
        ]
        for path in (output / "winners").glob("*.json"):
            a = AnnotatedProgramPlan.from_json(path.read_text())
            b = AnnotatedProgramPlan.from_json(
                (tmp_path / "serial" / "winners" / path.name).read_text()
            )
            assert (
                a.program.digest,
                a.result.schedule.digest,
                a.fixed_layout.digest,
            ) == (b.program.digest, b.result.schedule.digest, b.fixed_layout.digest)
    with pytest.raises(ValueError, match="different replan request"):
        replan(source, output, gbps=200, workers=1, jobs=jobs)


def test_missing_symmetry_is_not_silently_treated_as_single_rank(tmp_path):
    with pytest.raises(ValueError, match="symmetry was not verified"):
        record_requests(tmp_path, (), None, incumbents=True, options=None)


def test_input_export_deduplicates_equal_copies_but_preserves_changed_profiles(
    tmp_path,
):
    from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
    from tests.shadowspill.planner.test_annotated_plan import _pressurefit_program

    original = _pressurefit_program()
    equal = replace(original, program=replace(original.program))
    changed = replace(
        original,
        program=replace(
            original.program,
            profiles=(replace(original.program.profiles[0], runtime_ns=77),),
        ),
    )
    problems = [original, equal, changed]
    budgets = [(value, 1024) for value in (224, 256, 384)]
    record_requests(
        tmp_path,
        (original,),
        {
            (0, budget): (problem, original.transfer_bandwidths, object(), {})
            for budget, problem in zip(budgets, problems, strict=True)
        },
        incumbents=True,
        options=None,
    )
    assert len(list((tmp_path / "problems").glob("*.json"))) == 2
    for budget, problem in zip(budgets, problems, strict=True):
        request = json.loads(
            (
                tmp_path
                / "requests"
                / original.digest
                / f"{budget[0]}-{budget[1]}.json"
            ).read_text()
        )
        restored = ShadowSpillPlanningProblem.from_json(
            (tmp_path / request["problem"]).read_text()
        )
        assert restored == problem
        assert request["problem_digest"] == problem.digest
