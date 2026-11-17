"""Require measured parameter-gradient spill before another backward mutation."""

import json

from shadowspill import MemoryBudgets, plan_program
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search import GenericPlanningOptions, SearchOptions


def gradient_spill_evidence(plan):
    """Exclude activation gradients, terminal exports and initial gradient fetches."""
    paths = []
    for mode, selected in (("fresh", plan), ("accumulate", plan._accumulation_plan)):
        if selected is None:
            continue
        objects = {obj.object_id: obj for obj in selected.program.objects}
        names = {
            objects[item["gradient_object_id"]].alias_group_id: item["parameter_name"]
            for item in selected.program.bindings.get("gradients", ())
        }
        ordinals = {
            item.task_id: index
            for index, item in enumerate(selected.timeline.task_intervals)
        }
        writes = {alias: [] for alias in names}
        mutations = {alias: [] for alias in names}
        for task in selected.program.tasks:
            ordinal = ordinals[task.task_id]
            for object_id in task.outputs:
                alias = objects[object_id].alias_group_id
                if alias in names:
                    writes[alias].append(ordinal)
            for mutation in task.mutations:
                alias = objects[mutation.object_id].alias_group_id
                if alias in names:
                    writes[alias].append(ordinal)
                    if task.phase == "backward":
                        mutations[alias].append(ordinal)
        actions = [
            dict(
                alias=action.alias_group_id,
                parameter=names[action.alias_group_id],
                kind=action.kind.value,
                ordinal=ordinals[action.trigger_task_id],
            )
            for action in selected.schedule.actions
            if action.alias_group_id in names
        ]
        roundtrips = []
        for evict in actions:
            if evict["kind"] != "evict":
                continue
            for fetch in actions:
                if (
                    fetch["kind"] != "fetch"
                    or fetch["alias"] != evict["alias"]
                    or fetch["ordinal"] <= evict["ordinal"]
                ):
                    continue
                # Actions follow their trigger task. A later mutation must be
                # strictly after the fetch; an eviction may follow its producer.
                prior = [i for i in writes[evict["alias"]] if i <= evict["ordinal"]]
                later = [i for i in mutations[evict["alias"]] if i > fetch["ordinal"]]
                if prior and later:
                    roundtrips.append(
                        dict(
                            evict=evict,
                            fetch=fetch,
                            prior_write=max(prior),
                            next_backward_mutation=min(later),
                        )
                    )
        paths.append(
            dict(
                mode=mode,
                roundtrips=roundtrips,
                layout_bytes=selected.memory_layout.required_bytes,
            )
        )
    return paths


def constrain_gradient_memory(profiled, communication, baseline, directory):
    """Use measured physical admission, retaining every refusal and exact winner."""
    baseline.save(directory / "unconstrained-plan.ssp")
    capacity_kib = (
        max(row["layout_bytes"] for row in gradient_spill_evidence(baseline)) + 1023
    ) // 1024
    capacities = sorted(
        {
            max(16, ((int(capacity_kib * factor) + 3) // 4) * 4)
            for factor in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25)
        }
    )
    rows = []
    for kib in capacities:
        try:
            selected = plan_program(
                profiled,
                memory_budgets=MemoryBudgets(
                    baseline.fixed_device_bytes + kib * 1024, 64 << 20
                ),
                communication_info=communication,
                optimizer_ordering="tail",
                search_options=SearchOptions(
                    workers=1,
                    generic=GenericPlanningOptions(
                        minimum_object_bytes_evict_eligible=0
                    ),
                ),
            )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            rows.append(dict(kib=kib, refusal=type(error).__name__, detail=str(error)))
        else:
            assert not selected.simulation_only_reasons
            evidence = gradient_spill_evidence(selected)
            rows.append(dict(kib=kib, paths=evidence))
            if all(path["roundtrips"] for path in evidence):
                (directory / "gradient-spill-planning.json").write_text(
                    json.dumps(dict(selected_kib=kib, candidates=rows), indent=2) + "\n"
                )
                return selected
    (directory / "gradient-spill-planning.json").write_text(
        json.dumps(dict(selected_kib=None, candidates=rows), indent=2) + "\n"
    )
    raise AssertionError(
        "No admitted Plan restores a parameter gradient before accumulation"
    )


def check_gradient_spill_trace(trace, path):
    """Observed transfers must match the admitted Plan's gradient destinations."""
    assert trace["summary"]["trace_complete"]
    assert path["roundtrips"]
    for pair in path["roundtrips"]:
        events = {}
        for direction in ("evict", "fetch"):
            wanted = pair[direction]
            matches = [
                record
                for record in trace["transfers"][direction].values()
                if record["alias_group_id"] == wanted["alias"]
                and record["triggered_by"] == f"execution_{wanted['ordinal']:06d}"
            ]
            assert len(matches) == 1, (wanted, matches)
            record = matches[0]
            assert record["bytes"] > 0
            lane = record["lane"]
            assert (
                0 <= lane["lane_started_at_seconds"] <= lane["lane_finished_at_seconds"]
            )
            events[direction] = lane
        assert (
            events["evict"]["lane_finished_at_seconds"]
            <= events["fetch"]["lane_started_at_seconds"] + 1e-6
        )
