"""Adapt measured phase artifacts to the diagnostic planner's saved input form."""

from dataclasses import replace

from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.planning.admission.program import program_admission
from shadowspill.planning.costs import simulation_inputs
from shadowspill.planning.training.bind import bind_training


def problem(profiled, ordering, budgets, communication, fixed_bytes):
    profiled = replace(profiled, program=bind_training(profiled.program, ordering))
    costed, config = simulation_inputs(profiled, budgets, communication)
    # Compare every geometry at the same conservative overhead reservation.
    usable = budgets.execution_bytes - fixed_bytes
    config = replace(
        config, devices=(replace(config.devices[0], capacity_bytes=usable),)
    )
    facts = program_admission(profiled, costed, usable, communication)
    if facts is None:
        raise ValueError(
            "pressure-fit collection needs measured allocation evidence; "
            "use plan_program/write_report for simulation with estimated memory"
        )
    return ShadowSpillPlanningProblem(
        role="forward"
        if profiled.program.bindings.get("kind") == "forward"
        else "step",
        program=costed,
        initial_residency=profiled.program.initial_residency,
        final_residency=profiled.program.final_residency,
        simulation_config=config,
        admission_facts=facts,
        source_execution_budget_bytes=usable,
        maximum_execution_budget_bytes=usable,
        maximum_spill_budget_bytes=budgets.spill_bytes,
        fixed_execution_bytes=0,
        object_reserve_bytes=0,
        dynamic_scratch_reserve_bytes=0,
    ), labels(profiled.program, costed.digest)


def labels(program, digest):
    definitions = {item.profile_key: item for item in program.task_definitions}
    return {
        "program_digest": digest,
        "objects": {
            item.object_id: {"label": ", ".join(program.sources[item.alias_group_id])}
            for item in program.objects
            if program.sources.get(item.alias_group_id)
        },
        "tasks": {
            task.task_id: {
                "label": program.bindings.get("task_names", {}).get(
                    task.task_id, definitions[task.profile_id].name or task.task_id
                ),
                "module_paths": list(definitions[task.profile_id].module_paths),
            }
            for task in program.tasks
        },
    }
