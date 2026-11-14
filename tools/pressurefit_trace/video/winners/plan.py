"""Project a saved Plan into the diagnostic renderer's verified plan view."""

from dataclasses import replace

from shadowspill.planner.admission.layout.build import _simulation_input
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.planner.result import ProgramPlanResult, ResidentSlice
from shadowspill.planning.costs import simulation_inputs
from shadowspill.simulator import simulate


def diagnostic_plan(plan, options):
    layout, facts = plan.memory_layout, plan._admission_facts
    if layout is None or facts is None:
        raise ValueError("winner videos require a certified physical layout")
    costed, config = simulation_inputs(
        plan.profiled_program, plan.memory_budgets, plan.communication_info
    )
    if costed.digest != plan._costed_program.digest:
        raise ValueError("saved costs differ from the Plan's communication inputs")
    logical = simulate(costed, plan.schedule, selections=plan.selections, config=config)
    # The certified resident slice is a suffix of the fixed slice. Use its
    # actual placements, not a guessed cutoff or the current planner defaults.
    resident_start = layout.fixed_slice_bytes - layout.resident_slice_bytes
    resident = ResidentSlice(
        layout.resident_slice_bytes,
        tuple(
            sorted(
                {
                    item.alias_group_id
                    for item in layout.placements
                    if item.alias_group_id is not None and item.offset >= resident_start
                }
            )
        ),
    )
    result = ProgramPlanResult(
        program=costed,
        search_options=options,
        initial_residency=plan.program.initial_residency,
        final_residency=plan.program.final_residency,
        simulation_config=config,
        schedule=plan.schedule,
        selections=plan.selections,
        simulation=logical,
        diagnostics=plan.diagnostics.replace_selected_makespan(logical.makespan_ns),
        resident_slice=resident,
        placement_facts=facts,
    )
    problem = ShadowSpillPlanningProblem(
        role="forward" if plan.program.bindings.get("kind") == "forward" else "step",
        program=costed,
        initial_residency=result.initial_residency,
        final_residency=result.final_residency,
        simulation_config=config,
        admission_facts=facts,
        source_execution_budget_bytes=plan.memory_budgets.execution_bytes,
        maximum_execution_budget_bytes=plan.memory_budgets.execution_bytes,
        maximum_spill_budget_bytes=plan.memory_budgets.spill_bytes,
        fixed_execution_bytes=plan.fixed_device_bytes,
        object_reserve_bytes=0,
        dynamic_scratch_reserve_bytes=layout.scratch_reserve_bytes,
    )
    annotated = AnnotatedProgramPlan(
        program=problem,
        memory_budgets=plan.memory_budgets,
        transfer_bandwidths=problem.transfer_bandwidths,
        result=replace(
            result,
            simulation=plan.timeline,
            diagnostics=plan.diagnostics.replace_selected_makespan(
                plan.timeline.makespan_ns
            ),
        ),
        effective_facts=facts,
        fixed_layout=layout,
        simulation_admission=_simulation_input(result, layout),
        simulation=plan.timeline,
        attempts=(),
        plan_from_store=False,
        wall_time_ns=0,
    )
    return annotated, result
