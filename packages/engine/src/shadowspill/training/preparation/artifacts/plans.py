"""CPU source, cost and admission rechecks for saved operation Plans."""

from dataclasses import replace

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.planning.admission.program import program_admission
from shadowspill.planning.costs import simulation_inputs
from shadowspill.planning.memory_search.admission.layout.dependencies import (
    simulator_reuse_dependencies,
)
from shadowspill.planning.training.bind import bind_training
from shadowspill.profiling.apply import apply_profiles
from shadowspill.simulator import SimulationAdmission, simulate
from shadowspill.step.ordering import StepDataOrdering
from shadowspill.training.preparation.capture.templates import (
    TemplateInventory,
    TemplateUse,
    instantiate_template,
)


def validate_plan(
    inventory: TemplateInventory,
    use: TemplateUse,
    ordering: StepDataOrdering | None,
    source: ProfiledProgram,
    plan: Plan,
) -> None:
    """Re-derive saved IR and replay its schedule; never search or compile."""
    setup = inventory.layouts.setup
    if (plan._accumulation_plan is not None) != (use.kind == "forward_backward"):
        raise ValueError("saved Plan is missing or adds a gradient accumulation path")
    program = instantiate_template(
        inventory.templates[use.template], use, source.program
    )
    profiled = apply_profiles(program, source.profiles)
    ordering = ordering if use.kind in ("step", "forward_backward") else None
    budgets = MemoryBudgets(
        int(setup.config.memory.device_gib * (1 << 30)),
        int(setup.config.memory.spill_gib * (1 << 30)),
    )
    for accumulated, path in ((False, plan), (True, plan._accumulation_plan)):
        if path is None:
            continue
        expected = replace(
            profiled, program=bind_training(program, ordering, accumulate=accumulated)
        )
        if path.profiled_program.digest != expected.digest:
            raise ValueError(
                "saved Plan differs from its source operation/count/ordering"
            )
        if (
            path.memory_budgets != budgets
            or path.program.parallelism != setup.context.parallelism
        ):
            raise ValueError("saved Plan differs from requested budgets or parallelism")
        if path.simulation_only_reasons:
            raise ValueError("saved training Plan is simulation-only")
        if (
            path.pool_memory != plan.pool_memory
            or path.communication_info != plan.communication_info
        ):
            raise ValueError(
                "saved gradient paths disagree on pool or communication evidence"
            )
        costed, config = simulation_inputs(
            expected,
            budgets,
            path.communication_info,
            path.pool_memory,
            path.reserved_spill_bytes,
        )
        if costed.digest != path._costed_program.digest:
            raise ValueError("saved Plan costed program differs from its profiles")
        facts = program_admission(
            expected,
            costed,
            path.effective_execution_bytes - path.fixed_device_bytes,
            path.communication_info,
        )
        if (
            facts is None
            or path._admission_facts is None
            or facts.digest != path._admission_facts.digest
        ):
            raise ValueError(
                "saved Plan allocation facts differ from measured evidence"
            )
        layout = path.memory_layout
        assert layout is not None
        device = costed.devices[0].device_id
        replay = simulate(
            costed,
            path.schedule,
            selections=path.selections,
            config=config,
            admission=SimulationAdmission(
                initial_physical_bytes=((device, layout.required_bytes),),
                device_capacity_bytes=((device, layout.pool_capacity_bytes),),
                unlisted_delta_bytes=0,
                reuse_dependencies=simulator_reuse_dependencies(
                    layout.reuse_dependencies
                ),
            ),
        )
        # Fresh/additive modes may enlarge a common fixed arena after simulation;
        # its padded physical peak is re-admitted by joint_memory. Scheduling and
        # retained spill evidence must still match this exact saved invocation.
        if (
            replay.makespan_ns != path.timeline.makespan_ns
            or replay.spill_peak_bytes != path.timeline.spill_peak_bytes
            or replay.spill_transient_peak_bytes
            != path.timeline.spill_transient_peak_bytes
            or replay.spill_transient_end_bytes
            != path.timeline.spill_transient_end_bytes
        ):
            raise ValueError(
                "saved Plan timing/spill evidence differs from schedule replay"
            )
