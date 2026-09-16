"""Planning: the recurrent and, when the optimizer creates state on its first step,
the initial program searched and their fixed layouts certified."""

from __future__ import annotations

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.pipeline.admission import dynamic_scratch_reserve_bytes
from shadowspill.pipeline.common import (
    PlanningTimer,
    public_infeasible_plan_error,
    public_search_exhausted_error,
)
from shadowspill.planner import (
    ProgramPlanResult,
    validate_schedule_feasibility,
)
from shadowspill.planner.plan_store import resolve_plan
from shadowspill.planner.search import SearchOptions
from shadowspill.pytorch.planning.admission import (
    FixedLayoutInfeasibleError,
    placement_facts,
    resolve_fixed_layout_selection,
)

from ..artifacts import (
    TrainingProgramArtifacts,
    TrainingSelections,
)
from ..stores import PlanningStores


def plan_training_programs(
    programs: TrainingProgramArtifacts,
    *,
    stores: PlanningStores,
    timer: PlanningTimer,
    search_options: SearchOptions | None = None,
    incumbent: ProgramPlanResult | None = None,
) -> TrainingSelections:
    """Resolve recurrent and, when required, lazy-state first-step selections.

    `incumbent` is the plan to beat for the recurrent program; the first-step
    program, when there is one, is searched on its own.
    """

    needs_initial = any(
        item.created_on_first_step for item in programs.initial.optimizer_objects
    )
    with timer.measure("feasibility_preflight"):
        try:
            validate_schedule_feasibility(
                programs.recurrent.program,
                initial_residency=programs.recurrent.initial_residency,
                final_residency=programs.recurrent.final_residency,
                config=programs.simulation_config,
                admission=programs.recurrent_admission,
                search_options=search_options,
            )
            if needs_initial:
                validate_schedule_feasibility(
                    programs.initial.program,
                    initial_residency=programs.initial.initial_residency,
                    final_residency=programs.initial.final_residency,
                    config=programs.simulation_config,
                    admission=programs.initial_admission,
                    search_options=search_options,
                )
        except PlanInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PlanSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
    scratch_reserve = dynamic_scratch_reserve_bytes(
        programs.measurements_by_profile,
        minimum_bytes=programs.dynamic_scratch_reserve_bytes,
    )
    with timer.measure("search"):
        try:
            recurrent = resolve_fixed_layout_selection(
                programs.simulation_config,
                programs.recurrent_admission,
                lambda config: resolve_plan(
                    stores.store,
                    stores.plans,
                    programs.recurrent.program,
                    initial_residency=programs.recurrent.initial_residency,
                    final_residency=programs.recurrent.final_residency,
                    config=config,
                    search_options=search_options,
                    incumbent=incumbent,
                    placement=placement_facts(
                        programs.recurrent_admission,
                        scratch_reserve_bytes=scratch_reserve,
                    ),
                    progress=timer.progress,
                ),
                scratch_reserve_bytes=scratch_reserve,
                progress=timer.progress,
            )
            initial = (
                resolve_fixed_layout_selection(
                    programs.simulation_config,
                    programs.initial_admission,
                    lambda config: resolve_plan(
                        stores.store,
                        stores.plans,
                        programs.initial.program,
                        initial_residency=programs.initial.initial_residency,
                        final_residency=programs.initial.final_residency,
                        config=config,
                        search_options=search_options,
                        placement=placement_facts(
                            programs.initial_admission,
                            scratch_reserve_bytes=scratch_reserve,
                        ),
                        progress=timer.progress,
                    ),
                    scratch_reserve_bytes=scratch_reserve,
                    progress=timer.progress,
                )
                if needs_initial
                else None
            )
        except PlanInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PlanSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
        except FixedLayoutInfeasibleError as error:
            raise AdmissionError(f"fixed slab admission failed: {error}") from error
    return TrainingSelections(recurrent, initial)
