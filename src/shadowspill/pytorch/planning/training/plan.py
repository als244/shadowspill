"""Planning: the step's program searched and its fixed layout certified."""

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
    FixedLayoutSelection,
    placement_facts,
    resolve_fixed_layout_selection,
)

from ..artifacts import TrainingProgramArtifacts
from ..stores import PlanningStores


def plan_training_programs(
    programs: TrainingProgramArtifacts,
    *,
    stores: PlanningStores,
    timer: PlanningTimer,
    search_options: SearchOptions | None = None,
    incumbent: ProgramPlanResult | None = None,
) -> FixedLayoutSelection:
    """Resolve the step's plan: the one in the store, or a fresh search.

    `incumbent` is the plan to beat.
    """

    lowered = programs.lowered
    with timer.measure("feasibility_preflight"):
        try:
            validate_schedule_feasibility(
                lowered.program,
                initial_residency=lowered.initial_residency,
                final_residency=lowered.final_residency,
                config=programs.simulation_config,
                admission=programs.admission,
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
            return resolve_fixed_layout_selection(
                programs.simulation_config,
                programs.admission,
                lambda config: resolve_plan(
                    stores.store,
                    stores.plans,
                    lowered.program,
                    initial_residency=lowered.initial_residency,
                    final_residency=lowered.final_residency,
                    config=config,
                    search_options=search_options,
                    incumbent=incumbent,
                    placement=placement_facts(
                        programs.admission,
                        scratch_reserve_bytes=scratch_reserve,
                    ),
                    progress=timer.progress,
                ),
                scratch_reserve_bytes=scratch_reserve,
                progress=timer.progress,
            )
        except PlanInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PlanSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
        except FixedLayoutInfeasibleError as error:
            raise AdmissionError(f"fixed slab admission failed: {error}") from error
