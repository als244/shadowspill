"""The search over the forward program, and the layout it certifies."""

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.ir import (
    EntrypointSpec,
    ExecutionPlan,
    PhysicalAdmission,
)
from shadowspill.planner import (
    ProgramPlanResult,
    validate_schedule_feasibility,
)
from shadowspill.planner.plan_store import resolve_plan
from shadowspill.planner.search import SearchOptions

from ...lowering.forward import LoweredForwardProgram
from ..admission import (
    FixedLayoutInfeasibleError,
    FixedLayoutSelection,
    dynamic_scratch_reserve_bytes,
    placement_facts,
    resolve_fixed_layout_selection,
)
from ..artifacts import (
    ForwardProgramArtifacts,
)
from ..common import (
    PlanningTimer,
    public_infeasible_plan_error,
    public_search_exhausted_error,
)
from ..stores import PlanningStores


def plan_forward_program(
    program: ForwardProgramArtifacts,
    *,
    search_options: SearchOptions | None,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> FixedLayoutSelection:
    """Resolve the exact plan for a canonical forward program.

    `search_options` reaches both the preflight and the search, so a forward
    plan is searched under what the caller asked for rather than under the
    defaults.
    """

    with timer.measure("feasibility_preflight"):
        try:
            validate_schedule_feasibility(
                program.lowered.program,
                initial_residency=program.lowered.initial_residency,
                final_residency=program.lowered.final_residency,
                config=program.simulation_config,
                admission=program.admission,
                search_options=search_options,
            )
        except PlanInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PlanSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
    scratch_reserve = dynamic_scratch_reserve_bytes(
        program.measurements_by_profile,
        minimum_bytes=program.dynamic_scratch_reserve_bytes,
    )
    with timer.measure("search"):
        try:
            return resolve_fixed_layout_selection(
                program.simulation_config,
                program.admission,
                lambda config: resolve_plan(
                    stores.store,
                    stores.plans,
                    program.lowered.program,
                    initial_residency=program.lowered.initial_residency,
                    final_residency=program.lowered.final_residency,
                    config=config,
                    search_options=search_options,
                    placement=placement_facts(
                        program.admission,
                        scratch_reserve_bytes=scratch_reserve,
                    ),
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


def _forward_execution_plan(
    lowered: LoweredForwardProgram,
    selection: ProgramPlanResult,
    admission: PhysicalAdmission,
) -> ExecutionPlan:
    entrypoints = tuple(
        EntrypointSpec(
            task_id=item.task_id,
            entrypoint_id=f"entrypoint_{index:06d}",
            executor_id="pytorch_inductor",
            contract_digest=item.artifact.compatibility_digest,
        )
        for index, item in enumerate(lowered.entrypoints)
    )
    return selection.to_execution_plan(
        entrypoints=entrypoints,
        admission=admission,
    )
