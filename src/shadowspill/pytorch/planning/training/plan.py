"""Planning: the step's program searched and its fixed layout certified."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.ir import ShadowSpillProgram
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
from shadowspill.pytorch.distributed import current as distributed_preparation
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
    keep_resolutions: bool = False,
) -> FixedLayoutSelection:
    if distributed_preparation() is None:
        return _plan_local_training_program(
            programs,
            stores=stores,
            timer=timer,
            search_options=search_options,
            incumbent=incumbent,
            keep_resolutions=keep_resolutions,
        )
    from shadowspill.pytorch.distributed._selection import (
        certify_restored,
        choose,
        fixed_admission,
        record_decision,
        restore_result,
    )

    def attempt(
        fixed: ShadowSpillProgram, carried: ProgramPlanResult | None
    ) -> FixedLayoutSelection:
        result = _plan_local_training_program(
            replace(
                programs,
                lowered=replace(programs.lowered, program=fixed),
                admission=fixed_admission(programs.admission, fixed),
            ),
            stores=stores,
            timer=timer,
            search_options=search_options,
            incumbent=carried,
            keep_resolutions=False,
        )
        # The certificate belongs to the fixed local cache entry. The restored
        # original-program result below has a separate distributed decision.
        stores.plans.certify(result.plan, result.admission)
        return result

    local, key, successful, decisions = choose(
        programs.lowered.program,
        attempt,
        search_options=search_options,
        incumbent=incumbent,
    )
    result = restore_result(
        local,
        programs.lowered.program,
        key,
        successful,
        keep_resolutions=keep_resolutions,
    )
    facts, admission = certify_restored(
        result,
        programs.admission,
        local.facts,
        scratch_reserve_bytes=local.admission.layout.scratch_reserve_bytes,
    )
    record_decision(stores.store, decisions)
    return replace(
        local,
        plan=replace(local.plan, result=result, key="", certificate=admission),
        facts=facts,
        admission=admission,
    )


def _plan_local_training_program(
    programs: TrainingProgramArtifacts,
    *,
    stores: PlanningStores,
    timer: PlanningTimer,
    search_options: SearchOptions | None = None,
    incumbent: ProgramPlanResult | None = None,
    keep_resolutions: bool = False,
) -> FixedLayoutSelection:
    """Resolve the step's plan: the one in the store, or a fresh search.

    `incumbent` is the plan to beat; `keep_resolutions` files every
    resolution's best plan beside the answer.
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
                    keep_resolutions=keep_resolutions,
                ),
                scratch_reserve_bytes=scratch_reserve,
                progress=timer.progress,
                certify_resolution=(
                    stores.plans.certify_resolution if keep_resolutions else None
                ),
            )
        except PlanInfeasibleError as error:
            raise public_infeasible_plan_error(error) from error
        except PlanSearchExhaustedError as error:
            raise public_search_exhausted_error(error) from error
        except FixedLayoutInfeasibleError as error:
            raise AdmissionError(f"fixed slab admission failed: {error}") from error
