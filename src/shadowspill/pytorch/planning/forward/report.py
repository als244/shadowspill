"""The plan report one forward planning call writes."""

import torch.nn as nn

from shadowspill.ir import (
    ExecutionPlan,
)
from shadowspill.planner import (
    ProgramPlanResult,
)
from shadowspill.pytorch.diagnostics.builders import forward_stage_inventory
from shadowspill.runtime.plan import PlanMemory

from ...diagnostics import PlanReport
from ..admission import (
    FixedLayoutSelection,
    SelectedAdmission,
)
from ..artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
)
from ..common import (
    PlanningTimer,
)
from ..reporting import (
    build_forward_report,
    cache_artifacts,
    fixed_layout_diagnostic,
    publish_plan_report,
)
from ..stores import PlanningStores


def _forward_plan_report(
    model: nn.Module,
    captured: ForwardCaptureArtifacts,
    profiled: ForwardProfileArtifacts,
    program: ForwardProgramArtifacts,
    selection: FixedLayoutSelection,
    selected_admission: SelectedAdmission,
    admitted_result: ProgramPlanResult,
    execution_plan: ExecutionPlan,
    *,
    stores: PlanningStores,
    memory: PlanMemory,
    timer: PlanningTimer,
    started: int,
) -> PlanReport:
    with timer.measure("diagnostic_inventory"):
        task_stage_map, unique_stages = forward_stage_inventory(
            program.lowered,
            execution_plan,
            program.measurements_by_profile,
            profiled.compiled_tasks.manifests,
            profiling_metadata_digest=captured.workload.digest,
        )
    report = build_forward_report(
        captured.signature.digest,
        execution_plan,
        profiled.profiles,
        tuple(timer.values),
        started,
        planned_program_cache_hit=selection.from_store,
        search_results=(admitted_result,),
        captured_stage_count=len(captured.partitioned.stages),
        aot_unique_stage_contracts=profiled.profiles.unique_keys,
        task_stage_map=task_stage_map,
        unique_stages=unique_stages,
        compiler_phase_timings_ns=(
            profiled.profiler.executables.compilation_phase_timings_ns
        ),
        compiler_phase_timings_by_contract=(
            profiled.profiler.executables.compilation_phase_timings_by_contract
        ),
        store_directories=stores.store.diagnostics(),
        touched_cache_artifacts=cache_artifacts(stores.store),
        profiling_metadata=(captured.workload,),
        physical_layouts=(
            fixed_layout_diagnostic(
                "forward",
                selection,
                selected_admission,
            ),
        ),
        memory=memory,
    )
    return publish_plan_report(
        model,
        report,
        stores.store,
        started=started,
    )
