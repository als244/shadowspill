"""The plan report of a training plan: the stage inventory, the store's hits and
misses, the layouts, published beside the plan."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from shadowspill.ir import ExecutionPlan
from shadowspill.planner.search import SearchOptions
from shadowspill.pytorch.diagnostics.builders import training_stage_inventory
from shadowspill.runtime.plan import PlanMemory
from shadowspill.step import StepDataOrdering

from ...diagnostics import PlanReport
from ..artifacts import (
    TrainingAdmissionArtifacts,
    TrainingCaptureArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
    TrainingSelections,
)
from ..common import (
    PlanningTimer,
)
from ..reporting import (
    build_training_report,
    cache_artifacts,
    fixed_layout_diagnostic,
    publish_plan_report,
)
from ..stores import PlanningStores


def training_plan_report(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    admitted: TrainingAdmissionArtifacts,
    recurrent_plan: ExecutionPlan,
    initial_plan: ExecutionPlan | None,
    *,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: StepDataOrdering,
    stores: PlanningStores,
    memory: PlanMemory,
    timer: PlanningTimer,
    started: int,
    search_options: SearchOptions | None = None,
) -> PlanReport:
    with timer.measure("diagnostic_inventory"):
        task_stage_map, unique_stages = training_stage_inventory(
            captured.partitioned,
            programs.recurrent,
            recurrent_plan,
            programs.measurements,
            profiled.manifests.manifests,
            profiling_metadata_digests=tuple(
                item.digest for item in captured.workloads
            ),
            data_ordering=data_ordering,
        )
    hits = int(selections.recurrent.from_store) + (
        0 if selections.initial is None else int(selections.initial.from_store)
    )
    misses = int(not selections.recurrent.from_store) + (
        0 if selections.initial is None else int(not selections.initial.from_store)
    )
    report = build_training_report(
        tuple(signature.digest for signature in captured.signatures),
        recurrent_plan,
        profiled.profiles,
        tuple(timer.values),
        started,
        initial_execution_plan=initial_plan,
        planned_program_cache_hits=hits,
        planned_program_cache_misses=misses,
        captured_stage_count=sum(
            len(capture.stages) for capture in captured.partitioned
        ),
        aot_unique_stage_contracts=stores.graph_pairs.unique_keys,
        aot_graph_pair_cache_hits=stores.graph_pairs.hits,
        aot_graph_pair_cache_misses=stores.graph_pairs.misses,
        search_results=(
            (admitted.recurrent_result,)
            if admitted.initial_result is None
            else (admitted.initial_result, admitted.recurrent_result)
        ),
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
        profiling_metadata=captured.workloads,
        physical_layouts=(
            *(
                ()
                if selections.initial is None or admitted.initial_admission is None
                else (
                    fixed_layout_diagnostic(
                        "initial",
                        selections.initial,
                        admitted.initial_admission,
                    ),
                )
            ),
            fixed_layout_diagnostic(
                "recurrent",
                selections.recurrent,
                admitted.recurrent_admission,
            ),
        ),
        optimizer_ordering=optimizer_ordering,
        data_ordering=data_ordering,
        search_options=search_options,
        memory=memory,
    )
    return publish_plan_report(
        model,
        report,
        stores.store,
        started=started,
    )
