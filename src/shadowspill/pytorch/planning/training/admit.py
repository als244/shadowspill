"""Admission: the selected tasks compiled, the plans physically admitted and sealed,
and the training callable published with its report."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from shadowspill.errors import (
    CompilationError,
)
from shadowspill.ir import EntrypointSpec, ExecutionPlan, PhysicalAdmission
from shadowspill.pipeline.admission import (
    physical_admission,
    project_runtime_fixed_layout,
    reconcile_spill_pool,
    seal_physical_budget,
)
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.planner import (
    ProgramPlanResult,
)
from shadowspill.planner.search import SearchOptions
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.optimizer import (
    OptimizerCapture,
    OptimizerTaskArtifact,
)
from shadowspill.pytorch.planning.admission import (
    SelectedAdmission,
    build_fixed_selected_admission,
    output_bindings_for_entrypoints,
)
from shadowspill.pytorch.profiling import ResolvedTaskManifests
from shadowspill.runtime.abi import INITIAL_ACTIONS_TASK_ID
from shadowspill.runtime.bootstrap import (
    InstalledRuntime,
)
from shadowspill.runtime.plan import (
    PlanMemory,
    RuntimeBridge,
)
from shadowspill.step import StepDataOrdering

from ...callables import PlannedTrainStep
from ...execution import TrainingExecutor
from ...lowering.training import (
    LoweredTrainingProgram,
)
from ..artifacts import (
    TrainingAdmissionArtifacts,
    TrainingCaptureArtifacts,
    TrainingExecutableArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
    TrainingSelections,
)
from ..stores import PlanningStores
from .materialize import rollback_training_failure, rollback_training_materialization
from .profile import release_build_executables
from .report import training_plan_report


def compile_selected_training_tasks(
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    *,
    installed: InstalledRuntime,
    timer: PlanningTimer,
) -> TrainingExecutableArtifacts:
    """Retain executable callables only for selected task variants."""

    required = _selected_artifact_digests(
        programs.recurrent,
        selections.recurrent.result,
    )
    if selections.initial is not None:
        required.update(
            _selected_artifact_digests(programs.initial, selections.initial.result)
        )
    selected_tasks = tuple(
        artifact
        for artifact in profiled.compile_tasks
        if artifact.compatibility_digest in required
    )
    with timer.measure("compilation"):
        compiled = profiled.profiler.take_compiled_tasks(
            selected_tasks,
            progress=lambda index, total, state, digest: timer.progress(
                f"selected entrypoint {index}/{total} {state}: {digest[:12]}"
            ),
        )
        _verify_compiled_manifest_identity(profiled.manifests, compiled)
        release_build_executables(profiled, installed)
    timer.attribute_compilation_and_profiling(profiled.profiler.wall_times)
    return TrainingExecutableArtifacts(compiled)


def _entrypoint_contract(
    artifact: GraphArtifact | OptimizerTaskArtifact | None,
) -> str | None:
    """The compiled contract behind one task, where there is one."""

    return None if artifact is None else artifact.compatibility_digest


def admit_training_plan(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    executable: TrainingExecutableArtifacts,
    *,
    memory: PlanMemory,
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    data_ordering: StepDataOrdering,
    stores: PlanningStores,
    timer: PlanningTimer,
    started: int,
    search_options: SearchOptions | None = None,
) -> PlannedTrainStep:
    """Physically admit selections and publish the training callable/report."""

    try:
        admitted = _admit_training_execution_plans(
            captured,
            profiled,
            programs,
            selections,
            materialized.optimizer_capture,
            memory,
            timer,
        )
        recurrent_plan = admitted.recurrent
        initial_plan = admitted.initial
        recurrent_fixed_layout = admitted.recurrent_admission.fixed_layout
        if recurrent_fixed_layout is None:
            raise AssertionError("recurrent admission did not produce a fixed layout")
        recurrent_runtime_layout = project_runtime_fixed_layout(
            recurrent_fixed_layout,
            recurrent_plan.program,
            recurrent_plan.schedule,
            initial_task_id=INITIAL_ACTIONS_TASK_ID,
            dynamic_task_allocations=(
                admitted.recurrent_admission.dynamic_provider_allocations()
            ),
        )
        initial_runtime_layout = None
        if initial_plan is not None:
            initial_admission = admitted.initial_admission
            if initial_admission is None or initial_admission.fixed_layout is None:
                raise AssertionError("initial admission did not produce a fixed layout")
            initial_runtime_layout = project_runtime_fixed_layout(
                initial_admission.fixed_layout,
                initial_plan.program,
                initial_plan.schedule,
                initial_task_id=INITIAL_ACTIONS_TASK_ID,
                dynamic_task_allocations=(
                    initial_admission.dynamic_provider_allocations()
                ),
            )
        bridge = RuntimeBridge(
            memory.runtime,
            recurrent_plan.program,
            memory.plan_handle,
            execution_pool_id=memory.execution.pool_id,
            spill_pool_id=memory.spill.pool_id,
        )
        with timer.measure("plan_adoption"):
            materialized.state.adopt_execution_plan(
                bridge,
                programs.recurrent,
                optimizer=materialized.optimizer,
            )
        with timer.measure("physical_sealing"):
            seal_physical_budget(
                captured.installed,
                recurrent_plan,
                recurrent_fixed_layout,
            )
        with timer.measure("callable_construction"):
            executor = TrainingExecutor(
                None if initial_plan is None else (programs.initial, initial_plan),
                (programs.recurrent, recurrent_plan),
                bridge,
                materialized.state,
                executable.tasks.functions,
                materialized.optimizer,
                recurrent_simulation=admitted.recurrent_admission.simulation,
                initial_simulation=(
                    None
                    if admitted.initial_admission is None
                    else admitted.initial_admission.simulation
                ),
                initial_fixed_layout=initial_runtime_layout,
                recurrent_fixed_layout=recurrent_runtime_layout,
                initial_memory_envelopes=(
                    None
                    if admitted.initial_admission is None
                    else admitted.initial_admission.envelopes_by_task()
                ),
                recurrent_memory_envelopes=(
                    admitted.recurrent_admission.envelopes_by_task()
                ),
                # State planning installed is already initialized, so there
                # is nothing for an initial step to create.
                optimizer_state_preinitialized=(
                    materialized.optimizer_capture.initialized_state_dict is not None
                    or materialized.installed_state_entries > 0
                ),
                optimizer_state_was_lazy=bool(
                    materialized.optimizer_capture.created_state_names
                ),
            )
        report = training_plan_report(
            model,
            captured,
            profiled,
            programs,
            selections,
            admitted,
            recurrent_plan,
            initial_plan,
            optimizer_ordering=optimizer_ordering,
            data_ordering=data_ordering,
            stores=stores,
            memory=memory,
            timer=timer,
            started=started,
            search_options=search_options,
        )
        return PlannedTrainStep(
            model,
            captured.signatures,
            executor,
            materialized.state,
            report,
            memory.runtime,
            memory.plan_handle,
        )
    except BaseException as error:
        rollback_training_failure(
            memory.runtime,
            error,
            lambda: rollback_training_materialization(model, materialized),
            operation="admit training plan",
        )


def _admit_training_execution_plans(
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    optimizer_capture: OptimizerCapture,
    memory: PlanMemory,
    timer: PlanningTimer,
) -> TrainingAdmissionArtifacts:
    recurrent = selections.recurrent.result
    initial = None if selections.initial is None else selections.initial.result
    predicted_spill_peak = max(
        recurrent.simulation.spill_peak_bytes,
        0 if initial is None else initial.simulation.spill_peak_bytes,
    )
    with timer.measure("spill_admission"):
        reconcile_spill_pool(
            predicted_peak=predicted_spill_peak,
            budget=memory.spill_budget,
        )
    admissions = _build_training_admissions(
        captured,
        profiled,
        programs,
        selections,
        memory,
        timer,
    )
    admission = physical_admission(
        memory,
        captured.installed,
        workspace_reserve=programs.workspace_reserve,
        predicted_spill_peak_bytes=predicted_spill_peak,
        predicted_fragmentation_bytes=max(
            item.predicted_fragmentation_bytes for item in admissions
        ),
    )
    recurrent_plan = _execution_plan(
        programs.recurrent,
        admissions[0].apply_prediction(recurrent),
        optimizer_capture.optimizer_type,
        admission,
    )
    initial_plan = (
        _execution_plan(
            programs.initial,
            admissions[1].apply_prediction(initial),
            optimizer_capture.optimizer_type,
            admission,
        )
        if initial is not None
        else None
    )
    recurrent_admission = admissions[0]
    initial_admission = admissions[1] if len(admissions) == 2 else None
    recurrent_result = recurrent_admission.apply_prediction(recurrent)
    initial_result = (
        None
        if initial is None or initial_admission is None
        else initial_admission.apply_prediction(initial)
    )
    return TrainingAdmissionArtifacts(
        recurrent_plan,
        initial_plan,
        recurrent_admission,
        initial_admission,
        recurrent_result,
        initial_result,
    )


def _build_training_admissions(
    captured: TrainingCaptureArtifacts,
    profiled: TrainingProfileArtifacts,
    programs: TrainingProgramArtifacts,
    selections: TrainingSelections,
    memory: PlanMemory,
    timer: PlanningTimer,
) -> tuple[SelectedAdmission, ...]:
    pairs = [(programs.recurrent, selections.recurrent)]
    if selections.initial is not None:
        pairs.append((programs.initial, selections.initial))
    with timer.measure("slab_admission"):
        admitted: list[SelectedAdmission] = []
        for lowered, fixed_selection in pairs:
            selected = fixed_selection.result
            output_bindings = output_bindings_for_entrypoints(
                selected.program.selected_tasks(selected.selections),
                lowered.entrypoints,
                {
                    item.object_id: item.alias_group_id
                    for item in selected.program.objects
                },
            )
            admitted.append(
                build_fixed_selected_admission(
                    selected,
                    programs.measurements_by_profile,
                    fixed_admission=fixed_selection.admission,
                    output_bindings=output_bindings,
                ),
            )
        return tuple(admitted)


def _selected_artifact_digests(
    lowered: LoweredTrainingProgram,
    selected: ProgramPlanResult,
) -> set[str]:
    selected_task_ids = {
        task.task_id for task in lowered.program.selected_tasks(selected.selections)
    }
    digests = set()
    for entrypoint in lowered.entrypoints:
        if entrypoint.task_id not in selected_task_ids:
            continue
        artifact = lowered.executables.get(entrypoint.task_id)
        if artifact is not None:
            digests.add(artifact.compatibility_digest)
    return digests


def _verify_compiled_manifest_identity(
    planned: ResolvedTaskManifests,
    executable: CompiledTaskSet,
) -> None:
    for digest, manifest in executable.manifests.items():
        expected = planned.manifests.get(digest)
        if expected is None or (
            expected.compatibility_digest != manifest.compatibility_digest
        ):
            raise CompilationError(
                "selected compiled entrypoint changed its storage contract: "
                f"artifact={digest}"
            )


def _execution_plan(
    lowered: LoweredTrainingProgram,
    selected: ProgramPlanResult,
    optimizer_type: str,
    admission: PhysicalAdmission,
) -> ExecutionPlan:
    selected_ids = {
        task.task_id for task in lowered.program.selected_tasks(selected.selections)
    }
    active_entrypoints = tuple(
        item for item in lowered.entrypoints if item.task_id in selected_ids
    )
    return selected.to_execution_plan(
        entrypoints=tuple(
            EntrypointSpec(
                item.task_id,
                f"entrypoint_{index:06d}",
                "pytorch_inductor"
                if item.options.phase != "optimizer"
                else "pytorch_optimizer",
                _entrypoint_contract(lowered.executables.get(item.task_id))
                or optimizer_type,
            )
            for index, item in enumerate(active_entrypoints)
        ),
        admission=admission,
    )
