"""One captured graph as the report describes it, and what it allocates."""

from dataclasses import dataclass

from shadowspill.ir import (
    AliasGroupSpec,
    ObjectSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner.diagnostics.plan import (
    PlanAllocationABIStep,
    PlanAllocationEvent,
    PlanCompiledOutputView,
    PlanCompiledRoot,
    PlanGraphProfile,
    PlanMutationBinding,
    PlanObjectFootprint,
    PlanOutputView,
    PlanRepresentativeInput,
    PlanStorageRoot,
)
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    TaskStorageContract,
)
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.task.layout import (
    CompiledTaskLayout,
    reconcile_compiled_task_layout,
    replacement_transition_bytes,
)
from shadowspill.task.manifest import ExecutableTaskManifest


@dataclass(frozen=True, slots=True)
class _GraphProfileProblem:
    artifact: GraphArtifact
    direction: str
    task: TaskSpec
    profile: TaskProfile
    measurement: TaskMeasurement
    manifest: ExecutableTaskManifest
    layout: CompiledTaskLayout
    inputs: tuple[PlanObjectFootprint, ...]
    mutations: tuple[PlanObjectFootprint, ...]
    outputs: tuple[PlanObjectFootprint, ...]


def _graph_profile(
    artifact: GraphArtifact,
    direction: str,
    task: TaskSpec,
    program: ShadowSpillProgram,
    measurement: TaskMeasurement,
    manifest: ExecutableTaskManifest,
) -> PlanGraphProfile:
    inputs, mutations, outputs = _task_footprints(program, task)
    problem = _GraphProfileProblem(
        artifact=artifact,
        direction=direction,
        task=task,
        profile=_task_profile(program, task),
        measurement=measurement,
        manifest=manifest,
        layout=reconcile_compiled_task_layout(
            manifest.storage_contract,
            measurement,
            root_allocations=manifest.root_allocations,
        ),
        inputs=inputs,
        mutations=mutations,
        outputs=outputs,
    )
    return _build_graph_profile(problem)


def _build_graph_profile(problem: _GraphProfileProblem) -> PlanGraphProfile:
    artifact = problem.artifact
    contract = problem.manifest.storage_contract
    measurement = problem.measurement
    return PlanGraphProfile(
        direction=problem.direction,
        structural_contract_key=artifact.compatibility_digest,
        semantic_contract_digest=artifact.storage_contract.compatibility_digest,
        semantic_contract_capture_ns=artifact.storage_contract_capture_ns,
        semantic_roots=_plan_storage_roots(artifact.storage_contract),
        semantic_output_views=_plan_output_views(artifact.storage_contract),
        semantic_mutations=_plan_mutations(artifact.storage_contract),
        executable_contract_digest=contract.compatibility_digest,
        executable_contract_capture_ns=problem.manifest.contract_capture_ns,
        executable_roots=_plan_storage_roots(contract),
        executable_output_views=_plan_output_views(contract),
        executable_mutations=_plan_mutations(contract),
        compiled_layout_digest=problem.layout.compatibility_digest,
        compiled_roots=_plan_compiled_roots(problem.layout),
        compiled_output_views=_plan_compiled_views(problem.layout),
        physical_profile_wall_time_ns=measurement.profiling_wall_time_ns,
        representative_task_id=problem.task.task_id,
        runtime_ns=measurement.runtime_ns,
        samples_ns=measurement.samples_ns,
        provenance=measurement.provenance,
        representative_inputs=_plan_representative_inputs(measurement),
        profile_phase_timings_ns=measurement.phase_timings_ns,
        timing_relative_mad=measurement.timing_relative_mad,
        timing_half_drift=measurement.timing_half_drift,
        timing_unstable=measurement.timing_unstable,
        inputs=problem.inputs,
        mutations=problem.mutations,
        outputs=problem.outputs,
        input_logical_bytes=_logical_bytes(problem.inputs),
        input_allocation_bytes=_unique_allocation_bytes(problem.inputs),
        mutation_logical_bytes=_logical_bytes(problem.mutations),
        mutation_allocation_bytes=_unique_allocation_bytes(problem.mutations),
        output_logical_bytes=_logical_bytes(problem.outputs),
        output_allocation_bytes=_unique_allocation_bytes(problem.outputs),
        workspace_requested_bytes=measurement.workspace_requested_bytes,
        workspace_charged_bytes=measurement.workspace_charged_bytes,
        replacement_transition_bytes=replacement_transition_bytes(
            contract, problem.layout
        ),
        task_workspace_bytes=problem.profile.workspace_bytes,
        workspace_extent_bytes=measurement.workspace_extent_bytes,
        persistent_extent_bytes=measurement.persistent_extent_bytes,
        allocation_contract_digest=(
            None
            if measurement.allocation_contract is None
            else measurement.allocation_contract.compatibility_digest
        ),
        allocation_contract=_plan_allocation_contract(measurement),
        allocation_timeline=_plan_allocation_timeline(measurement),
    )


def _task_profile(program: ShadowSpillProgram, task: TaskSpec) -> TaskProfile:
    return next(
        profile for profile in program.profiles if profile.profile_id == task.profile_id
    )


def _task_footprints(
    program: ShadowSpillProgram,
    task: TaskSpec,
) -> tuple[
    tuple[PlanObjectFootprint, ...],
    tuple[PlanObjectFootprint, ...],
    tuple[PlanObjectFootprint, ...],
]:
    objects = {item.object_id: item for item in program.objects}
    aliases = {item.alias_group_id: item for item in program.alias_groups}

    def resolve(object_ids: tuple[str, ...]) -> tuple[PlanObjectFootprint, ...]:
        return tuple(
            _footprint(objects[object_id], aliases[objects[object_id].alias_group_id])
            for object_id in object_ids
        )

    return (
        resolve(task.inputs),
        resolve(tuple(item.object_id for item in task.mutations)),
        resolve(task.outputs),
    )


def _plan_storage_roots(
    contract: TaskStorageContract,
) -> tuple[PlanStorageRoot, ...]:
    return tuple(_plan_storage_root(root) for root in contract.roots)


def _plan_storage_root(root: StorageRoot) -> PlanStorageRoot:
    return PlanStorageRoot(
        root_id=root.root_id,
        kind=root.kind.value,
        source_input=root.source_input,
        producer_node=root.producer_node,
        producer_target=root.producer_target,
        producer_result=root.producer_result,
        minimum_span_bytes=root.minimum_span_bytes,
    )


def _plan_output_views(
    contract: TaskStorageContract,
) -> tuple[PlanOutputView, ...]:
    return tuple(_plan_output_view(view) for view in contract.output_views)


def _plan_output_view(view: OutputView) -> PlanOutputView:
    return PlanOutputView(
        leaf_index=view.leaf_index,
        root_id=view.root_id,
        offset_bytes=view.offset_bytes,
        span_bytes=view.span_bytes,
        shape=view.shape,
        stride=view.stride,
        dtype=view.dtype,
        layout=view.layout,
    )


def _plan_mutations(
    contract: TaskStorageContract,
) -> tuple[PlanMutationBinding, ...]:
    return tuple(_plan_mutation(mutation) for mutation in contract.mutations)


def _plan_mutation(mutation: MutationBinding) -> PlanMutationBinding:
    return PlanMutationBinding(
        input_position=mutation.input_position,
        replacement_output_leaf=mutation.replacement_output_leaf,
        producer_node=mutation.producer_node,
        producer_target=mutation.producer_target,
        argument_name=mutation.argument_name,
    )


def _plan_compiled_roots(
    layout: CompiledTaskLayout,
) -> tuple[PlanCompiledRoot, ...]:
    return tuple(
        PlanCompiledRoot(
            root_id=root.root_id,
            allocation_ordinal=root.allocation_ordinal,
            requested_bytes=root.requested_bytes,
            charged_bytes=root.charged_bytes,
        )
        for root in layout.roots
    )


def _plan_compiled_views(
    layout: CompiledTaskLayout,
) -> tuple[PlanCompiledOutputView, ...]:
    return tuple(
        PlanCompiledOutputView(
            leaf_index=view.leaf_index,
            root_id=view.root_id,
            allocation_ordinal=view.allocation_ordinal,
            offset_bytes=view.offset_bytes,
        )
        for view in layout.output_views
    )


def _plan_representative_inputs(
    measurement: TaskMeasurement,
) -> tuple[PlanRepresentativeInput, ...]:
    return tuple(
        PlanRepresentativeInput(
            position=item.position,
            role=item.role.value,
            source=item.source,
            value_policy=item.value_policy,
            dtype=item.dtype,
            shape=item.shape,
            stride=item.stride,
            storage_offset=item.storage_offset,
            alias_group=item.alias_group,
            consumer_targets=item.consumer_targets,
        )
        for item in measurement.representative_inputs
    )


def _plan_allocation_timeline(
    measurement: TaskMeasurement,
) -> tuple[PlanAllocationEvent, ...]:
    return tuple(
        PlanAllocationEvent(
            allocation_ordinal=event.allocation_ordinal,
            operation=event.operation.value,
            requested_bytes=event.requested_bytes,
            charged_bytes=event.charged_bytes,
            output_leaf_indices=event.output_leaf_indices,
            output_view_offsets=event.output_view_offsets,
            reuses_ordinal=event.reuses_ordinal,
        )
        for event in measurement.allocation_trace
    )


def _plan_allocation_contract(
    measurement: TaskMeasurement,
) -> tuple[PlanAllocationABIStep, ...]:
    if measurement.allocation_contract is None:
        return ()
    return tuple(
        PlanAllocationABIStep(
            operation_index=step.operation_index,
            allocation_ordinal=step.allocation_ordinal,
            operation=step.operation.value,
            requested_bytes=step.requested_bytes,
            charged_bytes=step.charged_bytes,
            alignment_bytes=step.alignment_bytes,
            output_leaf_indices=step.output_leaf_indices,
            mutation_input_positions=step.mutation_input_positions,
            persistent_after_task=step.persistent_after_task,
        )
        for step in measurement.allocation_contract.steps
    )


def _logical_bytes(values: tuple[PlanObjectFootprint, ...]) -> int:
    return sum(item.logical_size_bytes for item in values)


def _footprint(
    object_spec: ObjectSpec, alias_spec: AliasGroupSpec
) -> PlanObjectFootprint:
    return PlanObjectFootprint(
        object_id=object_spec.object_id,
        alias_group_id=object_spec.alias_group_id,
        role=object_spec.role.value,
        logical_size_bytes=object_spec.size_bytes,
        allocation_size_bytes=alias_spec.size_bytes,
        offset_bytes=object_spec.offset_bytes,
    )


def _unique_allocation_bytes(values: tuple[PlanObjectFootprint, ...]) -> int:
    return sum(
        value.allocation_size_bytes
        for value in {item.alias_group_id: item for item in values}.values()
    )
