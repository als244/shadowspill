"""Emit the ordered training task DAG from bound stage variants."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

from shadowspill.ir import (
    MutationSpec,
    RecomputationGroup,
    RecomputationOption,
    ResourceKind,
    ResourceSpec,
    TaskSpec,
)
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTask,
)

from ...contracts import CaptureError
from ..catalog import TensorSlot
from ..profiles import TaskProfileCatalog
from .artifacts import (
    GradientBinding,
    OptimizerObjectBinding,
    PreparedStageVariant,
    TrainingObjects,
    TrainingTaskEntrypoint,
    TrainingTaskGraph,
)


class _TrainingTaskEmitter:
    """Emit the ordered task DAG from prepared stage-local variants."""

    def __init__(
        self,
        prepared: tuple[tuple[dict[str, PreparedStageVariant], ...], ...],
        metadata: tuple[str | None, ...],
        objects: TrainingObjects,
        optimizer: OptimizerCapture,
        profiles: TaskProfileCatalog,
        *,
        optimizer_phase: Literal["initial", "recurrent"],
        optimizer_ordering: Literal["stage_interleaved", "tail"],
        device_id: str,
    ) -> None:
        self.prepared = prepared
        self.metadata = metadata
        self.objects = objects
        self.optimizer = optimizer
        self.profiles = profiles
        self.optimizer_phase = optimizer_phase
        self.device_id = device_id

        self.tasks: list[TaskSpec] = []
        self.entrypoints: list[TrainingTaskEntrypoint] = []
        self.forward_ids: dict[tuple[int, int, str], str] = {}
        self.backward_ids: dict[tuple[int, int, str], str] = {}
        self.all_backward_ids: list[str] = []
        self.optimizer_task_ids: list[str] = []
        self.initial_writers: dict[str, tuple[str, ...]] = {}
        self.latest_contributors: dict[str, tuple[str, ...]] = {}
        self.object_producers: dict[str, list[str]] = {}

        self.has_lazy_optimizer_outputs = any(
            item.created_on_first_step for item in objects.optimizer_objects
        )
        self.interleave_optimizer = optimizer_ordering == "stage_interleaved" and not (
            optimizer_phase == "initial" and self.has_lazy_optimizer_outputs
        )
        self.optimizer_by_stage, self.unplaced_optimizer = (
            self._optimizer_stage_assignments()
        )

    def build(self) -> TrainingTaskGraph:
        completion_ids: tuple[str, ...] = ()
        for position, variants in enumerate(self.prepared):
            self._emit_forwards(position, variants, completion_ids)
            completion_ids = self._emit_backwards(position, variants)
        groups = self._recomputation_groups()
        self._emit_tail_optimizer()
        if not self.optimizer_task_ids:
            raise CaptureError("optimizer has no executable task components")
        return TrainingTaskGraph(
            tuple(self.tasks),
            tuple(self.entrypoints),
            groups,
            tuple(self.optimizer_task_ids),
        )

    def _emit_forwards(
        self,
        position: int,
        position_variants: tuple[dict[str, PreparedStageVariant], ...],
        prior_microbatch: tuple[str, ...],
    ) -> None:
        previous_ids = prior_microbatch
        metadata_digest = self.metadata[position]
        for stage_index, variants in enumerate(position_variants):
            current_ids: list[str] = []
            for variant, item in variants.items():
                task_id = f"task_{len(self.tasks):06d}"
                self.forward_ids[(position, stage_index, variant)] = task_id
                input_aliases = {
                    self.objects.catalog.alias_id(slot.object_id)
                    for slot in item.forward_inputs
                }
                dependencies = list(previous_ids)
                for slot in item.forward_inputs:
                    dependencies.extend(self.object_producers.get(slot.object_id, ()))
                task = TaskSpec(
                    task_id,
                    ResourceSpec(self.device_id, ResourceKind.COMPUTE),
                    self.profiles.profile_id(
                        item.pair.forward,
                        self.profiles.mutation_transition_bytes(
                            item.pair.forward,
                            metadata_digest,
                        ),
                        metadata_digest=metadata_digest,
                    ),
                    dependencies=_unique(dependencies),
                    inputs=_unique(slot.object_id for slot in item.forward_inputs),
                    outputs=_unique(
                        slot.object_id
                        for slot in item.forward_outputs
                        if self.objects.catalog.alias_id(slot.object_id)
                        not in input_aliases
                    ),
                    mutations=tuple(
                        MutationSpec(object_id)
                        for object_id in item.mutation_object_ids
                    ),
                    phase="forward",
                )
                self.tasks.append(task)
                self._record_producers(task)
                self.entrypoints.append(
                    TrainingTaskEntrypoint(
                        task_id,
                        "forward",
                        position,
                        variant,
                        item.pair.forward,
                        item.forward_inputs,
                        item.forward_outputs,
                        public_output_count=len(item.public_output_leaves),
                        public_output_leaves=item.public_output_leaves,
                        stage_index=stage_index,
                        replacement_output_leaves=(item.replacement_output_leaves),
                        storage_handoffs=item.forward_storage_handoffs,
                    )
                )
                current_ids.append(task_id)
            previous_ids = tuple(current_ids)

    def _emit_backwards(
        self,
        position: int,
        position_variants: tuple[dict[str, PreparedStageVariant], ...],
    ) -> tuple[str, ...]:
        downstream_ids: tuple[str, ...] = ()
        metadata_digest = self.metadata[position]
        for stage_index in reversed(range(len(position_variants))):
            variants = position_variants[stage_index]
            stage_task_ids = tuple(
                f"task_{len(self.tasks) + offset:06d}"
                for offset in range(len(variants))
            )
            first_destinations = {
                slot.object_id
                for item in variants.values()
                for slot in item.contributions
                if slot.object_id not in self.initial_writers
            }
            for (variant, item), task_id in zip(
                variants.items(), stage_task_ids, strict=True
            ):
                self._emit_backward_variant(
                    position,
                    stage_index,
                    variant,
                    task_id,
                    item,
                    downstream_ids,
                    first_destinations,
                    metadata_digest,
                )
            self._publish_contributors(variants, stage_task_ids)
            downstream_ids = stage_task_ids
            self._emit_interleaved_optimizer(
                position,
                stage_index,
                stage_task_ids,
            )
        return downstream_ids

    def _emit_backward_variant(
        self,
        position: int,
        stage_index: int,
        variant: str,
        task_id: str,
        item: PreparedStageVariant,
        downstream_ids: tuple[str, ...],
        first_destinations: set[str],
        metadata_digest: str | None,
    ) -> None:
        self.backward_ids[(position, stage_index, variant)] = task_id
        destinations = _unique(slot.object_id for slot in item.contributions)
        mutated = tuple(
            object_id
            for object_id in destinations
            if object_id not in first_destinations
        )
        dependencies = self._backward_dependencies(
            position,
            stage_index,
            variant,
            item,
            downstream_ids,
            mutated,
        )
        task = self._backward_task(
            task_id,
            item,
            destinations,
            mutated,
            first_destinations,
            dependencies,
            metadata_digest,
        )
        self.tasks.append(task)
        self._record_producers(task)
        self.entrypoints.append(
            TrainingTaskEntrypoint(
                task_id,
                "backward",
                position,
                variant,
                item.pair.backward,
                item.backward_inputs,
                (),
                gradient_output_slots=item.contributions,
                stage_index=stage_index,
                storage_handoffs=item.backward_storage_handoffs,
            )
        )
        self.all_backward_ids.append(task_id)

    def _backward_dependencies(
        self,
        position: int,
        stage_index: int,
        variant: str,
        item: PreparedStageVariant,
        downstream_ids: tuple[str, ...],
        mutated: tuple[str, ...],
    ) -> tuple[str, ...]:
        dependencies = list(
            dict.fromkeys(
                (
                    self.forward_ids[(position, stage_index, variant)],
                    *downstream_ids,
                )
            )
        )
        for slot in item.backward_inputs:
            object_id = slot.object_id
            dependencies.extend(self.object_producers.get(object_id, ()))
            dependencies.extend(self.initial_writers.get(object_id, ()))
            dependencies.extend(self.latest_contributors.get(object_id, ()))
        for object_id in mutated:
            dependencies.extend(self.initial_writers[object_id])
            dependencies.extend(self.latest_contributors[object_id])
        return _unique(dependencies)

    def _backward_task(
        self,
        task_id: str,
        item: PreparedStageVariant,
        destinations: tuple[str, ...],
        mutated: tuple[str, ...],
        first_destinations: set[str],
        dependencies: tuple[str, ...],
        metadata_digest: str | None,
    ) -> TaskSpec:
        mutation_bytes = sum(
            self.objects.catalog.object_size(object_id) for object_id in mutated
        )
        inputs = [slot.object_id for slot in item.backward_inputs]
        inputs.extend(mutated)
        input_aliases = {
            self.objects.catalog.alias_id(object_id) for object_id in inputs
        }
        return TaskSpec(
            task_id,
            ResourceSpec(self.device_id, ResourceKind.COMPUTE),
            self.profiles.profile_id(
                item.pair.backward,
                mutation_bytes,
                metadata_digest=metadata_digest,
            ),
            dependencies=dependencies,
            inputs=_unique(inputs),
            outputs=tuple(
                object_id
                for object_id in destinations
                if object_id in first_destinations
                and self.objects.catalog.alias_id(object_id) not in input_aliases
            ),
            mutations=tuple(MutationSpec(object_id) for object_id in mutated),
            phase="backward",
        )

    def _publish_contributors(
        self,
        variants: dict[str, PreparedStageVariant],
        stage_task_ids: tuple[str, ...],
    ) -> None:
        destinations = _unique(
            slot.object_id for item in variants.values() for slot in item.contributions
        )
        for object_id in destinations:
            self.initial_writers.setdefault(object_id, stage_task_ids)
            self.latest_contributors[object_id] = stage_task_ids

    def _emit_interleaved_optimizer(
        self,
        position: int,
        stage_index: int,
        dependencies: tuple[str, ...],
    ) -> None:
        if not self.interleave_optimizer or position != len(self.prepared) - 1:
            return
        components = self.optimizer_by_stage.get(stage_index, ())
        if components:
            self._append_optimizer(components, dependencies)

    def _emit_tail_optimizer(self) -> None:
        components = (
            self.optimizer.recurrent_tasks
            if not self.interleave_optimizer
            else self.unplaced_optimizer
        )
        if components or (
            self.optimizer_phase == "initial" and self.has_lazy_optimizer_outputs
        ):
            self._append_optimizer(
                components,
                tuple(self.all_backward_ids),
            )

    def _append_optimizer(
        self,
        components: tuple[OptimizerTask, ...],
        dependencies: tuple[str, ...],
    ) -> None:
        self.optimizer_task_ids.extend(
            _append_optimizer_tasks(
                self.tasks,
                self.entrypoints,
                self.optimizer,
                self.objects.optimizer_objects,
                self.objects.gradients,
                optimizer_phase=self.optimizer_phase,
                dependencies=dependencies,
                device_id=self.device_id,
                profile_id=self.profiles.profile_id,
                components=components,
                object_dependencies=_object_dependencies(
                    self.object_producers,
                    self.initial_writers,
                    self.latest_contributors,
                ),
            )
        )

    def _record_producers(self, task: TaskSpec) -> None:
        for object_id in task.outputs:
            self.object_producers.setdefault(object_id, []).append(task.task_id)

    def _optimizer_stage_assignments(
        self,
    ) -> tuple[dict[int, tuple[OptimizerTask, ...]], tuple[OptimizerTask, ...]]:
        stage_count = len(self.prepared[0])
        invalid = tuple(
            component.completion_stage_index
            for component in self.optimizer.recurrent_tasks
            if component.completion_stage_index is not None
            and not 0 <= component.completion_stage_index < stage_count
        )
        if invalid:
            raise CaptureError(
                f"optimizer component refers to an unknown training stage: {invalid}"
            )
        by_stage = {
            stage_index: tuple(
                component
                for component in self.optimizer.recurrent_tasks
                if component.completion_stage_index == stage_index
            )
            for stage_index in range(stage_count)
        }
        unplaced = tuple(
            component
            for component in self.optimizer.recurrent_tasks
            if component.completion_stage_index is None
        )
        return by_stage, unplaced

    def _recomputation_groups(self) -> tuple[RecomputationGroup, ...]:
        return tuple(
            RecomputationGroup(
                f"recompute_{position:04d}_{stage_index:04d}",
                tuple(
                    RecomputationOption(
                        variant,
                        (
                            self.forward_ids[(position, stage_index, variant)],
                            self.backward_ids[(position, stage_index, variant)],
                        ),
                        tuple(
                            dict.fromkeys(
                                self.objects.catalog.alias_id(object_id)
                                for object_id in self.prepared[position][stage_index][
                                    variant
                                ].saved_internal_object_ids
                            )
                        ),
                    )
                    for variant in self.prepared[position][stage_index]
                ),
            )
            for position, position_variants in enumerate(self.prepared)
            for stage_index in range(len(position_variants))
        )


def emit_training_tasks(
    prepared: tuple[tuple[dict[str, PreparedStageVariant], ...], ...],
    metadata: tuple[str | None, ...],
    objects: TrainingObjects,
    optimizer: OptimizerCapture,
    profiles: TaskProfileCatalog,
    *,
    optimizer_phase: Literal["initial", "recurrent"],
    optimizer_ordering: Literal["stage_interleaved", "tail"],
    device_id: str,
) -> TrainingTaskGraph:
    """Emit the complete accumulated task graph in execution order."""

    return _TrainingTaskEmitter(
        prepared,
        metadata,
        objects,
        optimizer,
        profiles,
        optimizer_phase=optimizer_phase,
        optimizer_ordering=optimizer_ordering,
        device_id=device_id,
    ).build()


def _append_optimizer_tasks(
    tasks: list[TaskSpec],
    entrypoints: list[TrainingTaskEntrypoint],
    optimizer: OptimizerCapture,
    optimizer_objects: tuple[OptimizerObjectBinding, ...],
    gradients: tuple[GradientBinding, ...],
    *,
    optimizer_phase: Literal["initial", "recurrent"],
    dependencies: tuple[str, ...],
    device_id: str,
    profile_id: Callable[..., str],
    components: tuple[OptimizerTask, ...] | None = None,
    object_dependencies: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Append dependency-closed optimizer components in semantic order."""

    appender = _OptimizerTaskAppender(
        tasks,
        entrypoints,
        optimizer,
        optimizer_objects,
        gradients,
        optimizer_phase=optimizer_phase,
        dependencies=dependencies,
        device_id=device_id,
        profile_id=profile_id,
        components=components,
        object_dependencies=object_dependencies,
    )
    return appender.append()


class _OptimizerTaskAppender:
    def __init__(
        self,
        tasks: list[TaskSpec],
        entrypoints: list[TrainingTaskEntrypoint],
        optimizer: OptimizerCapture,
        optimizer_objects: tuple[OptimizerObjectBinding, ...],
        gradients: tuple[GradientBinding, ...],
        *,
        optimizer_phase: Literal["initial", "recurrent"],
        dependencies: tuple[str, ...],
        device_id: str,
        profile_id: Callable[..., str],
        components: tuple[OptimizerTask, ...] | None,
        object_dependencies: dict[str, tuple[str, ...]] | None,
    ) -> None:
        if optimizer.recurrent is None:
            raise CaptureError("optimizer has no recurrent artifact")
        self.tasks = tasks
        self.entrypoints = entrypoints
        self.optimizer = optimizer
        self.optimizer_objects = optimizer_objects
        self.optimizer_phase = optimizer_phase
        self.dependencies = dependencies
        self.device_id = device_id
        self.profile_id = profile_id
        self.components = components
        self.object_dependencies = object_dependencies or {}
        self.object_by_name = _optimizer_object_ids(gradients, optimizer_objects)
        self.binding_by_name = {item.name: item for item in optimizer_objects}

    def append(self) -> tuple[str, ...]:
        components = self._selected_components()
        if not components:
            raise CaptureError("optimizer has no executable task components")
        task_ids: list[str] = []
        preceding = self.dependencies
        for component in components:
            task = self._task(component, preceding)
            self.tasks.append(task)
            self.entrypoints.append(self._entrypoint(component, task.task_id))
            task_ids.append(task.task_id)
            preceding = _unique((*self.dependencies, task.task_id))
        return tuple(task_ids)

    def _selected_components(self) -> tuple[OptimizerTask, ...]:
        lazy_outputs = any(
            item.created_on_first_step for item in self.optimizer_objects
        )
        if self.optimizer_phase == "initial" and lazy_outputs:
            if self.optimizer.initial is None:
                raise CaptureError(
                    "lazy optimizer state has no profiled first-step artifact"
                )
            return (
                OptimizerTask(
                    self.optimizer.initial,
                    tuple(binding.name for binding in self.optimizer.bindings),
                    self.optimizer.mutation_names,
                ),
            )
        return (
            self.optimizer.recurrent_tasks
            if self.components is None
            else self.components
        )

    def _task(
        self,
        component: OptimizerTask,
        preceding: tuple[str, ...],
    ) -> TaskSpec:
        task_id = f"task_{len(self.tasks):06d}"
        objects = self._component_objects(component)
        outputs = self._component_outputs(component)
        dependencies = _unique(
            (
                *preceding,
                *(
                    dependency
                    for object_id in objects
                    for dependency in self.object_dependencies.get(object_id, ())
                ),
            )
        )
        return TaskSpec(
            task_id,
            ResourceSpec(self.device_id, ResourceKind.COMPUTE),
            self.profile_id(component.artifact),
            dependencies=dependencies,
            inputs=tuple(
                object_id for object_id in objects if object_id not in outputs
            ),
            outputs=outputs,
            mutations=tuple(
                MutationSpec(self.object_by_name[name])
                for name in component.mutation_names
                if name in self.object_by_name
                and self.object_by_name[name] not in outputs
            ),
            phase="optimizer",
        )

    def _component_objects(self, component: OptimizerTask) -> tuple[str, ...]:
        return tuple(
            self.object_by_name[name]
            for name in component.binding_names
            if name in self.object_by_name
        )

    def _component_outputs(self, component: OptimizerTask) -> tuple[str, ...]:
        if self.optimizer_phase != "initial":
            return ()
        return tuple(
            self.object_by_name[name]
            for name in component.mutation_names
            if name in self.binding_by_name
            and self.binding_by_name[name].created_on_first_step
        )

    def _entrypoint(
        self,
        component: OptimizerTask,
        task_id: str,
    ) -> TrainingTaskEntrypoint:
        artifact = component.artifact
        output_names = (
            artifact.profile_output_names
            if isinstance(artifact, OpaqueOptimizerArtifact)
            else ()
        )
        output_slots = tuple(
            TensorSlot(leaf_index, self.object_by_name[name])
            for leaf_index, name in enumerate(output_names)
            if name in self.object_by_name
        )
        return TrainingTaskEntrypoint(
            task_id,
            "optimizer",
            None,
            None,
            artifact,
            (),
            output_slots,
            optimizer_binding_names=component.binding_names,
            optimizer_output_names=tuple(
                name for name in output_names if name in self.object_by_name
            ),
        )


def _optimizer_object_ids(
    gradients: tuple[GradientBinding, ...],
    optimizer_objects: tuple[OptimizerObjectBinding, ...],
) -> dict[str, str]:
    return {
        **{item.parameter_name: item.parameter_object_id for item in gradients},
        **{
            f"gradient.{item.parameter_name}": item.gradient_object_id
            for item in gradients
        },
        **{item.name: item.object_id for item in optimizer_objects},
    }


def _object_dependencies(
    object_producers: dict[str, list[str]],
    initial_writers: dict[str, tuple[str, ...]],
    latest_contributors: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Snapshot direct producer edges required by later state consumers."""

    object_ids = set(object_producers) | set(initial_writers) | set(latest_contributors)
    return {
        object_id: _unique(
            (
                *object_producers.get(object_id, ()),
                *initial_writers.get(object_id, ()),
                *latest_contributors.get(object_id, ()),
            )
        )
        for object_id in object_ids
    }


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = ["emit_training_tasks"]
