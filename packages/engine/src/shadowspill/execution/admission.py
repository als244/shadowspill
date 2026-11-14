"""Bind portable task contracts and placements to worker-local runtime handles."""

from dataclasses import dataclass, replace
from typing import Any

from shadowspill.artifacts.entrypoints import entrypoint
from shadowspill.artifacts.plan import Plan
from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.contracts import collective_measurement
from shadowspill.planning.admission.outputs import output_bindings_for_entrypoints
from shadowspill.preparation.lifecycle import LoadedPreparations
from shadowspill.profiling.records import measured_task
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.numerics import program_numerical_contracts
from shadowspill.runtime.plan.admission import (
    admit_caller_acquisition,
    admit_fixed_layout,
    admit_task,
    seal_fixed_layout,
)
from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.plan.common import (
    TaskMemoryEnvelope,
    TaskPublication,
    actions_by_task,
)
from shadowspill.runtime.plan.envelopes import task_memory_envelope
from shadowspill.runtime.plan.layout import project_runtime_fixed_layout
from shadowspill.task.allocations.events import TaskAllocationOperation
from shadowspill.task.entrypoints import execution_entrypoints
from shadowspill.task.profiles import TaskMeasurement

from .distributed.collectives import CollectiveTask


@dataclass
class BoundTask:
    task: Any
    entry: Any
    description: Any
    function: Any
    handle: int
    inputs: tuple[str, ...]
    outputs: Any
    publications: tuple[TaskPublication, ...]
    actions: Any
    host_inputs: tuple[tuple[int, str], ...]
    memory_envelope: TaskMemoryEnvelope


def prepare(
    plan: Plan,
    bridge: RuntimeBridge,
    *,
    shared_slice_bytes: int | None = None,
    groups: GroupBindings | None = None,
    preparations: LoadedPreparations | None = None,
) -> tuple[tuple[BoundTask, ...], int]:
    assert plan.memory_layout is not None
    ir = plan._costed_program
    tasks = ir.selected_tasks(plan.selections)
    entries = execution_entrypoints(
        tasks, tuple(entrypoint(e) for e in plan.program.bindings["entrypoints"])
    )
    aliases = {o.object_id: o.alias_group_id for o in ir.objects}
    outputs = output_bindings_for_entrypoints(tasks, entries, aliases)
    original_keys = {t.task_id: t.profile_id for t in plan.program.tasks}
    measurements: dict[str, TaskMeasurement] = {}
    allocation_bytes = {}
    for task in tasks:
        if not task.requires_entrypoint:
            continue
        key = original_keys[task.task_id]
        if key not in measurements:
            measurements[key] = (
                collective_measurement(plan.program, key, plan.communication_info)
                if key in plan.program.bindings.get("collective_tasks", {})
                else measured_task(
                    plan.profiled_program.task_profiles[key],
                    plan.program.bindings["manifests"][key],
                )
            )
        allocation_bytes.update(
            {
                (task.task_id, event.allocation_ordinal): event.charged_bytes
                for event in measurements[key].allocation_trace
                if event.operation is TaskAllocationOperation.ALLOCATE
                and not event.output_leaf_indices
            }
        )
    layout = project_runtime_fixed_layout(
        plan.memory_layout,
        ir,
        plan.schedule,
        anonymous_allocation_bytes=allocation_bytes,
    )
    if shared_slice_bytes is not None:
        if shared_slice_bytes < layout.slice_bytes:
            raise ValueError("shared arena cannot be smaller than the saved layout")
        layout = replace(layout, slice_bytes=shared_slice_bytes)
    admit_fixed_layout(bridge, layout)
    actions = actions_by_task(plan.schedule.actions)
    descriptions = {e["task_id"]: e for e in plan.program.bindings["entrypoints"]}
    functions, result = {}, []
    for task, entry in zip(tasks, entries, strict=True):
        function, description, envelope = None, {}, None
        bindings = outputs.get(task.task_id, ())
        publications = tuple(
            TaskPublication(b.alias_group_id, b.replacement)
            for b in bindings
            if bridge.objects.requires_storage(b.alias_group_id)
        )
        if task.requires_entrypoint:
            key = original_keys[task.task_id]
            collective = plan.program.bindings.get("collective_tasks", {}).get(key)
            if collective is not None:
                if groups is None:
                    raise ValueError("collective task requires bound logical groups")
                if key not in functions:
                    functions[key] = CollectiveTask(collective, groups)
            else:
                if key not in functions:
                    functions[key] = restore_executable(
                        plan.program.executables[key],
                        dependencies=plan.program.bindings.get(
                            "operator_dependencies", {}
                        ).get(key, ()),
                        triton_choices=plan.program.bindings.get(
                            "triton_choices", {}
                        ).get(key),
                        numerical_contracts=program_numerical_contracts(
                            plan.program, (key,)
                        ),
                    )
                    if preparations is not None:
                        functions[key] = preparations.wrap(
                            plan.program, key, functions[key]
                        )
                    elif plan.program.bindings.get("operator_preparation"):
                        raise ValueError(
                            "prepared operators must load before admission"
                        )
            measurement = measurements[key]
            function = functions[key]
            description = plan.program.bindings["task_inputs"][key]
            envelope = task_memory_envelope(
                measurement,
                retained_output_leaves=tuple(b.leaf_index for b in bindings),
            )
        input_aliases = tuple(dict.fromkeys(aliases[o] for o in task.inputs))
        task_actions = actions.get(task.task_id, ())
        handle = admit_task(
            bridge,
            task,
            input_aliases,
            task_actions,
            memory_envelope=envelope,
            publications=publications,
            trace_label=plan.program.bindings.get("task_names", {}).get(
                task.task_id, task.task_id
            ),
        )
        result.append(
            BoundTask(
                task,
                entry,
                description,
                function,
                handle,
                input_aliases,
                bindings,
                publications,
                task_actions,
                tuple(
                    (item["leaf_index"], item["name"])
                    for item in descriptions.get(task.task_id, {}).get(
                        "host_inputs", ()
                    )
                ),
                envelope or TaskMemoryEnvelope(),
            )
        )
    public = plan.program.bindings["public_outputs"]
    if plan.program.bindings["kind"] != "forward":
        public = [value for example in public for value in example]
    public_aliases = tuple(aliases[o] for o in public if o is not None)
    acquisition = admit_caller_acquisition(bridge, public_aliases)
    seal_fixed_layout(bridge)
    return tuple(result), acquisition
