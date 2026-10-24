"""Static training storage and saved stage templates, independent of measurement."""

from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch.utils._pytree import treespec_dumps

from shadowspill.artifacts.program import Program
from shadowspill.ir import DeviceSpec, MemoryLocation, Persistence, ResidencySpec
from shadowspill.parallelism import Parallelism
from shadowspill.planning.training.gradients import complete_gradients
from shadowspill.planning.training.identity import canonicalize_tasks
from shadowspill.planning.training.optimizer import place_optimizer
from shadowspill.planning.training.tasks import assemble_training
from shadowspill.pytorch.lowering.training.bindings import bind_training_boundaries
from shadowspill.pytorch.lowering.training.objects import register_training_objects
from shadowspill.pytorch.lowering.training.residency import derive_training_residency
from shadowspill.pytorch.optimizer.staging import training_parameters_with_gradients
from shadowspill.step.ordering import StepDataOrdering

from ..compilation import BuiltTask
from ..gradient_ops import GradientOperations
from ..metadata import model_tensor_geometry, root_sources, signature_record
from ..optimizer import BuiltOptimizer
from ..training_capture import TrainingCapture
from .catalog import CompiledTasks
from .controls import register_device_controls
from .gradients import gradient_templates
from .optimizer import optimizer_templates
from .training_templates import stage_templates


def lower_training(
    captured: TrainingCapture,
    compiled: dict[str, BuiltTask],
    *,
    setup: dict[str, Any],
    gradient_dtype: torch.dtype | None,
    update: BuiltOptimizer | None = None,
    gradient_operations: dict[str, GradientOperations] | None = None,
    parallelism: Parallelism | None = None,
) -> Program:
    captures = captured.examples
    trained = training_parameters_with_gradients(
        captures, dict(captured.model.named_parameters())
    )
    geometry = CompiledTasks(compiled, setup)
    mode = detect_fake_mode(captures[0].training.exported.flat_inputs)
    assert mode is not None
    with mode:
        objects = register_training_objects(
            captured.model,
            captures,
            None if update is None else update.captured,
            device_id="cuda_0",
            gradient_dtypes={
                name: gradient_dtype or parameter.dtype
                for name, parameter in captured.model.named_parameters()
                if name in trained
            },
        )
        objects = register_device_controls(objects, update)
        boundaries = bind_training_boundaries(
            captures, objects, geometry, (None,) * len(captures)
        )
        templates = stage_templates(captures, objects, boundaries, geometry)
        updates, optimizer_bindings = optimizer_templates(update, objects, geometry)
        completions, gradient_bindings = (
            ([], {})
            if gradient_operations is None
            else gradient_templates(gradient_operations, objects, geometry)
        )
    alias_by_object = {
        item.object_id: item.alias_group_id for item in objects.catalog.objects()
    }
    # A canonical topological inventory, not a chosen execution traversal.
    # Planning rebinds these saved templates without a framework/compiler import.
    sequence = assemble_training(
        templates, alias_by_object, StepDataOrdering.depth_first(len(captures))
    )
    sequence = complete_gradients(
        sequence, completions, accumulate=False, device_id="cuda_0"
    )
    sequence = place_optimizer(
        sequence, updates, ordering="tail", aliases=alias_by_object, device_id="cuda_0"
    )
    task_names = canonicalize_tasks(sequence)
    initial, final = derive_training_residency(
        objects, boundaries, tuple(sequence.tasks)
    )
    gradients = {item.gradient_object_id for item in objects.gradients}
    final_by_alias = {entry.alias_group_id: entry for entry in final}
    if update is None:
        stored_gradients = (
            gradients
            if gradient_operations is None
            else {
                item["gradient_object_id"]
                for item in gradient_bindings["persistent_gradients"]
            }
        )
        for object_id in stored_gradients:
            alias = alias_by_object[object_id]
            final_by_alias[alias] = ResidencySpec(alias, MemoryLocation.SPILL)
    sources: dict[str, list[str]] = defaultdict(list)
    for item in objects.registrations:
        sources[alias_by_object[item.object_id]].append(item.name)
    for gradient in objects.gradients:
        sources[alias_by_object[gradient.gradient_object_id]].append(
            f"{gradient.parameter_name}.grad"
        )
    for optimizer_object in objects.optimizer_objects:
        sources[alias_by_object[optimizer_object.object_id]].append(
            optimizer_object.name
        )
    for gradient in gradient_bindings.get("persistent_gradients", ()):
        sources[alias_by_object[gradient["gradient_object_id"]]].append(
            f"{gradient['parameter_name']}.grad (reduced shard)"
        )
    roots = root_sources(captures[0].training.exported, captured.constant_values)
    for example, slots in enumerate(objects.root_slots):
        for slot in slots:
            sources[alias_by_object[slot.object_id]].append(
                f"example[{example}].{roots[slot.leaf_index]['name']}"
            )
    schema = captures[0].training.objective_schema
    return Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=objects.catalog.alias_groups(),
        objects=tuple(
            replace(item, persistence=Persistence.RUN)
            if item.object_id in gradients
            and update is None
            and gradient_operations is None
            else item
            for item in objects.catalog.objects()
        ),
        tasks=tuple(sequence.tasks),
        task_alternative_groups=tuple(sequence.groups),
        task_definitions=tuple(geometry.definitions.values()),
        initial_residency=initial,
        final_residency=tuple(final_by_alias.values()),
        name=type(captured.model).__name__,
        setup=setup,
        parallelism=parallelism,
        sources=sources,
        executables=geometry.executables,
        bindings={
            **geometry.bindings(),
            **optimizer_bindings,
            **gradient_bindings,
            **(
                {
                    "gradient_templates": completions,
                    "distributed_training_lowered": "gradients",
                }
                if gradient_operations is not None
                else {}
            ),
            "kind": "forward_backward" if update is None else "step",
            "optimizer_templates": updates,
            "task_names": task_names,
            "num_examples": len(captures),
            "training_templates": templates,
            "entrypoints": sequence.entrypoints,
            "transient_outputs": sequence.transient_outputs,
            "gradients": [asdict(item) for item in objects.gradients],
            "registrations": [asdict(item) for item in objects.registrations],
            "model_tensor_geometry": model_tensor_geometry(
                captured.model, objects.registrations
            ),
            "root_inputs": [
                [asdict(slot) for slot in slots] for slots in objects.root_slots
            ],
            "root_sources": roots,
            "input_signature": signature_record(captured.input_signature),
            "public_outputs": [
                boundaries.public_outputs[i] for i in range(len(captures))
            ],
            "fixed_inputs": [
                {"object_id": item.object_id, "fill": 1}
                for item in boundaries.fixed_tensors.values()
            ],
            "objective_schema": {
                "tree": treespec_dumps(schema.metric_tree_spec),
                "leaf_count": schema.metric_leaf_count,
                "tensor_positions": schema.tensor_metric_positions,
                "static_leaves": schema.static_metric_leaves,
            },
            "ordering_constraints": _ordering_constraints(captured),
        },
    )


def _ordering_constraints(captured: TrainingCapture) -> list[str]:
    reasons = set()
    for stage in captured.examples[0].stages:
        if stage.example.stage.mutations:
            reasons.add("model mutations require the original example order")
        for node in stage.example.stage.graph_module.graph.nodes:
            if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
                reasons.add("RNG effects require the original example order")
    return sorted(reasons)
