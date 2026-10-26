"""Portable local optimizer components and their named tensor inputs."""

from dataclasses import asdict
from typing import Any

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.lowering.training.artifacts import TrainingObjects
from shadowspill.pytorch.lowering.training.tasks import optimizer_object_ids

from ..optimizer import BuiltOptimizer
from .catalog import CompiledTasks


def optimizer_templates(
    update: BuiltOptimizer | None,
    objects: TrainingObjects,
    compiled: CompiledTasks,
    *,
    owned_names: dict[str, str] | None = None,
    publications: dict[str, Any] | None = None,
    reductions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if update is None:
        return [], {}
    names = optimizer_object_ids(objects.gradients, objects.optimizer_objects)
    names.update(owned_names or {})
    components = []
    pending = {"gradient." + item["name"]: item for item in reductions or ()}
    for index, component in enumerate(update.captured.update_tasks):
        artifact = component.artifact
        assert isinstance(artifact, GraphArtifact)
        if component.output_names or component.metric_schema:
            raise CaptureError(
                "portable optimizer initialization/metrics need lowering"
            )
        key = compiled.key(artifact, ("optimizer",))
        contract = compiled.contract(artifact)
        components.append(
            {
                "task_id": f"optimizer_{index:04d}",
                "profile_key": key,
                "inputs": [
                    {"leaf_index": position, "object_id": names[name]}
                    for position, name in enumerate(component.binding_names)
                    if name in names
                ],
                "host_inputs": [
                    {"leaf_index": position, "name": name}
                    for position, name in enumerate(component.binding_names)
                    if name not in names
                ],
                "mutations": [
                    names[name] for name in component.mutation_names if name in names
                ],
                "replacements": [view.leaf_index for view in contract.output_views],
                "publications": [
                    {"name": name, **(publications or {})[name]}
                    for name in component.mutation_names
                    if name in (publications or {})
                ],
                "reductions": [
                    pending.pop(name)
                    for name in component.binding_names
                    if name in pending
                ],
            }
        )
    bindings = [
        {
            "name": binding.name,
            "role": binding.role.value,
            "parameter_name": binding.parameter_name,
            "object_id": names.get(binding.name),
            "mutable": binding.mutable,
            "shape": tuple(binding.tensor.shape),
            "stride": tuple(binding.tensor.stride()),
            "dtype": str(binding.tensor.dtype),
            "device": binding.tensor.device.type,
            "initialization": update.initialization.get(binding.name),
        }
        for binding in update.captured.bindings
    ]
    return components, {
        "optimizer_type": update.captured.optimizer_type,
        "optimizer_zero_lr_preserves_state": update.zero_lr_preserves_state,
        "optimizer_learning_rates": update.learning_rates,
        "optimizer_bindings": bindings,
        "optimizer_objects": [asdict(item) for item in objects.optimizer_objects],
    }
