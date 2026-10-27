"""Portable local optimizer components and their named tensor inputs."""

from dataclasses import asdict
from typing import Any

from shadowspill.build.optimizer import BuiltOptimizer
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from ..catalog import CompiledTasks
from .records import GradientBinding, OptimizerObjectBinding, TrainingObjects


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


def optimizer_object_ids(
    gradients: tuple[GradientBinding, ...],
    optimizer_objects: tuple[OptimizerObjectBinding, ...],
) -> dict[str, str]:
    """The object behind each name an optimizer task binds.

    A parameter's name is its weights, unless the optimizer holds a master copy
    of them: then the master is one of the optimizer's own objects, found under
    the same name, and the weights are ``compute.<name>``, which the update
    writes from it.
    """

    return {
        **{item.parameter_name: item.parameter_object_id for item in gradients},
        **{
            f"compute.{item.parameter_name}": item.parameter_object_id
            for item in gradients
        },
        **{
            f"gradient.{item.parameter_name}": item.gradient_object_id
            for item in gradients
        },
        **{item.name: item.object_id for item in optimizer_objects},
    }
