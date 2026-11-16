"""Check saved capture bindings without importing or recapturing model code."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.examples import input_groups
from shadowspill.artifacts.program import Program

if TYPE_CHECKING:
    from shadowspill.training.definitions.registry import ResolvedOptimizer

    from .binding import OperationBinding


def _reads(program: Program) -> tuple[set[str], set[str]]:
    """Actual graph consumers, not merely optimizer groups or input placeholders."""
    tasks = {task.task_id: task for task in program.tasks}
    objects: set[str] = set()
    hosts: set[str] = set()
    for entry in program.bindings["entrypoints"]:
        task = tasks[entry["task_id"]]
        if task.profile_id in program.bindings.get("collective_tasks", {}):
            # Lowered weight/gradient collectives have a native buffer ABI, not
            # an FX graph or captured input provenance. They cannot establish
            # that a caller coefficient is consumed by the model/optimizer.
            continue
        provenance = program.bindings["task_inputs"][task.profile_id]["provenance"]
        for kind, name, target in (
            ("input_slots", "object_id", objects),
            ("host_inputs", "name", hosts),
        ):
            for slot in entry.get(kind, ()):
                if provenance[slot["leaf_index"]]["consumer_targets"]:
                    target.add(slot[name])
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    return {aliases[obj] for obj in objects}, hosts


def validate_optimizer_program(program: Program, optimizer: ResolvedOptimizer) -> None:
    bindings = {item["name"]: item for item in program.bindings["optimizer_bindings"]}
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    objects, hosts = _reads(program)
    for control, names in optimizer.definition.control_bindings.items():
        dtype = {"float": "torch.float32", "int": "torch.int64", "bool": "torch.bool"}[
            optimizer.definition.controls.fields[control].dtype
        ]
        for name in names:
            item = bindings.get(name)
            if (
                item is None
                or item["role"] != "hyperparameter"
                or item["mutable"]
                or prod(item["shape"]) != 1
                or item["dtype"] != dtype
            ):
                raise ValueError(
                    f"optimizer control lacks a read-only scalar binding: {name}"
                )
            object_id = item["object_id"]
            if (object_id is None and name not in hosts) or (
                object_id is not None and aliases[object_id] not in objects
            ):
                raise ValueError(
                    f"optimizer control is not consumed by captured tasks: {name}"
                )
    scale = program.bindings.get("gradient_scale_control")
    if scale is None or scale not in hosts:
        raise ValueError("optimizer operation requires a captured final gradient scale")


def validate_operation_program(
    program: Program, binding: OperationBinding, *, output_only: bool = False
) -> None:
    expected_kind = "forward" if output_only else binding.contract.key.operation
    if program.bindings.get("kind") != expected_kind:
        raise ValueError("captured program kind differs from the requested operation")
    signatures = program.bindings.get(
        "input_signatures", (program.bindings["input_signature"],)
    )
    sources = program.bindings.get(
        "example_root_sources", (program.bindings["root_sources"],)
    )
    roots = input_groups(program.bindings)
    controls = binding.contract.controls
    fields: list[tuple[tuple[int, ...], str]] = [
        (binding.inputs.shapes[path], "torch." + binding.inputs.dtypes[path])
        for path in sorted(binding.inputs.shapes)
    ]
    first_control = len(fields)
    fields.extend(
        (
            (),
            {"float": "torch.float32", "int": "torch.int64", "bool": "torch.bool"}[
                field.dtype
            ],
        )
        for field in controls.fields.values()
    )
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    reads, _ = _reads(program)
    input_aliases: set[str] = set()
    for signature, source, slots in zip(signatures, sources, roots, strict=True):
        leaves = signature["leaves"]
        if len(leaves) != len(fields) or any(
            leaf["kind"] != "tensor"
            or tuple(leaf["shape"]) != shape
            or leaf["dtype"] != dtype
            for leaf, (shape, dtype) in zip(leaves, fields, strict=True)
        ):
            raise ValueError("captured input ABI differs from the prepared operation")
        users = {
            s["root_index"]: s["user_leaf"] for s in source if s["kind"] == "USER_INPUT"
        }
        by_leaf = {
            users[slot["leaf_index"]]: aliases[slot["object_id"]]
            for slot in slots
            if slot["leaf_index"] in users
        }
        input_aliases.update(by_leaf.values())
        for index, control in enumerate(controls.fields, first_control):
            if by_leaf.get(index) not in reads:
                raise ValueError(
                    f"objective control is not consumed by captured tasks: {control}"
                )
    _validate_mutations(program, binding, input_aliases, output_only=output_only)


def _validate_mutations(
    program: Program,
    binding: OperationBinding,
    input_aliases: set[str],
    *,
    output_only: bool,
) -> None:
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    registrations: dict[str, list[Any]] = {}
    for item in program.bindings["registrations"]:
        registrations.setdefault(aliases[item["object_id"]], []).append(item)
    declared = set(binding.model.definition.mutable_buffers)
    available = {
        item["name"]
        for values in registrations.values()
        for item in values
        if not item["parameter"]
    }
    if declared - available:
        raise ValueError("declared mutable buffers are absent from the captured model")
    for task in program.tasks:
        for mutation in task.mutations:
            alias = aliases[mutation.object_id]
            if alias in input_aliases:
                raise ValueError("operation mutates a submitted input or coefficient")
            if task.phase != "forward" or alias not in registrations:
                continue
            entries = registrations[alias]
            if output_only or binding.contract.key.mode == "evaluation":
                raise ValueError(
                    "output-only and evaluation must preserve model buffers "
                    "and parameters"
                )
            if any(item["parameter"] for item in entries) or not any(
                item["name"] in declared for item in entries
            ):
                raise ValueError("forward mutates undeclared model state")
