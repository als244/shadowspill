"""Omit only parameter storage proven unchanged by every admitted Plan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import registration_name, shared_model_aliases

if TYPE_CHECKING:
    from shadowspill.execution import Executor


def validate_identity(identity: str | None) -> None:
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ValueError("frozen_state_id must be a nonempty string")


def _retained_aliases(program: Program) -> set[str]:
    aliases = {item.object_id: item.alias_group_id for item in program.objects}
    retained = {
        aliases[value]
        for task in program.tasks
        for value in (*task.outputs, *(m.object_id for m in task.mutations))
    }
    for item in program.bindings["registrations"]:
        if type(item.get("trainable")) is not bool:
            raise ValueError("rebuild Program with parameter trainability metadata")
        if not item["parameter"] or item["trainable"]:
            retained.add(aliases[item["object_id"]])
    return retained


def omitted_parameters(executor: Executor) -> set[str]:
    program = executor.plan.program
    aliases = {item.object_id: item.alias_group_id for item in program.objects}
    retained = _retained_aliases(program)
    for plan in executor.plans.values():
        shared = shared_model_aliases(program, plan.program)
        retained.update(
            shared[alias]
            for alias in _retained_aliases(plan.program)
            if alias in shared
        )
    return {
        registration_name(item)
        for item in program.bindings["registrations"]
        if aliases[item["object_id"]] not in retained
    }


def validate_restore(
    executor: Executor, manifest: Mapping[str, Any], identity: str | None
) -> set[str]:
    validate_identity(identity)
    if manifest.get("frozen_state_id") != identity:
        raise ValueError(
            "checkpoint frozen_state_id differs; supply the same frozen base"
        )
    omitted = set() if identity is None else omitted_parameters(executor)
    if manifest.get("omitted_parameters", []) != sorted(omitted):
        raise ValueError("checkpoint omitted parameter inventory differs")
    if omitted.intersection(entry["name"] for entry in manifest["tensors"]):
        raise ValueError("checkpoint contains a parameter declared omitted")
    return omitted
