"""Declare framework collectives without compiling them or assigning timings."""

from typing import Any

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import TaskDefinition

from .catalog import CompiledTasks


def declare_collective(
    description: dict[str, Any],
    compiled: CompiledTasks,
    descriptions: dict[str, Any],
) -> str:
    key = "collective_" + digest(description)
    compiled.definitions[key] = TaskDefinition(key, key, None, description["operation"])
    descriptions[key] = description
    return key
