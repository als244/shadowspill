"""Attach portable provider descriptions and bind task identity to their setup."""

from collections.abc import Mapping
from dataclasses import replace

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import Program
from shadowspill.resources import Resource
from shadowspill.resources.factories import sources


def attach_resources(program: Program, resources: Mapping[str, Resource]) -> Program:
    if not resources:
        return program
    identities = sources(resources)
    description = {name: value.to_dict() for name, value in resources.items()}
    return replace(
        program,
        resources=resources,
        bindings={**program.bindings, "resource_sources": identities},
        task_definitions=tuple(
            replace(
                task,
                compatibility_digest=digest(
                    {
                        "task": task.compatibility_digest,
                        "resources": description,
                        "sources": identities,
                    }
                ),
            )
            for task in program.task_definitions
        ),
    )
