"""Share identical named resources across admitted training/evaluation Plans."""

from collections.abc import Iterable

from shadowspill.artifacts.program import Program

from .description import Resource


def resource_inventory(
    programs: Iterable[Program],
) -> tuple[dict[str, Resource], dict[str, str]]:
    resources: dict[str, Resource] = {}
    sources: dict[str, str] = {}
    for program in programs:
        for name, description in program.resources.items():
            source = program.bindings["resource_sources"][name]
            if name in resources and (
                resources[name] != description or sources[name] != source
            ):
                raise ValueError(
                    f"Plans have conflicting descriptions for resource {name!r}"
                )
            resources[name], sources[name] = description, source
    return resources, sources
