"""Measure a resolved training template with its allocation declarations."""

from collections.abc import Mapping
from pathlib import Path

from shadowspill import (
    ProfiledProgram,
    Program,
    TaskProfileCache,
    apply_profiles,
    measure_program,
)
from shadowspill.task.profiling import ProfilingOptions

from .allocation_coverage import validate_allocation_coverage
from .templates import TemplateInventory, validate_template


def profile_template(
    inventory: TemplateInventory,
    template_id: str,
    program: Program,
    *,
    device: str = "cuda",
    rank_map: Mapping[int, int] | None = None,
    cache: TaskProfileCache | None = None,
) -> ProfiledProgram:
    """Measure one saved form with exact resolved policy and scenario provenance."""
    template = inventory.templates[template_id]
    validate_template(template, program)
    validate_allocation_coverage(template, program)
    profiles = measure_program(
        program,
        device=device,
        rank_map=rank_map,
        options=ProfilingOptions.from_dict(
            inventory.layouts.setup.config.preparation.profiling.to_dict()
        ),
        cache=cache,
        external_headroom_bytes=(
            inventory.layouts.setup.config.preparation.external_headroom_bytes
        ),
    )
    profiles.metadata["preparation_template"] = template.id
    return apply_profiles(program, profiles)


def run(directory: str | Path) -> None:
    from .job_phases import profile

    profile(directory)


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
