"""Reuse planner IR fixtures in offline phase tests without shipping an adapter."""

from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.program import Program, TaskDefinition
from shadowspill.ir import ResidencySpec, ShadowSpillProgram


def split_costed_program(
    captured: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    name: str = "",
    fixed_device_bytes: int,
) -> tuple[Program, Profiles]:
    """Separate old costs without inventing physical allocation evidence."""
    return (
        Program(
            captured.devices,
            captured.alias_groups,
            captured.objects,
            captured.tasks,
            tuple(
                TaskDefinition(item.profile_id, item.compatibility_digest)
                for item in captured.profiles
            ),
            initial_residency,
            final_residency,
            captured.task_alternative_groups,
            name=name,
        ),
        Profiles(
            tasks={
                item.profile_id: TaskProfile(
                    duration_seconds=item.runtime_ns / 1e9,
                    workspace_bytes=item.workspace_bytes,
                    compatibility_digest=item.compatibility_digest,
                    resource_phases=item.resource_phases,
                    overlap_costs=item.overlap_costs,
                    metadata={"source": "legacy costed IR; evidence not imported"},
                )
                for item in captured.profiles
            },
            fixed_device_bytes=fixed_device_bytes,
        ),
    )
