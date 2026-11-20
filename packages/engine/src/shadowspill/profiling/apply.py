"""CPU-only application of supplied profiles; never measures or fills guesses."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.artifacts._values import freeze
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.program import Program
from shadowspill.communication.task_costs import task_collectives


def apply_profiles(program: Program, profiles: Profiles) -> ProfiledProgram:
    """Attach profiles to a Program, ready for planning.

    Every task form needs a workspace size, and every one but a collective
    task a duration. A workspace size
    missing from ``profiles`` comes from what the build recorded, which a
    supplied value may exceed but not undercut; nothing else missing is
    filled in. A collective task takes its timing,
    and may take its workspace, from `CommunicationInfo` during planning.
    Estimated memory leaves a Plan simulation-only. It runs on the CPU and
    needs no process group.

    Parameters
    ----------
    program : Program
        The Program to profile.
    profiles : Profiles
        Profiles for the Program's setup, such as `measure_program` returns,
        with any overlay merged.

    Returns
    -------
    ProfiledProgram
        The Program with each task form's profile.

    Raises
    ------
    ValueError
        If the profiles' setup differs from the Program's, a profile is for
        another implementation, a duration or memory size is missing, a
        supplied size is below the recorded one, or a collective task has a
        timing.
    """
    # TaskProfile records are immutable. ProfiledProgram snapshots the editable
    # collection/setup/metadata below; serializing every task here needlessly
    # copies and freezes potentially large allocation traces a second time.
    if program.setup and freeze(program.setup) != freeze(profiles.setup):
        raise ValueError("profile setup differs from the Program target/build setup")
    missing: list[str] = []
    applied = {}
    communication = task_collectives(program)
    for definition in program.task_definitions:
        key = definition.profile_key
        record = profiles.tasks.get(key)
        if key in communication:
            record = record or TaskProfile()
            if (
                record.duration_seconds is not None
                or record.resource_phases
                or record.overlap_costs
            ):
                raise ValueError(
                    f"{key}: supply collective timing through CommunicationInfo"
                )
        elif record is None or record.duration_seconds is None:
            missing.append(f"{key}: duration_seconds")
        if record is None:
            if definition.workspace_bytes is None:
                missing.append(f"{key}: workspace_bytes")
            continue
        if record.compatibility_digest not in (None, definition.compatibility_digest):
            raise ValueError(
                f"{key}: task implementation/layout does not match profile"
            )
        workspace = record.workspace_bytes
        known_workspace = (
            program.bindings.get("compiler_memory", {})
            .get(key, {})
            .get("known_workspace_bytes", 0)
        )
        metadata = record.metadata
        if workspace is None:
            workspace = definition.workspace_bytes
            if workspace is not None:
                metadata = dict(metadata, memory_kind="static", memory_source="program")
        elif workspace < max(definition.workspace_bytes or 0, known_workspace):
            raise ValueError(
                f"{key}: workspace is smaller than compiler-visible memory"
            )
        if workspace is None:
            if key in communication:
                metadata = dict(metadata, memory_source="communication_info")
            else:
                missing.append(f"{key}: workspace_bytes")
        if (
            workspace == record.workspace_bytes
            and definition.compatibility_digest == record.compatibility_digest
            and metadata is record.metadata
        ):
            applied[key] = record
        else:
            applied[key] = replace(
                record,
                workspace_bytes=workspace,
                compatibility_digest=definition.compatibility_digest,
                metadata=metadata,
            )
    overhead = profiles.fixed_device_bytes
    metadata = dict(profiles.metadata)
    if overhead is None:
        missing.append("fixed_device_bytes (runtime/provider overhead)")
    if missing:
        raise ValueError("incomplete profiles:\n  " + "\n  ".join(missing))
    assert overhead is not None
    result = ProfiledProgram(program, applied, profiles.setup, overhead, metadata)
    result._compute_profiles()  # Validate timings without inventing collective costs.
    return result
