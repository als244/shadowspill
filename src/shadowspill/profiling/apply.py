"""CPU-only application of supplied profiles; never measures or fills guesses."""

from __future__ import annotations

from dataclasses import replace

from shadowspill.artifacts._values import freeze
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.profiles import Profiles
from shadowspill.artifacts.program import Program


def apply_profiles(program: Program, profiles: Profiles) -> ProfiledProgram:
    """Require a duration and complete workspace size for every task form.

    Static workspace/overhead may come from build. A supplied workspace can
    exceed a static bound but cannot erase compiler-visible memory. Estimates
    remain explicit and do not qualify an artifact for execution.
    """
    profiles = Profiles.from_dict(profiles.to_dict())
    if program.setup and freeze(program.setup) != freeze(profiles.setup):
        raise ValueError("profile setup differs from the Program target/build setup")
    missing: list[str] = []
    applied = {}
    for definition in program.task_definitions:
        key = definition.profile_key
        record = profiles.tasks.get(key)
        if record is None or record.duration_seconds is None:
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
        metadata = dict(record.metadata)
        if workspace is None:
            workspace = definition.workspace_bytes
            if workspace is not None:
                metadata["memory_kind"] = "static"
                metadata["memory_source"] = "program"
        elif (
            definition.workspace_bytes is not None
            and workspace < definition.workspace_bytes
        ):
            raise ValueError(
                f"{key}: workspace is smaller than compiler-visible memory"
            )
        if workspace is None:
            missing.append(f"{key}: workspace_bytes")
        applied[key] = replace(
            record,
            workspace_bytes=workspace,
            compatibility_digest=definition.compatibility_digest,
            metadata=metadata,
        )
    overhead = profiles.fixed_device_bytes
    metadata = dict(profiles.metadata)
    if overhead is None:
        overhead = program.fixed_device_bytes
        if overhead is not None:
            metadata["memory_kind"] = "static"
            metadata["memory_source"] = "program"
    elif (
        program.fixed_device_bytes is not None and overhead < program.fixed_device_bytes
    ):
        raise ValueError(
            "fixed device overhead is smaller than the Program requirement"
        )
    if overhead is None:
        missing.append("fixed_device_bytes (runtime/provider overhead)")
    if missing:
        raise ValueError("incomplete profiles:\n  " + "\n  ".join(missing))
    assert overhead is not None
    result = ProfiledProgram(program, applied, profiles.setup, overhead, metadata)
    result._costed_ir()  # Validate resource phase timing against the supplied duration.
    return result
