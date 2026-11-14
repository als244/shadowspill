"""Resume measurements after checking the current target and saved artifact."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shadowspill import ProfiledProgram, Profiles, Program, apply_profiles
from shadowspill.orchestration.launch.workers import Workers
from shadowspill.orchestration.profiling import profiling_options

from .target import check_target


def completed(
    path: Path, program: Program, options: Mapping[str, Any], group: Workers
) -> ProfiledProgram | None:
    if options.get("refresh_profiles", False):
        return None
    # Every collective participant must choose the same branch before any task
    # invokes provider communication. Missing files cause a fresh measurement.
    if not all(group.gather(path.is_file())):
        return None
    result, error = None, None
    try:
        check_target(program)
        profiles = Profiles.load(path)
        if profiles.metadata.get("memory_kind") != "measured":
            raise ValueError("completed profile has no measured memory evidence")
        if profiles.metadata.get("external_headroom_bytes") != (
            options["external_headroom_mib"] << 20
        ):
            raise ValueError("completed profile has different worker headroom")
        if (
            profiles.metadata.get("profiling_options")
            == profiling_options(options).to_dict()
        ):
            result = apply_profiles(program, profiles)
    except Exception as failure:
        error = str(failure)
    errors = [value for value in group.gather(error) if value is not None]
    if errors:
        raise ValueError(f"cannot reuse completed profile {path}: {'; '.join(errors)}")
    if not all(group.gather(result is not None)):
        return None
    return result
