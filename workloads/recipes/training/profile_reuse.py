"""Resume measurements after checking the current target and saved artifact."""

from shadowspill import Profiles, apply_profiles


def check_target(program):
    import torch

    from shadowspill.artifacts._values import plain
    from shadowspill.artifacts.device import require_compiled_device
    from shadowspill.pytorch.numerics import (
        program_numerical_contracts,
        validate_numerical_settings,
    )
    from shadowspill.pytorch.toolchain import build_setup

    require_compiled_device(program, 0)
    if plain(program.setup) != build_setup(torch.device("cuda", 0)):
        raise ValueError(
            "measurement device/toolchain differs from the finalized Program"
        )
    validate_numerical_settings(program_numerical_contracts(program))


def completed(path, program, options, group):
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
        result = apply_profiles(program, profiles)
    except Exception as failure:
        error = str(failure)
    errors = [value for value in group.gather(error) if value is not None]
    if errors:
        raise ValueError(f"cannot reuse completed profile {path}: {'; '.join(errors)}")
    return result
