"""Current target checks shared by saved build and measurement reuse."""

from shadowspill import Program


def check_target(program: Program) -> None:
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
