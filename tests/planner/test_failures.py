from __future__ import annotations

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryLocation,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    PressureFitInfeasibleError,
    pressurefit,
    validate_schedule_feasibility,
)

from ._examples import config


def test_missing_initial_residency_uses_semantic_diagnostic() -> None:
    resource = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(AliasGroupSpec("input_storage", "cuda_0", 61),),
        objects=(ObjectSpec("input", "input_storage", 0, 61),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(TaskSpec("consume", resource, "profile", inputs=("input",)),),
    )

    with pytest.raises(ValueError, match="has no initial residency"):
        pressurefit(program, initial_residency=(), config=config(122))


def test_required_task_geometry_reports_the_exact_capacity_constraint() -> None:
    resource = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("left_storage", "cuda_0", 61),
            AliasGroupSpec("right_storage", "cuda_0", 61),
            AliasGroupSpec("output_storage", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("left", "left_storage", 0, 61),
            ObjectSpec("right", "right_storage", 0, 61),
            ObjectSpec("output", "output_storage", 0, 61),
        ),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "impossible",
                resource,
                "profile",
                inputs=("left", "right"),
                outputs=("output",),
            ),
        ),
    )

    initial_residency = (
        ResidencySpec("left_storage", MemoryLocation.DEVICE),
        ResidencySpec("right_storage", MemoryLocation.DEVICE),
    )
    simulation_config = config(122)

    with pytest.raises(PressureFitInfeasibleError) as preflight:
        validate_schedule_feasibility(
            program,
            initial_residency=initial_residency,
            config=simulation_config,
        )

    error = preflight.value
    assert error.kind == "required_capacity"
    assert error.boundary_task_id == "impossible"
    assert error.required_bytes == 183
    assert error.capacity_bytes == 122

    # Direct framework-neutral PressureFit callers retain the same check as a
    # defensive invariant even though public planning runs preflight first.
    with pytest.raises(PressureFitInfeasibleError) as pressurefit_failure:
        pressurefit(
            program,
            initial_residency=initial_residency,
            config=simulation_config,
        )
    assert pressurefit_failure.value.kind == error.kind
    assert pressurefit_failure.value.required_bytes == error.required_bytes
    assert pressurefit_failure.value.capacity_bytes == error.capacity_bytes


def test_workspace_larger_than_the_device_is_rejected_before_search() -> None:
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(),
        objects=(),
        profiles=(TaskProfile("profile", 10, 123, "abi"),),
        tasks=(
            TaskSpec(
                "workspace",
                ResourceSpec("cuda_0", ResourceKind.COMPUTE),
                "profile",
            ),
        ),
    )

    with pytest.raises(PressureFitInfeasibleError) as caught:
        pressurefit(program, initial_residency=(), config=config(122))

    assert caught.value.kind == "workspace_capacity"
    assert caught.value.required_bytes == 123
