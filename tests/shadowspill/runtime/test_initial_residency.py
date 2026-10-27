"""An initial device declaration requires an actual current device copy."""

from types import SimpleNamespace

import pytest

from shadowspill.ir import MemoryLocation, ResidencySpec
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.runtime.plan.bridge import RuntimeBridge


@pytest.mark.parametrize(
    ("pointer", "version", "unexpected", "valid"),
    (
        (1234, 3, False, True),
        (0, 3, False, False),
        (1234, 2, False, False),
        (1234, 3, True, False),
    ),
)
def test_declared_device_copy_must_exist_and_be_current(
    pointer: int,
    version: int,
    unexpected: bool,
    valid: bool,
) -> None:
    def snapshot(runtime: int, object_id: str, target: object) -> int:
        value = target._obj
        value.execution_pointer = (
            pointer if object_id == "declared" else (9876 if unexpected else 0)
        )
        value.execution_version = version
        value.authoritative_version = 3
        return 0

    bridge = SimpleNamespace(
        runtime=SimpleNamespace(_runtime_handle=1),
        runtime_library=SimpleNamespace(shadowspill_object_snapshot=snapshot),
        objects=SimpleNamespace(
            requires_storage=lambda alias: True, runtime_object_id=lambda alias: alias
        ),
        _owned_aliases=("declared", "spill"),
        require=lambda status, message: None,
    )
    residency = (
        ResidencySpec("declared", MemoryLocation.DEVICE),
        ResidencySpec("spill", MemoryLocation.SPILL),
    )
    if valid:
        RuntimeBridge.require_initial_residency(bridge, residency)  # type: ignore[arg-type]
    else:
        with pytest.raises(RuntimeExecutionError, match="differs from the plan"):
            RuntimeBridge.require_initial_residency(bridge, residency)  # type: ignore[arg-type]
