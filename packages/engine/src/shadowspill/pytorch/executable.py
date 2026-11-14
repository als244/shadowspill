"""Restore finalized task code and the intrinsic operations it can call."""

from collections.abc import Mapping, Sequence
from typing import Any

from torch._inductor.standalone_compile import AOTCompiledArtifact

from . import operations as _operations  # noqa: F401
from .dependencies import restore_dependencies
from .numerics import validate_numerical_settings


def restore_executable(
    payload: bytes,
    *,
    dependencies: Sequence[Mapping[str, str]] = (),
    numerical_contracts: Sequence[str] = (),
    triton_choices: Mapping[str, Any] | None = None,
) -> Any:
    validate_numerical_settings(numerical_contracts)
    from .chunking.executable import MAGIC, ChunkedExecutable
    from .chunking.region_executable import MAGIC as REGION_MAGIC
    from .chunking.region_executable import RegionExecutable

    if payload.startswith(b"SHADOWSPILL-ROW-COMPOSITION-") and not payload.startswith(
        MAGIC
    ):
        raise ValueError("unsupported row composition version; rebuild the Program")
    restore_dependencies(dependencies)
    if payload.startswith(
        b"SHADOWSPILL-REGION-COMPOSITION-"
    ) and not payload.startswith(REGION_MAGIC):
        raise ValueError("unsupported region composition version; rebuild the Program")
    function = (
        RegionExecutable.deserialize(payload)
        if payload.startswith(REGION_MAGIC)
        else ChunkedExecutable.deserialize(payload)
        if payload.startswith(MAGIC)
        else AOTCompiledArtifact.deserialize(payload)
    )
    if triton_choices is not None:
        from .tuning import use_choices

        return use_choices(function, triton_choices)
    return function
