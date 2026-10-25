"""Restore finalized task code and the intrinsic operations it can call."""

from collections.abc import Mapping, Sequence
from typing import Any

from torch._inductor.standalone_compile import AOTCompiledArtifact

from . import operations as _operations  # noqa: F401
from .dependencies import restore_dependencies


def restore_executable(
    payload: bytes,
    *,
    dependencies: Sequence[Mapping[str, str]] = (),
    triton_choices: Mapping[str, Any] | None = None,
) -> Any:
    restore_dependencies(dependencies)
    function = AOTCompiledArtifact.deserialize(payload)
    if triton_choices is not None:
        from .tuning import use_choices

        function = use_choices(function, triton_choices)
    return function
