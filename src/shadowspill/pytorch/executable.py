"""Restore finalized task code and the intrinsic operations it can call."""

from collections.abc import Mapping, Sequence
from typing import Any

from torch._inductor.standalone_compile import AOTCompiledArtifact

from . import operations as _operations  # noqa: F401
from .dependencies import restore_dependencies


def restore_executable(
    payload: bytes, *, dependencies: Sequence[Mapping[str, str]] = ()
) -> Any:
    restore_dependencies(dependencies)
    return AOTCompiledArtifact.deserialize(payload)
