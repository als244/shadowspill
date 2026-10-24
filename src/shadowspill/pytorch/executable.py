"""Restore finalized task code and the intrinsic operations it can call."""

from typing import Any

from torch._inductor.standalone_compile import AOTCompiledArtifact

from . import operations as _operations  # noqa: F401


def restore_executable(payload: bytes) -> Any:
    return AOTCompiledArtifact.deserialize(payload)
