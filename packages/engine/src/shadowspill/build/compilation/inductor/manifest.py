"""What one compiled task is: its roots, its views, and how long it took."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    TaskStorageContract,
)
from shadowspill.task.manifest import (
    ExecutableRootAllocation,
    ExecutableTaskManifest,
    build_manifest,
)


@dataclass(frozen=True, slots=True)
class _LoweredOutput:
    semantic_view: OutputView
    optimized_view: OutputView
    provenance: StorageRoot
    root_name: str
    offset_bytes: int
    span_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class _GraphLoweringManifest:
    storage_contract: TaskStorageContract
    root_allocations: tuple[ExecutableRootAllocation, ...]


def toolchain(numerical_contracts: tuple[str, ...] = ()) -> dict[str, str]:
    """What compiled a task here: the framework and the device provider."""

    identity = {"torch": torch.__version__, "provider": provider_version() or ""}
    if numerical_contracts:
        identity["numerical_contracts"] = json.dumps(numerical_contracts)
    return identity


def _make_manifest(
    semantic_contract: TaskStorageContract,
    optimized_contract: TaskStorageContract,
    executable_contract: TaskStorageContract,
    root_allocations: tuple[ExecutableRootAllocation, ...],
    *,
    capture_ns: int,
    numerical_contracts: tuple[str, ...] = (),
) -> ExecutableTaskManifest:
    return build_manifest(
        semantic_contract,
        optimized_contract,
        executable_contract,
        root_allocations,
        capture_ns=capture_ns,
        toolchain=toolchain(numerical_contracts),
    )
