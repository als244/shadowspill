"""What one compiled task is: its roots, its views, and how long it took."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)

from .values import _validate_value_contract


@dataclass(frozen=True, slots=True)
class ExecutableRootAllocation:
    """Compiler-owned allocation extent for one executable storage root."""

    root_id: int
    requested_bytes: int

    def __post_init__(self) -> None:
        if self.root_id < 0 or self.requested_bytes < 0:
            raise ValueError("executable root allocation fields must be non-negative")

    def identity(self) -> dict[str, int]:
        return {
            "root_id": self.root_id,
            "requested_bytes": self.requested_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExecutableTaskManifest:
    """Offline storage contract emitted for one optimized compiled task."""

    semantic_contract_digest: str
    storage_contract: TaskStorageContract
    contract_capture_ns: int
    compatibility_digest: str
    optimized_storage_contract: TaskStorageContract | None = None
    root_allocations: tuple[ExecutableRootAllocation, ...] = ()

    def __post_init__(self) -> None:
        if len(self.semantic_contract_digest) != 64:
            raise ValueError("semantic contract digest must be SHA-256")
        if self.contract_capture_ns < 0:
            raise ValueError("executable contract timing must be non-negative")
        if len(self.compatibility_digest) != 64:
            raise ValueError("executable manifest digest must be SHA-256")
        if tuple(item.root_id for item in self.root_allocations) != tuple(
            range(len(self.storage_contract.roots))
        ):
            raise ValueError(
                "executable root allocations must have contiguous root indices"
            )
        for root, allocation in zip(
            self.storage_contract.roots, self.root_allocations, strict=True
        ):
            if root.kind is StorageRootKind.INPUT and allocation.requested_bytes:
                raise ValueError("input executable root cannot allocate storage")
            if (
                root.kind is StorageRootKind.FRESH
                and allocation.requested_bytes < root.minimum_span_bytes
            ):
                raise ValueError(
                    "executable root allocation is smaller than its output views"
                )

    def identity(self) -> dict[str, object]:
        return {
            "semantic_contract_digest": self.semantic_contract_digest,
            "optimized_storage_contract": (
                self.optimized_storage_contract.identity()
                if self.optimized_storage_contract is not None
                else self.storage_contract.identity()
            ),
            "storage_contract": self.storage_contract.identity(),
            "storage_contract_digest": self.storage_contract.compatibility_digest,
            "root_allocations": [item.identity() for item in self.root_allocations],
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the compiler-owned storage contract for a profile sidecar."""

        if self.optimized_storage_contract is None:
            raise ValueError("compiled task manifest has no optimized contract")
        return {
            "semantic_contract_digest": self.semantic_contract_digest,
            "storage_contract": self.storage_contract.to_dict(),
            "contract_capture_ns": self.contract_capture_ns,
            "compatibility_digest": self.compatibility_digest,
            "optimized_storage_contract": (self.optimized_storage_contract.to_dict()),
            "root_allocations": [item.identity() for item in self.root_allocations],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        semantic_contract: TaskStorageContract,
    ) -> ExecutableTaskManifest:
        """Restore and validate one cached compiler storage contract."""

        expected = {
            "semantic_contract_digest",
            "storage_contract",
            "contract_capture_ns",
            "compatibility_digest",
            "optimized_storage_contract",
            "root_allocations",
        }
        if set(payload) != expected:
            raise ValueError("compiled task manifest fields differ from schema")
        if payload["semantic_contract_digest"] != (
            semantic_contract.compatibility_digest
        ):
            raise ValueError("compiled task manifest has the wrong semantic contract")
        executable = payload["storage_contract"]
        optimized = payload["optimized_storage_contract"]
        capture_ns = payload["contract_capture_ns"]
        declared_digest = payload["compatibility_digest"]
        raw_allocations = payload["root_allocations"]
        if not isinstance(executable, dict) or not isinstance(optimized, dict):
            raise ValueError("compiled task manifest contracts must be objects")
        if (
            not isinstance(capture_ns, int)
            or isinstance(capture_ns, bool)
            or capture_ns < 0
        ):
            raise ValueError("compiled task manifest timing is invalid")
        if not isinstance(declared_digest, str):
            raise ValueError("compiled task manifest digest is invalid")
        if not isinstance(raw_allocations, list):
            raise ValueError("compiled task root allocations must be a list")
        allocations: list[ExecutableRootAllocation] = []
        for item in raw_allocations:
            if not isinstance(item, dict) or set(item) != {
                "root_id",
                "requested_bytes",
            }:
                raise ValueError("compiled task root allocation is invalid")
            root_id = item["root_id"]
            requested_bytes = item["requested_bytes"]
            if (
                not isinstance(root_id, int)
                or isinstance(root_id, bool)
                or not isinstance(requested_bytes, int)
                or isinstance(requested_bytes, bool)
            ):
                raise ValueError("compiled task root allocation is invalid")
            allocations.append(ExecutableRootAllocation(root_id, requested_bytes))
        restored = _make_manifest(
            semantic_contract,
            TaskStorageContract.from_dict(optimized),
            TaskStorageContract.from_dict(executable),
            tuple(allocations),
            capture_ns=capture_ns,
        )
        if restored.compatibility_digest != declared_digest:
            raise ValueError("compiled task manifest digest does not match")
        return restored


@dataclass(frozen=True, slots=True)
class InductorCompilation:
    """Callable and the optimized storage contract it actually implements."""

    function: Callable[..., object]
    manifest: ExecutableTaskManifest
    phase_timings_ns: tuple[tuple[str, int], ...] = ()


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


def _make_manifest(
    semantic_contract: TaskStorageContract,
    optimized_contract: TaskStorageContract,
    executable_contract: TaskStorageContract,
    root_allocations: tuple[ExecutableRootAllocation, ...],
    *,
    capture_ns: int,
) -> ExecutableTaskManifest:
    _validate_value_contract(semantic_contract, optimized_contract)
    _validate_value_contract(semantic_contract, executable_contract)
    identity = {
        "semantic_contract_digest": semantic_contract.compatibility_digest,
        "optimized_storage_contract": optimized_contract.identity(),
        "optimized_storage_contract_digest": optimized_contract.compatibility_digest,
        "storage_contract": executable_contract.identity(),
        "storage_contract_digest": executable_contract.compatibility_digest,
        "root_allocations": [item.identity() for item in root_allocations],
        "torch": torch.__version__,
        "provider": provider_version(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return ExecutableTaskManifest(
        semantic_contract.compatibility_digest,
        executable_contract,
        capture_ns,
        hashlib.sha256(encoded.encode()).hexdigest(),
        optimized_contract,
        root_allocations,
    )
