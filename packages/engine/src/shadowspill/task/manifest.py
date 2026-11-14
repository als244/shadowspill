"""What one compiled task promises, and how to tell it kept the promise.

`ExecutableTaskManifest` is the record: the storage contract the compiled task
executes against, the root allocations it needs, and the digest that identifies
the compilation. `validate_value_contract` is the check a compiler must pass --
it may introduce aliases, and it may not change what a task's outputs are worth.

Building a manifest stamps the framework and provider versions into its digest,
so that is the frontend's, in the frontend's compilation package. Reading one is
everyone's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.errors import CaptureError

from .storage import StorageRootKind, TaskStorageContract


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
        toolchain: Mapping[str, str],
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
        restored = build_manifest(
            semantic_contract,
            TaskStorageContract.from_dict(optimized),
            TaskStorageContract.from_dict(executable),
            tuple(allocations),
            capture_ns=capture_ns,
            toolchain=toolchain,
        )
        if restored.compatibility_digest != declared_digest:
            raise ValueError("compiled task manifest digest does not match")
        return restored


def build_manifest(
    semantic_contract: TaskStorageContract,
    optimized_contract: TaskStorageContract,
    executable_contract: TaskStorageContract,
    root_allocations: tuple[ExecutableRootAllocation, ...],
    *,
    capture_ns: int,
    toolchain: Mapping[str, str],
) -> ExecutableTaskManifest:
    """One manifest, and the digest that identifies the compilation behind it.

    ``toolchain`` names what compiled the task -- the framework and the device
    provider, by version. It is part of the digest, so a manifest cached by one
    toolchain does not match a task compiled by another, and it is an argument
    rather than a lookup because only a frontend knows what compiled anything.
    """

    validate_value_contract(semantic_contract, optimized_contract)
    validate_value_contract(semantic_contract, executable_contract)
    identity = {
        "semantic_contract_digest": semantic_contract.compatibility_digest,
        "optimized_storage_contract": optimized_contract.identity(),
        "optimized_storage_contract_digest": optimized_contract.compatibility_digest,
        "storage_contract": executable_contract.identity(),
        "storage_contract_digest": executable_contract.compatibility_digest,
        "root_allocations": [item.identity() for item in root_allocations],
        **dict(sorted(toolchain.items())),
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


def validate_value_contract(
    semantic: TaskStorageContract,
    executable: TaskStorageContract,
) -> None:
    """Require the compiler to preserve output values while allowing new aliases."""

    semantic_views = {view.leaf_index: view for view in semantic.output_views}
    executable_views = {view.leaf_index: view for view in executable.output_views}
    if semantic_views.keys() != executable_views.keys():
        raise CaptureError(
            "the compiler changed the output leaves of the task contract: "
            f"semantic={sorted(semantic_views)}, "
            f"executable={sorted(executable_views)}"
        )
    for leaf_index, semantic_view in semantic_views.items():
        executable_view = executable_views[leaf_index]
        semantic_geometry = (
            semantic_view.shape,
            semantic_view.stride,
            semantic_view.dtype,
            semantic_view.layout,
            semantic_view.span_bytes,
        )
        executable_geometry = (
            executable_view.shape,
            executable_view.stride,
            executable_view.dtype,
            executable_view.layout,
            executable_view.span_bytes,
        )
        same_significant_strides = all(
            extent <= 1 or semantic_stride == executable_stride
            for extent, semantic_stride, executable_stride in zip(
                semantic_view.shape,
                semantic_view.stride,
                executable_view.stride,
                strict=True,
            )
        )
        if (
            semantic_view.shape != executable_view.shape
            or semantic_view.dtype != executable_view.dtype
            or semantic_view.layout != executable_view.layout
            or semantic_view.span_bytes != executable_view.span_bytes
            or not same_significant_strides
        ):
            raise CaptureError(
                "the compiler changed task output geometry: "
                f"leaf={leaf_index}, semantic={semantic_geometry}, "
                f"executable={executable_geometry}"
            )
    semantic_mutations = {
        (item.input_position, item.replacement_output_leaf)
        for item in semantic.mutations
    }
    executable_mutations = {
        (item.input_position, item.replacement_output_leaf)
        for item in executable.mutations
    }
    if semantic_mutations != executable_mutations:
        raise CaptureError(
            "the compiler changed the task mutation contract: "
            f"semantic={sorted(semantic_mutations)}, "
            f"executable={sorted(executable_mutations)}"
        )


__all__ = [
    "ExecutableRootAllocation",
    "ExecutableTaskManifest",
    "build_manifest",
    "validate_value_contract",
]
