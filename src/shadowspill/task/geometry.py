"""Storage extents and offsets needed to bind outputs, without profiling."""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import ExecutableTaskManifest


@dataclass(frozen=True, slots=True)
class TaskStorageGeometry:
    """Output geometry; this is not an allocation trace or execution admission.

    Build uses compiler-requested extents. Measured lowering can additionally
    carry allocator-rounded capacities without changing the binding algorithm.
    Anonymous scratch and allocation order are deliberately not inferred here.
    """

    contract_digest: str
    requested_bytes: tuple[int, ...]
    capacity_bytes: tuple[int, ...]
    output_offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.contract_digest) != 64:
            raise ValueError("storage geometry requires a contract digest")
        if len(self.requested_bytes) != len(self.capacity_bytes):
            raise ValueError("storage geometry root counts differ")
        if any(
            requested < 0 or capacity < requested
            for requested, capacity in zip(
                self.requested_bytes, self.capacity_bytes, strict=True
            )
        ):
            raise ValueError("storage geometry has invalid root extents")
        leaves = [leaf for leaf, _ in self.output_offsets]
        if len(set(leaves)) != len(leaves) or any(
            leaf < 0 or offset < 0 for leaf, offset in self.output_offsets
        ):
            raise ValueError("storage geometry has invalid output offsets")

    @classmethod
    def from_manifest(cls, manifest: ExecutableTaskManifest) -> TaskStorageGeometry:
        sizes = tuple(item.requested_bytes for item in manifest.root_allocations)
        return cls(
            manifest.storage_contract.compatibility_digest,
            sizes,
            sizes,
            tuple(
                (view.leaf_index, view.offset_bytes)
                for view in manifest.storage_contract.output_views
            ),
        )

    def output_offset(self, leaf_index: int) -> int:
        return dict(self.output_offsets)[leaf_index]
