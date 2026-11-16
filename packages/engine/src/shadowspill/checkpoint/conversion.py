"""Materialize compatible logical state before ordinary checkpoint publication."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO

import torch

from shadowspill.artifacts._values import freeze
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.tensors import TensorChunk, export_selection

from .compatibility import describe_family, require_compatible
from .frozen import omitted_parameters
from .inspection import MANIFEST_LIMIT
from .mapping import LogicalView
from .reading import read_rank
from .redistribution import transfer_family
from .staging import TensorStaging, stage_tensors
from .state import PlanState
from .tensors import region_bytes
from .weights import WeightRepresentation, select_weights

type InitializeTemporary = Callable[[int, TensorChunk, torch.Tensor], None]


@dataclass(frozen=True)
class ConvertedRank:
    state: PlanState
    staging: TensorStaging
    lock: Lock

    @property
    def plan(self) -> Plan:
        return self.state.plan

    @property
    def plans(self) -> Mapping[str, Plan]:
        return self.state.plans

    @property
    def logical_rank(self) -> int:
        return self.state.logical_rank

    def tensor_chunks(self) -> Iterator[TensorChunk]:
        return self.state.tensor_chunks()

    def export_tensors(
        self,
        callback: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int,
        names: Collection[str] | None = None,
    ) -> None:
        with self.lock:
            selected = export_selection(self.tensor_chunks(), names)
            self.staging.export(
                self.logical_rank, callback, chunk_bytes=chunk_bytes, names=selected
            )


@dataclass(frozen=True)
class ConvertedState:
    ranks: tuple[ConvertedRank, ...]
    logical_identity: str
    weights: WeightRepresentation
    frozen_state_id: str | None
    source_metadata: Mapping[str, Any]


def conversion_tensors(
    targets: Sequence[PlanState],
    weights: WeightRepresentation,
    frozen_state_id: str | None = None,
) -> dict[tuple[int, str], TensorChunk]:
    """Exact staged representations, shared with pre-conversion disk admission."""
    chunks = {}
    for state in targets:
        selection = select_weights(state, weights)
        excluded = set(selection.derived) | set(selection.gathered)
        if frozen_state_id is not None:
            excluded.update(omitted_parameters(state))
        for chunk in state.tensor_chunks():
            if chunk.name not in excluded:
                chunks[(state.logical_rank, chunk.name)] = chunk
    return chunks


@contextmanager
def convert_state(
    sources: Sequence[tuple[PlanState, Path | BinaryIO]],
    targets: Sequence[PlanState],
    directory: Path,
    *,
    scratch_bytes: int,
    max_disk_bytes: int,
    frozen_state_id: str | None = None,
    initialize_temporary: InitializeTemporary | None = None,
    max_source_manifest_bytes: int = MANIFEST_LIMIT,
    expected_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[ConvertedState]:
    """Prepare unpublished state; the owner uses the existing rank/global writers.

    Source files stay immutable through validation and transfer. All source read
    contexts close successfully before the converted state is yielded, so a
    publisher can then commit the complete target generation. Temporary disk and
    payload scratch are bounded here; decoded metadata and final archive storage
    remain separate reservations of the preparation owner.

    ``initialize_temporary`` supplies only target-private values absent from the
    checkpointable logical inventory (such as call-local controls and cleared
    gradients). It never receives weights, moments, counters or registered state.
    Missing temporary preparation fails explicitly instead of copying private
    state from unrelated operation Plans or opening a fresh model initializer.

    Rank-specific attachments require exact preflighted ``expected_metadata``
    for every source. Without it, all ranks must retain identical metadata.
    """
    sources, targets = tuple(sources), tuple(targets)
    if expected_metadata is not None and len(expected_metadata) != len(sources):
        raise ValueError("conversion requires metadata for every source rank")
    if expected_metadata is not None:
        expected_metadata = freeze(expected_metadata)
    source_family = describe_family([state for state, _ in sources])
    target_family = describe_family(targets)
    require_compatible(source_family, target_family)
    if type(scratch_bytes) is not int or scratch_bytes < 2:
        raise ValueError("checkpoint conversion requires a positive scratch allowance")
    transfer_budget = scratch_bytes // 2
    with ExitStack() as owned:
        reading = owned.enter_context(ExitStack())
        readers = [
            reading.enter_context(
                read_rank(
                    state,
                    path,
                    chunk_bytes=transfer_budget,
                    frozen_state_id=frozen_state_id,
                    max_manifest_bytes=max_source_manifest_bytes,
                )
            )
            for state, path in sources
        ]
        first = readers[0].manifest
        if any(
            reader.manifest["metadata"]
            != (
                first["metadata"]
                if expected_metadata is None
                else expected_metadata[rank]
            )
            or reader.manifest["weights"] != first["weights"]
            for rank, reader in enumerate(readers)
        ):
            raise ValueError(
                "checkpoint ranks disagree on source boundary or weight selection"
            )
        weights: WeightRepresentation = first["weights"]
        chunks = conversion_tensors(targets, weights, frozen_state_id)
        logical_keys = {
            (entry.rank, entry.tensor.archive.name)
            for entries in target_family.tensors.values()
            for entry in entries
            if entry.tensor.archive is not None
        }
        temporary = {
            key for key in chunks.keys() - logical_keys if region_bytes(chunks[key])
        }
        if temporary and initialize_temporary is None:
            raise ValueError(
                "checkpoint conversion requires target temporary initialization"
            )
        staging = owned.enter_context(
            stage_tensors(
                chunks,
                directory,
                scratch_bytes=scratch_bytes - transfer_budget,
                max_disk_bytes=max_disk_bytes,
            )
        )
        transfer_family(
            source_family.tensors,
            target_family.tensors,
            readers,
            chunks,
            staging,
            scratch_bytes=transfer_budget,
        )
        for key in temporary:
            assert initialize_temporary is not None
            chunk = chunks[key]
            step = transfer_budget // chunk.dtype.itemsize
            if not step:
                raise ValueError("checkpoint temporary budget cannot hold one element")
            elements = region_bytes(chunk) // chunk.dtype.itemsize
            buffer = torch.empty(min(step, elements), dtype=chunk.dtype, device="cpu")
            for start in range(0, elements, step):
                value = buffer[: min(step, elements - start)]
                initialize_temporary(
                    key[0], replace(chunk, offset=chunk.region_offset + start), value
                )
                staging.write(key, start, value)
                del value
            del buffer
            staging.require_written(
                key,
                LogicalView(
                    (chunk.storage_elements,),
                    (chunk.storage_elements,),
                    (0,),
                    (1,),
                    0,
                    chunk.storage_elements,
                    chunk.region_offset,
                    elements,
                ),
            )
        for entries in target_family.tensors.values():
            for entry in entries:
                if entry.tensor.archive is not None:
                    key = entry.rank, entry.tensor.archive.name
                    if key in chunks:
                        staging.require_written(key, entry.tensor.view)
        # Validate immutable source handles before allowing any publication.
        reading.close()
        staging.seal()
        lock = Lock()
        yield ConvertedState(
            tuple(ConvertedRank(state, staging, lock) for state in targets),
            target_family.identity,
            weights,
            frozen_state_id,
            first["metadata"],
        )
