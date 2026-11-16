"""Move verified logical families into existing bounded unpublished staging."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from math import prod

from shadowspill.artifacts.state import StateId
from shadowspill.artifacts.tensors import TensorChunk

from .compatibility import RankedTensor
from .reading import RankReader
from .staging import TensorStaging
from .tensors import region_bytes
from .transfer import SourceView, TargetView, transfer_tensor


def _source(entry: RankedTensor, readers: Sequence[RankReader]) -> SourceView | None:
    item = entry.tensor
    if item.archive is None or not item.view.region_elements:
        return None
    reader = readers[entry.rank]
    name = item.archive.name
    if name in reader.manifest["omitted_parameters"]:
        return None
    view = item.view
    if name not in reader.chunks:
        if item.tensor.identity.role not in {"parameter", "master"}:
            raise ValueError("checkpoint logical state has no saved representation")
        name = reader.manifest["derived_tensors"].get(name) or reader.manifest[
            "gathered_tensors"
        ].get(name, {}).get("source")
        if name not in reader.chunks:
            # Empty selected shards have no file member; other logical ranks
            # must still supply complete coverage of the authoritative state.
            return None
        selected = reader.chunks[name]
        view = replace(
            view,
            storage_elements=selected.storage_elements,
            region_offset=selected.region_offset,
            region_elements=region_bytes(selected) // selected.dtype.itemsize,
        )
    return SourceView(reader, reader.chunks[name], view)


def transfer_family(
    sources: Mapping[StateId, Sequence[RankedTensor]],
    targets: Mapping[StateId, Sequence[RankedTensor]],
    readers: Sequence[RankReader],
    chunks: Mapping[tuple[int, str], TensorChunk],
    staging: TensorStaging,
    *,
    scratch_bytes: int,
) -> None:
    """Copy declared source representations; compatibility is preflighted by owner."""
    for identity, entries in targets.items():
        selected = [
            entry
            for entry in entries
            if entry.tensor.archive is not None
            and (entry.rank, entry.tensor.archive.name) in chunks
        ]
        if not selected or not prod(selected[0].tensor.view.shape):
            continue
        source_views = [
            source
            for entry in sources[identity]
            if (source := _source(entry, readers)) is not None
        ]
        if not source_views:
            raise ValueError(
                f"checkpoint state requires an unavailable dependency: {identity}"
            )
        destinations = []
        for entry in selected:
            assert entry.tensor.archive is not None
            key = entry.rank, entry.tensor.archive.name
            destinations.append(
                TargetView(
                    entry.tensor.view,
                    entry.tensor.archive.dtype,
                    partial(staging.write, key),
                )
            )
        transfer_tensor(
            source_views,
            destinations,
            scratch_bytes=scratch_bytes,
            transformation="scalar_cast"
            if identity.role in {"parameter", "master"}
            and source_views[0].chunk.dtype != destinations[0].dtype
            else "identity",
        )
