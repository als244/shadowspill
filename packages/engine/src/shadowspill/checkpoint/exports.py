"""Inference model state through the existing owned tensor archive encoder.

These archives contain model weights/buffers and explicit logical ownership, never
optimizer, gradient, RNG or execution-temporary payloads. They are deliberately not
resumable training checkpoints. Workload formatters can consume this representation
without reimplementing worker tensor transfers or reconstructing sharding by name.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import digest, encoded, plain
from shadowspill.artifacts.tensors import TensorChunk

from .archive import MANIFEST_LIMIT, WriteAll, WriteSink, finish, header, padding
from .encoding import ArchiveSize, TensorWriter, tensor_entry
from .frozen import omitted_parameters, validate_identity
from .identity import state_identity
from .state import CheckpointExport, CheckpointState, PlanState
from .tensors import geometry, region_bytes
from .weights import WeightRepresentation, select_weights

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class _Description:
    chunks: tuple[TensorChunk, ...]
    fields: dict[str, Any]

    def manifest(self, tensors: list[dict[str, Any]]) -> dict[str, Any]:
        return {**self.fields, "tensors": tensors}


def _describe(
    state: CheckpointState,
    metadata: Mapping[str, Any] | None,
    weights: WeightRepresentation,
    frozen_state_id: str | None,
) -> _Description:
    body = encoded(dict(metadata or {}))
    if len(body) > MANIFEST_LIMIT:
        raise ValueError("weights metadata exceeds 16 MiB")
    saved_metadata = json.loads(body)
    del body
    validate_identity(frozen_state_id)
    omitted = set() if frozen_state_id is None else omitted_parameters(state)
    selected = select_weights(state, weights)
    layout = PlanState(state.plan, state.plans, logical_rank=state.logical_rank)
    if state_identity(layout) != state_identity(state):
        raise ValueError("exported state differs from saved tensor ownership")
    logical = tuple(layout.logical_tensors(roles=("parameter", "buffer", "master")))
    model_names = {item.archive.name for item in logical if item.archive is not None}
    excluded = omitted | selected.derived.keys() | selected.gathered.keys()
    chunks = tuple(
        chunk
        for chunk in state.tensor_chunks()
        if chunk.name in model_names
        and chunk.name not in excluded
        and region_bytes(chunk)
    )
    views = [
        {
            "tensor": item.tensor.to_dict(),
            "archive": None if item.archive is None else item.archive.name,
            "view": plain(asdict(item.view)),
        }
        for item in logical
    ]
    fields = {
        "format": "shadowspill.weights",
        "version": 1,
        "byteorder": sys.byteorder,
        "logical_rank": state.logical_rank,
        "model_state": digest(views),
        "program": state.plan.program.digest,
        "additional_programs": {
            name: plan.program.digest for name, plan in state.plans.items()
        },
        "model_tensors": views,
        "weights": weights,
        "frozen_state_id": frozen_state_id,
        "omitted_parameters": sorted(omitted),
        "derived_tensors": {
            name: source
            for name, source in selected.derived.items()
            if name in model_names
        },
        "gathered_tensors": {
            name: row for name, row in selected.gathered.items() if name in model_names
        },
        "retained_weight_copies": selected.retained,
        "metadata": saved_metadata,
    }
    return _Description(chunks, fields)


def _size(description: _Description) -> int:
    counter = ArchiveSize()
    tensors: list[dict[str, Any]] = []
    for chunk in description.chunks:
        entry = tensor_entry(chunk, len(tensors))
        entry["blake3"] = "0" * 64
        tensors.append(entry)
        header(counter, entry["member"], entry["bytes"])
        counter.size_bytes += entry["bytes"]
        padding(counter, entry["bytes"])
    finish(counter, encoded(description.manifest(tensors)))
    return counter.size_bytes


def weights_size(
    state: CheckpointState,
    *,
    metadata: Mapping[str, Any] | None = None,
    weights: WeightRepresentation = "master",
    frozen_state_id: str | None = None,
) -> int:
    """Exact inference archive bytes from saved declarations, without tensor I/O."""
    return _size(_describe(state, metadata, weights, frozen_state_id))


def write_weights(
    state: CheckpointExport,
    stream: WriteSink,
    *,
    metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = 64 << 20,
    weights: WeightRepresentation = "master",
    frozen_state_id: str | None = None,
) -> dict[str, Any]:
    """Stream model state only; the owner commits files and their complete rank set.

    Unchanged base weights are omitted only with a verified external identity.
    Mutated frozen tensors and persistent buffers remain necessary for inference.
    Authoritative master/compute selection follows the ordinary checkpoint rules.
    """
    description = _describe(state, metadata, weights, frozen_state_id)
    _size(description)  # Bound all manifest/framing metadata before touching data.
    expected = {chunk.name: chunk for chunk in description.chunks}
    target = WriteAll(stream)
    writer = TensorWriter(target)

    def write(chunk: TensorChunk, value: torch.Tensor) -> None:
        original = expected.get(chunk.name)
        if (
            original is None
            or geometry(original) != geometry(chunk)
            or value.dtype != chunk.dtype
        ):
            raise ValueError(
                "weights exporter returned an unselected or changed tensor"
            )
        writer.write(chunk, value)

    try:
        state.export_tensors(write, chunk_bytes=chunk_bytes, names=frozenset(expected))
        writer.close()
        if {entry["name"] for entry in writer.entries} != expected.keys():
            raise ValueError("weights exporter omitted selected model state")
        manifest = description.manifest(writer.entries)
        finish(target, encoded(manifest))
        return manifest
    except BaseException:
        writer.abort()
        raise
