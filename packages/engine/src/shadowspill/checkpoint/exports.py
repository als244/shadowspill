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
    identity = state_identity(state)
    if state_identity(layout) != identity:
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
        "state": identity,
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
    """Return the exact size of a worker's weights archive, in bytes.

    It is computed from the tensor descriptions and the manifest, without
    reading any tensor. `write_weights` with the same arguments writes
    exactly this many bytes.

    Parameters
    ----------
    state : Executor or checkpoint state
        The executor whose model state the archive holds, or any object that
        describes one worker's state as an executor does: ``plan``,
        ``plans``, ``logical_rank`` and ``tensor_chunks``.
    metadata : mapping of str to object, optional
        JSON data saved with the weights, as for `write_weights`.
    weights : str, default "master"
        Which copy of a weight held in two precisions is saved, as for
        `write_weights`.
    frozen_state_id : str, optional
        Identity of a frozen base the caller keeps elsewhere, as for
        `write_weights`.

    Returns
    -------
    int
        The archive's size, in bytes.

    Raises
    ------
    ValueError
        If ``metadata`` or the manifest holding it exceeds 16 MiB,
        ``frozen_state_id`` is empty, or ``weights`` is neither
        ``"master"`` nor ``"compute"``.
    """
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
    """Write a worker's model weights to a stream, as an inference archive.

    The archive holds the weights and persistent buffers this worker owns,
    and records each model tensor's logical shape, the slice this worker
    holds and which tensors share storage. It holds no optimizer state,
    gradients or random state, so it cannot resume training:
    `load_checkpoint` refuses it. Only the saved tensors are read, through
    the executor's bounded export, and the stream is written without
    seeking, flushing or closing: the caller publishes the archives of every
    worker, and discards what was written if this raises.

    Parameters
    ----------
    state : Executor or checkpoint state
        An initialized executor, or any object that describes and exports
        one worker's state as an executor does: ``plan``, ``plans``,
        ``logical_rank``, ``tensor_chunks`` and ``export_tensors``. An
        executor's outstanding run is waited for before tensors are read.
    stream : writable binary file
        Receives the archive through ``write``, as for `write_checkpoint`.
    metadata : mapping of str to object, optional
        JSON data saved with the weights, at most 16 MiB encoded.
    chunk_bytes : int, default 64 << 20
        Largest chunk read from the executor at once, in bytes.
    weights : str, default "master"
        Which copy of a weight held in two precisions is saved: the masters
        with ``"master"``, the compute copies with ``"compute"``. Encoded
        representations and incompatible layouts keep both, with the reason
        in the manifest's ``retained_weight_copies``.
    frozen_state_id : str, optional
        Identity of a frozen base the caller keeps elsewhere. With it, the
        parameters no admitted Plan changes are left out. Trainable
        parameters, buffers and anything a Plan changes are always saved.

    Returns
    -------
    dict
        The archive's manifest, as written: each saved tensor's name,
        geometry, size and BLAKE3 checksum, each model tensor's view, the
        ``metadata``, and the state's identity.

    Raises
    ------
    RuntimeError
        If the executor is closed, failed or not initialized.
    ValueError
        If ``metadata`` or the manifest holding it exceeds 16 MiB,
        ``frozen_state_id`` is empty, or ``weights`` is neither
        ``"master"`` nor ``"compute"``.
    OSError
        If ``write`` returns anything but a positive count no larger than
        the data it was given.
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
