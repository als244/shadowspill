"""Prepare model weights for fresh initialization, independently of optimizer state."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO

from shadowspill.artifacts._values import freeze
from shadowspill.artifacts.state import StateId
from shadowspill.artifacts.tensors import TensorChunk

from .compatibility import LogicalFamily, RankedTensor, describe_family
from .conversion import ConvertedRank, ConvertedState
from .frozen import omitted_parameters, validate_identity
from .inspection import MANIFEST_LIMIT
from .redistribution import transfer_family
from .staging import stage_tensors
from .state import PlanState
from .weights_reading import read_weights

_FLOATS = {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}


def _require_model(source: LogicalFamily, target: LogicalFamily) -> None:
    if source.tensors.keys() != target.tensors.keys():
        raise ValueError(
            "weights model inventory differs; an explicit mapping is required"
        )
    for key in source.tensors:
        left, right = source.identities[key], target.identities[key]
        # Training selection and optimizer descriptions do not define a model
        # weight. Ordinary floating parameters may change precision explicitly;
        # buffers and encoded components retain their exact interpretation.
        if left[0] != right[0] or (
            left[1] != right[1]
            and not (
                key.role == "parameter"
                and not key.component_path
                and {left[1], right[1]} <= _FLOATS
            )
        ):
            raise ValueError(f"weights model interpretation differs: {key}")
        if source.ties[key] != target.ties[key]:
            raise ValueError(f"weights model structural ties differ: {key}")
        if source.relations[key] != target.relations[key]:
            raise ValueError(f"weights model tied view relationship differs: {key}")


def _omitted(states: Sequence[PlanState]) -> frozenset[StateId]:
    return frozenset(
        item.tensor.identity
        for state in states
        for names in (omitted_parameters(state),)
        for item in state.logical_tensors(roles=("parameter", "buffer"))
        if item.archive is not None and item.archive.name in names
    )


def weights_tensors(
    targets: Sequence[PlanState], frozen_state_id: str | None = None
) -> dict[tuple[int, str], TensorChunk]:
    """Admit all target model copies; training state never enters this staging."""
    validate_identity(frozen_state_id)
    return {
        (state.logical_rank, item.archive.name): item.archive
        for state in targets
        for omitted in (
            set() if frozen_state_id is None else omitted_parameters(state),
        )
        for item in state.logical_tensors(roles=("parameter", "buffer", "master"))
        if item.archive is not None and item.archive.name not in omitted
    }


def _sources(
    source: LogicalFamily, target: LogicalFamily
) -> dict[StateId, tuple[RankedTensor, ...]]:
    result = {}
    for key, entries in target.tensors.items():
        original = key
        if key.role == "master":
            owners = {entry.tensor.tensor.parameter_name for entry in entries}
            if len(owners) != 1 or None in owners or key.component_path:
                raise ValueError(
                    "weight masters require an explicit scalar parameter owner"
                )
            owner = next(iter(owners))
            assert owner is not None
            if key not in source.tensors:
                original = replace(key, role="parameter", name=owner)
            if target.identities[key][1] not in _FLOATS:
                raise ValueError("weight masters require ordinary floating storage")
        if original not in source.tensors:
            raise ValueError(f"weight representation has no source parameter: {key}")
        result[key] = source.tensors[original]
    return result


@contextmanager
def convert_weights(
    sources: Sequence[tuple[PlanState, Path | BinaryIO]],
    targets: Sequence[PlanState],
    directory: Path,
    *,
    scratch_bytes: int,
    max_disk_bytes: int,
    frozen_state_id: str | None = None,
    max_source_manifest_bytes: int = MANIFEST_LIMIT,
    expected_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[ConvertedState]:
    """Repartition only model state using the ordinary verified transfer path.

    Every compute/master copy comes from the export's selected representation;
    importing masters therefore never round-trips through reduced precision.
    Ties, complete coverage and replicas are checked before publication. An exact
    frozen-base dependency can remain omitted, but its omitted logical inventory
    cannot change. The owner must initialize that base separately.

    No optimizer, RNG, gradients or temporary values are copied or generated.
    Fresh worker initialization supplies their usual defaults. Checkpoint resume
    continues to require its complete state and strict optimizer compatibility.
    """
    sources, targets = tuple(sources), tuple(targets)
    if type(scratch_bytes) is not int or scratch_bytes < 2:
        raise ValueError("weights conversion requires a positive scratch allowance")
    if expected_metadata is not None:
        if len(expected_metadata) != len(sources):
            raise ValueError(
                "weights conversion requires metadata for every source rank"
            )
        expected_metadata = freeze(expected_metadata)
    source_states = [state for state, _ in sources]
    source_model = describe_family(source_states, roles=("parameter", "buffer"))
    target_model = describe_family(targets, roles=("parameter", "buffer"))
    _require_model(source_model, target_model)
    target_family = describe_family(targets, roles=("parameter", "buffer", "master"))
    source_family = describe_family(
        source_states, roles=("parameter", "buffer", "master")
    )
    source_entries = _sources(source_family, target_family)
    chunks = weights_tensors(targets, frozen_state_id)
    if frozen_state_id is not None and _omitted(source_states) != _omitted(targets):
        raise ValueError("weights conversion changes the omitted frozen-base inventory")
    transfer_budget = scratch_bytes // 2
    with ExitStack() as owned:
        reading = owned.enter_context(ExitStack())
        readers = [
            reading.enter_context(
                read_weights(
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
                "weights ranks disagree on source boundary or weight selection"
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
            source_entries,
            target_family.tensors,
            readers,
            chunks,
            staging,
            scratch_bytes=transfer_budget,
        )
        for entries in target_family.tensors.values():
            for entry in entries:
                if entry.tensor.archive is not None:
                    key = entry.rank, entry.tensor.archive.name
                    if key in chunks:
                        staging.require_written(key, entry.tensor.view)
        reading.close()
        staging.seal()
        lock = Lock()
        yield ConvertedState(
            tuple(ConvertedRank(state, staging, lock) for state in targets),
            target_model.identity,
            first["weights"],
            frozen_state_id,
            first["metadata"],
        )
