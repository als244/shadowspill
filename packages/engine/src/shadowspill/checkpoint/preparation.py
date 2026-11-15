"""CPU conversion preparation, isolated from the service controller process."""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.plan import Plan
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.worker.checkpoint_reader import rank_randomness
from shadowspill.training.worker.checkpoint_records import (
    RankRestore,
    RankSave,
    RankSaved,
)
from shadowspill.training.worker.conversion_records import (
    ConversionPrepared,
    ConversionRequest,
    ConversionWritten,
)
from shadowspill_client._records import mapping
from shadowspill_client.protocol.checkpoints import CheckpointConversion

from .compatibility import describe_family, require_compatible
from .conversion import conversion_tensors, convert_state
from .defaults import TemporaryDefaults
from .identity import state_identity
from .load import validate_checkpoint
from .random_streams import (
    RandomState,
    default_streams,
    map_random_streams,
    read_snapshot,
    snapshot,
)
from .save import checkpoint_size, write_checkpoint
from .staging import staging_bytes
from .state import PlanState


def plan_states(plans: Mapping[int, Mapping[str, Plan]]) -> tuple[PlanState, ...]:
    if set(plans) != set(range(len(plans))) or not plans:
        raise ValueError("checkpoint Plans require every logical rank")
    states = []
    for rank in range(len(plans)):
        additional = dict(plans[rank])
        # The same primary selection as WorkerBindings, independent of kernels.
        primary = additional.pop(min(additional))
        states.append(PlanState(primary, additional, logical_rank=rank))
    return tuple(states)


@dataclass
class PreparedConversion:
    request: ConversionRequest
    description: ConversionPrepared
    sources: tuple[PlanState, ...]
    targets: tuple[PlanState, ...]
    defaults: TemporaryDefaults
    scratch_bytes: int
    staging_bytes: int
    source_metadata: tuple[Mapping[str, Any], ...]


def prepare(request: ConversionRequest) -> PreparedConversion:
    from .preparation_files import verified_file

    host_bytes = request.host_bytes
    scratch_bytes = min(64 << 20, host_bytes // 16)
    if scratch_bytes < 64 << 10:
        raise ValueError("checkpoint conversion host envelope is too small")
    saved = request.source.info
    states = []
    for reference in (saved.plans, request.plans):
        with verified_file(reference, Path(request.files[reference.id])) as stream:
            bundle = TrainingBundle.load(
                stream,
                max_metadata_bytes=host_bytes // 512,
                max_payload_bytes=host_bytes // 8,
            )
        states.append(plan_states(bundle.plans))
    sources, targets = states
    if len(sources) != saved.world_size or any(
        state_identity(state) != rank.state_sha256
        for state, rank in zip(sources, request.source.ranks, strict=True)
    ):
        raise ValueError("checkpoint source Plans differ from saved rank ownership")
    old, new = describe_family(sources), describe_family(targets)
    require_compatible(old, new)
    defaults = TemporaryDefaults(targets)
    provenance = CheckpointConversion(request.checkpoint, saved.plans, new.identity)
    identity = digest(
        {
            "conversion": provenance.to_dict(),
            "plans": request.plans.to_dict(),
            "configuration": request.configuration.to_dict(),
        }
    )
    metadata: list[Mapping[str, Any]] = []
    saved_states: list[RandomState] = []
    seeds: set[int] = set()
    for state, rank in zip(sources, request.source.ranks, strict=True):
        with verified_file(
            rank.archive, Path(request.files[rank.archive.id])
        ) as stream:
            manifest = validate_checkpoint(
                state,
                stream,
                frozen_state_id=saved.frozen_state_id,
                max_manifest_bytes=host_bytes // (256 * len(sources)),
            )
        randomness = rank_randomness(
            manifest["metadata"], RankRestore(request.checkpoint, saved, rank)
        )
        metadata.append(manifest["metadata"])
        if randomness is not None:
            seed, streams = read_snapshot(randomness)
            seeds.add(seed)
            saved_states.extend(streams)
    target_randomness: dict[int, Mapping[str, Any]] = {}
    if any("randomness" in row for row in metadata):
        if len(seeds) != 1 or any("randomness" not in row for row in metadata):
            raise ValueError("checkpoint ranks disagree on saved RNG seed or ownership")
        target_streams = [
            stream
            for state in targets
            for stream in default_streams(
                state.logical_rank, torch.__version__, accelerator=True
            )
        ]
        mapped = map_random_streams(
            saved_states,
            target_streams,
            root_seed=seeds.pop(),
            restore_id=identity,
            initializers={stream.representation for stream in target_streams},
        )
        provenance = replace(provenance, randomness=mapped.provenance())
        target_randomness = {
            state.logical_rank: snapshot(
                mapped.root_seed,
                [row for row in mapped.states if row.stream.rank == state.logical_rank],
            )
            for state in targets
        }
    info = replace(
        saved,
        request_id="conversion-" + identity,
        world_size=len(targets),
        configuration=request.configuration,
        plans=request.plans,
        conversion=provenance,
    )
    ranks = tuple(
        RankSave(
            state.logical_rank,
            checkpoint_size(
                state,
                metadata={
                    **info.rank_metadata(),
                    **(
                        {"randomness": target_randomness[state.logical_rank]}
                        if target_randomness
                        else {}
                    ),
                },
                weights=info.weights,
                frozen_state_id=info.frozen_state_id,
            ),
            state_identity(state),
            state.plan.program.digest,
            {name: plan.program.digest for name, plan in state.plans.items()},
            target_randomness.get(state.logical_rank),
        )
        for state in targets
    )
    chunks = conversion_tensors(targets, info.weights, info.frozen_state_id)
    staged = staging_bytes(chunks)
    files = 1 + 2 * len(chunks) + len(ranks)
    disk_bytes = staged + sum(rank.size_bytes for rank in ranks) + 4096 * files
    return PreparedConversion(
        request,
        ConversionPrepared(info, ranks, disk_bytes, files),
        sources,
        targets,
        defaults,
        scratch_bytes,
        staged,
        tuple(metadata),
    )


def materialize(prepared: PreparedConversion) -> ConversionWritten:
    from .preparation_files import file_digest, verified_file

    request, description = prepared.request, prepared.description
    info = description.info
    with ExitStack() as owned:
        sources = []
        for state, row in zip(prepared.sources, request.source.ranks, strict=True):
            stream = owned.enter_context(
                verified_file(row.archive, Path(request.files[row.archive.id]))
            )
            sources.append((state, stream))
        converted = owned.enter_context(
            convert_state(
                sources,
                prepared.targets,
                Path(request.directory),
                scratch_bytes=prepared.scratch_bytes,
                max_disk_bytes=prepared.staging_bytes,
                initialize_temporary=prepared.defaults,
                frozen_state_id=info.frozen_state_id,
                max_source_manifest_bytes=request.host_bytes // (256 * len(sources)),
                expected_metadata=prepared.source_metadata,
            )
        )
        if (
            converted.weights != info.weights
            or converted.frozen_state_id != info.frozen_state_id
            or mapping(converted.source_metadata, "source metadata")
            != mapping(prepared.source_metadata[0], "saved boundary")
        ):
            raise ValueError("checkpoint rank metadata differs from its generation")
        completed = []
        for rank in description.ranks:
            path = Path(request.directory) / f"rank-{rank.rank}.ssp"
            with path.open("xb") as stream:
                manifest = write_checkpoint(
                    converted.ranks[rank.rank],
                    stream,
                    metadata=rank.metadata(info),
                    chunk_bytes=prepared.scratch_bytes // 2,
                    weights=info.weights,
                    frozen_state_id=info.frozen_state_id,
                )
                if stream.tell() != rank.size_bytes:
                    raise ValueError("converted rank size differs from its admission")
            rank.confirm(info, manifest)
            with path.open("rb") as stream:
                sha256, size = file_digest(stream)
            if size != rank.size_bytes:
                raise ValueError("converted rank size changed before completion")
            completed.append(RankSaved(sha256, rank.confirmation(info)))
    return ConversionWritten(completed)
