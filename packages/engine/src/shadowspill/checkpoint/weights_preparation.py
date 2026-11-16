"""Bounded CPU preparation of exported model weights for a fresh target layout."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.worker.checkpoint_records import RankSave
from shadowspill.training.worker.conversion_records import (
    ConversionPrepared,
    ConversionRequest,
    ConversionWritten,
    weights_conversion_info,
)
from shadowspill_client._records import mapping

from .compatibility import describe_family
from .exports import weights_size
from .identity import state_identity
from .preparation import plan_states
from .preparation_files import verified_file, write_ranks
from .staging import staging_bytes
from .state import PlanState
from .weights_conversion import _require_model, convert_weights, weights_tensors
from .weights_reading import read_weights


@dataclass(frozen=True)
class PreparedWeights:
    request: ConversionRequest
    description: ConversionPrepared
    sources: tuple[PlanState, ...]
    targets: tuple[PlanState, ...]
    scratch_bytes: int
    staging_bytes: int


def prepare(request: ConversionRequest) -> PreparedWeights:
    host_bytes = request.host_bytes
    scratch_bytes = min(64 << 20, host_bytes // 16)
    if scratch_bytes < 64 << 10:
        raise ValueError("weights conversion host envelope is too small")
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
    info = weights_conversion_info(request, len(targets))
    if len(sources) != saved.world_size or any(
        state_identity(state) != rank.state_sha256
        for state, rank in zip(sources, request.source.ranks, strict=True)
    ):
        raise ValueError("weights source Plans differ from saved rank ownership")
    _require_model(
        describe_family(sources, roles=("parameter", "buffer")),
        describe_family(targets, roles=("parameter", "buffer")),
    )
    for state, rank in zip(sources, request.source.ranks, strict=True):
        with (
            verified_file(rank.archive, Path(request.files[rank.archive.id])) as stream,
            read_weights(
                state,
                stream,
                chunk_bytes=scratch_bytes // 2,
                frozen_state_id=saved.frozen_state_id,
                max_manifest_bytes=host_bytes // (256 * len(sources)),
            ) as reader,
        ):
            if reader.manifest["weights"] != saved.weights or mapping(
                reader.manifest["metadata"], "weights metadata"
            ) != mapping(saved.rank_metadata(), "export provenance"):
                raise ValueError("weights rank differs from its export generation")
    ranks = tuple(
        RankSave(
            state.logical_rank,
            weights_size(
                state,
                metadata=info.rank_metadata(),
                weights=info.weights,
                frozen_state_id=info.frozen_state_id,
            ),
            state_identity(state),
            state.plan.program.digest,
            {name: plan.program.digest for name, plan in state.plans.items()},
            kind="weights",
        )
        for state in targets
    )
    chunks = weights_tensors(targets, info.frozen_state_id)
    staged = staging_bytes(chunks)
    files = 1 + 2 * len(chunks) + len(ranks)
    disk_bytes = staged + sum(rank.size_bytes for rank in ranks) + 4096 * files
    return PreparedWeights(
        request,
        ConversionPrepared(info, ranks, disk_bytes, files),
        sources,
        targets,
        scratch_bytes,
        staged,
    )


def materialize(prepared: PreparedWeights) -> ConversionWritten:
    request, description = prepared.request, prepared.description
    info = description.info
    with ExitStack() as owned:
        sources = [
            (
                state,
                owned.enter_context(
                    verified_file(row.archive, Path(request.files[row.archive.id]))
                ),
            )
            for state, row in zip(prepared.sources, request.source.ranks, strict=True)
        ]
        converted = owned.enter_context(
            convert_weights(
                sources,
                prepared.targets,
                Path(request.directory),
                scratch_bytes=prepared.scratch_bytes,
                max_disk_bytes=prepared.staging_bytes,
                frozen_state_id=info.frozen_state_id,
                max_source_manifest_bytes=request.host_bytes // (256 * len(sources)),
                expected_metadata=[request.source.info.rank_metadata()] * len(sources),
            )
        )
        if converted.weights != info.weights:
            raise ValueError("weights rank selection differs from its generation")
        return write_ranks(
            converted,
            description,
            Path(request.directory),
            chunk_bytes=prepared.scratch_bytes // 2,
        )
