"""Publish CPU-prepared state without framework imports in the controller."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from shadowspill.artifacts._values import digest
from shadowspill.orchestration.launch.connected import ConnectedProcess
from shadowspill.orchestration.launch.publication import OutputWriter
from shadowspill.training.session.checkpoint import (
    CHECKPOINT_METADATA_BYTES,
    ResolvedCheckpoint,
)
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.checkpoint_records import (
    RankSave,
    RankSaved,
    checkpoint_id,
)
from shadowspill.training.worker.conversion_records import (
    ConversionCommand,
    ConversionFailed,
    ConversionPrepared,
    ConversionRequest,
    ConversionWritten,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import CheckpointConversion
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining

from .artifacts import ArtifactStore
from .checkpoint_restore import CheckpointResolver
from .checkpoints import publish_checkpoint


def _description(
    request: ConversionRequest, value: object, world_size: int
) -> ConversionPrepared:
    if isinstance(value, ConversionFailed):
        raise ValueError("checkpoint conversion failed: " + value.message)
    if not isinstance(value, ConversionPrepared):
        raise RuntimeError("checkpoint preparation ended without a description")
    conversion = value.info.conversion
    if conversion is None:
        raise ValueError("checkpoint conversion has no immutable provenance")
    expected = CheckpointConversion(
        request.checkpoint, request.source.info.plans, conversion.logical_state_sha256
    )
    identity = digest(
        {
            "conversion": expected.to_dict(),
            "plans": request.plans.to_dict(),
            "configuration": request.configuration.to_dict(),
        }
    )
    if conversion.randomness is not None:
        if (
            conversion.randomness.get("version") != 1
            or conversion.randomness.get("restore_id") != identity
            or any(
                rank.randomness is None
                or rank.randomness.get("root_seed")
                != conversion.randomness.get("root_seed")
                for rank in value.ranks
            )
        ):
            raise ValueError("RNG conversion differs from its stable restore identity")
        expected = replace(expected, randomness=conversion.randomness)
    elif any(rank.randomness is not None for rank in value.ranks):
        raise ValueError("converted random streams require mapping provenance")
    if value.info != replace(
        request.source.info,
        request_id="conversion-" + identity,
        world_size=world_size,
        configuration=request.configuration,
        plans=request.plans,
        conversion=expected,
    ):
        raise ValueError("checkpoint conversion changed its source boundary or target")
    if value.temporary_bytes < sum(
        rank.size_bytes for rank in value.ranks
    ) or value.files < len(value.ranks):
        raise ValueError("checkpoint preparation underdeclared its target files")
    return value


def prepare_checkpoint(
    artifacts: ArtifactStore,
    source: ResolvedCheckpoint,
    child: ConnectedProcess,
    *,
    plans: ArtifactRef,
    configuration: ArtifactRef,
    config: ResolvedTraining,
    world_size: int,
    host_bytes: int,
    root: Path,
    memory: BufferBudget,
    disk: BufferBudget,
    max_disk_bytes: int,
    max_files: int,
) -> ResolvedCheckpoint:
    """Admit temporary and final bytes before asking the isolated child to write.

    LocalCreation owns the host-only process lease and capped logs. Artifact files
    remain pinned throughout its lifetime; paths here are private grants chosen
    by storage, never public request fields. No source rank payload is copied.
    Target private archives plus conversion scratch are admitted explicitly, then
    the existing publisher copies and commits the complete verified generation.
    """
    with (
        memory.reserve(CHECKPOINT_METADATA_BYTES + (128 << 10), timeout=0),
        TemporaryDirectory(prefix="conversion-", dir=root) as directory,
        ExitStack() as owned,
    ):
        files = {}
        for reference in (
            source.manifest.info.plans,
            plans,
            *(row.archive for row in source.manifest.ranks),
        ):
            path = owned.enter_context(artifacts.local_path(reference))
            files[reference.id] = str(path)
        # Stop the child before source pins or its output directory can retire.
        owned.callback(child.close)
        request = ConversionRequest(
            source.reference,
            source.manifest,
            plans,
            configuration,
            files,
            directory,
            host_bytes,
        )
        write_message(child.stream, request)
        description = _description(request, read_message(child.stream), world_size)
        child.ready()
        resolver = CheckpointResolver(
            artifacts,
            session_id=description.info.session_id,
            world_size=world_size,
            configuration=config,
        )
        existing = artifacts.get(checkpoint_id(description.info))
        if existing is not None:
            result = resolver.resolve(existing)
            if result.manifest.info != description.info:
                raise ValueError("converted checkpoint has conflicting provenance")
            write_message(child.stream, ConversionCommand("close"))
            if child.process.wait() != 0:
                raise RuntimeError("checkpoint preparation failed to close")
            return result
        if (
            description.files > max_files
            or description.temporary_bytes > max_disk_bytes
        ):
            raise ValueError(
                "checkpoint conversion exceeds preparation disk/file bounds"
            )
        with disk.reserve(description.temporary_bytes, timeout=0):
            written: ConversionWritten | None = None

            def write(rank: RankSave, writer: OutputWriter) -> RankSaved:
                nonlocal written
                if written is None:
                    # The publisher has already reserved every final rank and
                    # the global manifest before its first producer is invoked.
                    write_message(child.stream, ConversionCommand("write"))
                    completed = read_message(child.stream)
                    if isinstance(completed, ConversionFailed):
                        raise ValueError(
                            "checkpoint conversion failed: " + completed.message
                        )
                    if not isinstance(completed, ConversionWritten) or len(
                        completed.ranks
                    ) != len(description.ranks):
                        raise RuntimeError(
                            "checkpoint preparation ended without every rank"
                        )
                    if child.process.wait() != 0:
                        raise RuntimeError(
                            "checkpoint preparation did not exit successfully"
                        )
                    written = completed
                saved = written.ranks[rank.rank]
                rank.confirm(description.info, saved.manifest)
                with (Path(directory) / f"rank-{rank.rank}.ssp").open("rb") as stream:
                    while block := stream.read(64 << 10):
                        writer.write(memoryview(block))
                return saved

            reference = publish_checkpoint(
                artifacts, description.info, description.ranks, write
            )
        return resolver.resolve(reference)
