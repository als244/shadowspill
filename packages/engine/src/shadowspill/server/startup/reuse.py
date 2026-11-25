"""Authenticate measured-run provenance before staging preparation artifacts."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.records.runs import RunCatalog
from shadowspill.training.preparation.artifacts.measurement_reuse import (
    stage_measurements,
)
from shadowspill.training.preparation.phases.job import (
    JOB_NAME,
    PreparationJob,
)
from shadowspill_client.protocol.training import ResolvedTraining


def stage_preparation_reuse(
    artifacts: ArtifactStore,
    catalog: RunCatalog,
    config: ResolvedTraining,
    root: Path,
    *,
    packages: Sequence[str],
    context: Mapping[str, Any],
    max_disk_bytes: int,
    max_files: int,
    host_limit_bytes: int | None = None,
) -> Mapping[str, Any] | None:
    """The source must be this authenticated owner's completed service job."""
    reuse = config.preparation.reuse
    if not reuse:
        return None
    source = reuse.get("measurements")
    if (
        set(reuse) != {"measurements"}
        or not isinstance(source, Mapping)
        or set(source) != {"session_id", "run_id"}
        or any(not isinstance(source[key], str) for key in source)
    ):
        raise ValueError("reuse requires a measured preparation session_id and run_id")
    # Values are validated again by RunIdentity; no owner can be supplied here.
    identity = RunIdentity(
        artifacts.owner_id, str(source["session_id"]), str(source["run_id"])
    )
    info = catalog.get(identity)
    if info is None or info.state != "prepared" or info.preparation_export is None:
        raise ValueError(
            "reuse source is not a completed preparation owned by this caller"
        )
    requested = PreparationJob(tuple(packages), config, host_limit_bytes)
    _, inventory = requested.resolve()
    # Never turn an uploaded ZIP into executable evidence by accepting its label.
    # The durable run record pins this exact server-produced export.
    with artifacts.open(info.preparation_export) as stream:
        root.mkdir()
        requested.save(root / JOB_NAME)
        return stage_measurements(
            stream,
            info.preparation_export,
            root,
            inventory,
            context=context,
            max_disk_bytes=max_disk_bytes,
            max_files=max_files,
        )
