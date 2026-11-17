"""Use exported cases with the ordinary CPU planner, without workload packages."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar, cast

from shadowspill.artifacts.archive import Artifact
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.profiles import Profiles
from shadowspill.communication.info import CommunicationInfo
from shadowspill_client.protocol.training import ResolvedTraining

from .local_plans import LocalPlans
from .planning_requests import PlanningRequest

T = TypeVar("T", bound=Artifact)

if TYPE_CHECKING:
    from .local_sweep import PlanningSweep


@dataclass(frozen=True)
class PlanningInputs:
    """All exported viable captures and operation variants, with original costs.

    Edit copies from ``source.profiles`` and CommunicationInfo using ``replace``.
    ``plan`` returns selected schedules; server submission still rechecks every
    capture, coverage witness and shared-state memory requirement.
    """

    config: ResolvedTraining
    sources: Mapping[str, ProfiledProgram]
    communication: Mapping[str, CommunicationInfo]
    requests: Mapping[str, PlanningRequest]
    candidates: Mapping[str, tuple[str, ...]]
    descriptions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> PlanningInputs:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            files = {row.filename: row for row in entries}
            if (
                len(files) != len(entries)
                or len(files) > 16384
                or "manifest.json" not in files
                or files["manifest.json"].file_size > 16 << 20
                or any(row.compress_type != zipfile.ZIP_STORED for row in entries)
            ):
                raise ValueError("invalid measurement export framing")
            manifest = json.loads(archive.read("manifest.json"))
            if (
                manifest.get("format") != "shadowspill.preparation-export"
                or type(manifest.get("version")) is not int
                or manifest["version"] != 1
                or set(files) != {"manifest.json", *manifest["files"]}
            ):
                raise ValueError("unsupported measurement export")

            def load(key: str, stage: str, kind: type[T]) -> T:
                name = manifest["captures"][key]["files"][stage]
                expected = manifest["files"][name]
                if (
                    name != f"{stage}/{key}.ssp"
                    or expected["size_bytes"] != files[name].file_size
                ):
                    raise ValueError("measurement file differs from its capture")
                with archive.open(name) as stream:
                    checksum = hashlib.sha256()
                    while chunk := stream.read(64 << 10):
                        checksum.update(chunk)
                    if checksum.hexdigest() != expected["sha256"]:
                        raise ValueError("measurement file failed its content checksum")
                    stream.seek(0)
                    return kind.load(
                        cast(BinaryIO, stream),
                        max_metadata_bytes=256 << 20,
                        max_payload_bytes=8 << 30,
                    )

            requests = {
                key: PlanningRequest.from_dict(row)
                for key, row in manifest["planning"]["requests"].items()
            }
            if any(key != row.id for key, row in requests.items()):
                raise ValueError("exported request has an inconsistent identity")
            sources, communication = {}, {}
            for key in {row.use.template for row in requests.values()}:
                sources[key] = load(key, "calibrated", ProfiledProgram)
                communication[key] = load(key, "communication", CommunicationInfo)
            return cls(
                ResolvedTraining.from_dict(manifest["configuration"]),
                sources,
                communication,
                requests,
                {
                    key: tuple(value)
                    for key, value in manifest["planning"]["candidates"].items()
                },
                manifest["planning"].get("descriptions", {}),
            )

    def sweep(
        self,
        *,
        device_gib: float | Sequence[float] | None = None,
        spill_gib: float | Sequence[float] | None = None,
        transfer_scale: float | Sequence[float] = 1.0,
    ) -> PlanningSweep:
        """Replan a budget/bandwidth grid using the exported measurements.

        Omitted budgets inherit their measured setup. Transfer scale multiplies
        fetch/evict bandwidths, including concurrent rates; compute, latency and
        collective costs stay unchanged. All results are predictions. Empty
        feasible sets remain inspectable, and no training run is started.
        """
        from .local_sweep import sweep

        return sweep(
            self,
            device_gib=device_gib,
            spill_gib=spill_gib,
            transfer_scale=transfer_scale,
        )

    def plan(
        self,
        *,
        config: ResolvedTraining,
        profiles: Mapping[str, Profiles] | None = None,
        communication: Mapping[str, CommunicationInfo] | None = None,
        candidates: Sequence[str] | None = None,
    ) -> LocalPlans:
        """Plan selected exported alternatives; never invent missing measurements.

        Budgets, initialization, execution settings and planner search controls
        may change. New build forms/operations/counts need another export. Known
        infeasible searches are retained as diagnostics; no runtime fallback is
        produced. The server makes the final joint admission decision.
        """
        from .local_search import search

        return search(
            self,
            config=config,
            profiles=profiles,
            communication=communication,
            candidates=candidates,
        ).proposal()
