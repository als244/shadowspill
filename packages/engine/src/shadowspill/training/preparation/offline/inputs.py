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
from shadowspill.training.preparation.offline.plans import LocalPlans
from shadowspill.training.preparation.planning.requests import PlanningRequest
from shadowspill_client.protocol.training import ResolvedTraining

T = TypeVar("T", bound=Artifact)

if TYPE_CHECKING:
    from shadowspill.training.preparation.offline.sweep import PlanningSweep


@dataclass(frozen=True)
class PlanningInputs:
    """A preparation's exported measurements, for planning without the service.

    ``PlanningInputs.load`` reads the archive `PreparationHandle.export`
    gives. `sweep` replans a grid of budgets and transfer rates; `plan` plans
    selected candidates into a proposal for the service, which still checks
    every capture, input coverage and joint memory admission. Neither needs a
    GPU, the model package or the service.

    Attributes
    ----------
    config : ResolvedTraining
        The preparation's resolved settings.
    sources : mapping of str to ProfiledProgram
        The measured Program of each capture template, by template ID;
        ``sources[key].profiles`` gives an editable copy of its profiles.
    communication : mapping of str to CommunicationInfo
        Each template's communication costs, by template ID.
    requests : mapping of str to PlanningRequest
        Every rank-local planning request, by request ID.
    candidates : mapping of str to tuple of str
        Each candidate's request IDs, by candidate ID.
    descriptions : mapping of str to mapping
        Each candidate's operation, batch, scheduled capacities, build case,
        microbatch count and ordering, by candidate ID.
    """

    config: ResolvedTraining
    sources: Mapping[str, ProfiledProgram]
    communication: Mapping[str, CommunicationInfo]
    requests: Mapping[str, PlanningRequest]
    candidates: Mapping[str, tuple[str, ...]]
    descriptions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> PlanningInputs:
        """Read an exported measurement archive.

        Each file in it is checked against the manifest's size and SHA-256
        before it is read.

        Parameters
        ----------
        path : str or pathlib.Path
            The archive `PreparationHandle.export` gives.

        Returns
        -------
        PlanningInputs
            The exported measurements.

        Raises
        ------
        ValueError
            If the archive is not a supported measurement export, or a file
            differs from the manifest.
        """
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
                manifest.get("format") != "shadowspill.operator_preparation-export"
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
        """Replan every exported candidate over a grid of budgets and transfer rates.

        Each point of the grid runs the ordinary planner again. A transfer
        scale multiplies the fetch and eviction rates, concurrent rates
        included, and leaves compute, latency and collective costs unchanged.
        The results are predictions: an infeasible point stays in the result
        with its reason, and no training runs.

        Parameters
        ----------
        device_gib : float or sequence of float, optional
            Device budgets per worker, in GiB, each positive. By default the
            preparation's.
        spill_gib : float or sequence of float, optional
            Spill budgets per worker, in GiB, each at least zero. By default
            the preparation's.
        transfer_scale : float or sequence of float, default 1.0
            Factors on the measured transfer rates, each positive.

        Returns
        -------
        PlanningSweep
            One row per grid point and candidate, with its predicted time and
            peaks; its ``write_report`` writes an HTML comparison.

        Raises
        ------
        ValueError
            If a list of values is empty, repeats a value, or holds a value
            out of range.
        """
        from shadowspill.training.preparation.offline.sweep import sweep

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
        """Plan exported candidates into a proposal for the service.

        Budgets, initialization, execution settings and the planner's search
        settings may differ from the preparation's; anything else needs a new
        export. A request with no feasible Plan is kept as rejected, and a
        candidate is selected only when every request it needs has a Plan.
        Nothing missing is measured or estimated.

        Parameters
        ----------
        config : ResolvedTraining
            The resolved settings to plan for, whose preparation reuses the
            exported one.
        profiles : mapping of str to Profiles, optional
            Edited profiles by template ID, instead of ``sources[key].profiles``.
            Only timings may change, marked ``duration_kind="estimated"`` in
            their metadata.
        communication : mapping of str to CommunicationInfo, optional
            Edited communication costs by template ID. Only rates, latencies
            and collective durations may change, marked
            ``timing_kind="estimated"`` in the metadata.
        candidates : sequence of str, optional
            The candidates to plan, without repeats. By default all.

        Returns
        -------
        LocalPlans
            The selected schedules, without executable code.

        Raises
        ------
        ValueError
            If ``config`` differs from the export beyond those settings, an
            edit names an unexported template or changes anything else,
            ``candidates`` names an unknown candidate or repeats one, a Plan
            lacks physical memory admission, or no candidate has every Plan
            it needs.
        """
        from shadowspill.training.preparation.offline.search import search

        return search(
            self,
            config=config,
            profiles=profiles,
            communication=communication,
            candidates=candidates,
        ).proposal()
