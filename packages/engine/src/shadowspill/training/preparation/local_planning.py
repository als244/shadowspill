"""Use exported cases with the ordinary CPU planner, without workload packages."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

from shadowspill.artifacts.archive import Artifact
from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.profiles import Profiles
from shadowspill.communication.info import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planner import SearchOptions
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill_client.protocol.training import ResolvedTraining

from .cost_evidence import communication_edits, profile_edits
from .identity import without_initialization
from .local_plans import LocalPlans
from .planning_requests import PlanningRequest
from .templates import instantiate_program

T = TypeVar("T", bound=Artifact)


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
        normalized = replace(
            config,
            memory=self.config.memory,
            execution=self.config.execution,
            origins=self.config.origins,
            preparation=replace(
                config.preparation,
                reuse=self.config.preparation.reuse,
                origins=self.config.preparation.origins,
                planning=replace(
                    config.preparation.planning,
                    search=self.config.preparation.planning.search,
                    workers=self.config.preparation.planning.workers,
                ),
            ),
        )
        if without_initialization(normalized) != without_initialization(self.config):
            raise ValueError("local planning settings require different exported cases")
        supplied = {} if profiles is None else profiles
        comm = {} if communication is None else communication
        if (
            not supplied.keys() <= self.sources.keys()
            or not comm.keys() <= self.sources.keys()
        ):
            raise ValueError("edited costs refer to an unexported capture")
        costs = {
            key: supplied.get(key, source.profiles)
            for key, source in self.sources.items()
        }
        communication = {
            key: comm.get(key, value) for key, value in self.communication.items()
        }
        for key, source in self.sources.items():
            profile_edits(source.profiles, costs[key])
            communication_edits(self.communication[key], communication[key])
        selected = tuple(self.candidates if candidates is None else candidates)
        if not set(selected) <= self.candidates.keys() or len(set(selected)) != len(
            selected
        ):
            raise ValueError("local planning requires distinct exported candidates")
        required = {key for candidate in selected for key in self.candidates[candidate]}
        settings = config.preparation.planning
        assert settings.search is not None and settings.workers is not None
        search = replace(
            SearchOptions.from_dict(settings.search, "search"), workers=settings.workers
        )
        plans: dict[str, Any] = {}
        rejected = {}
        for key in sorted(required):
            request = self.requests[key]
            template = request.use.template
            program = instantiate_program(request.use, self.sources[template].program)
            try:
                plan = plan_program(
                    apply_profiles(program, costs[template]),
                    memory_budgets=MemoryBudgets(
                        int(config.memory.device_gib * (1 << 30)),
                        int(config.memory.spill_gib * (1 << 30)),
                    ),
                    communication_info=communication[template],
                    search_options=search,
                    ordering=request.ordering,
                )
                if plan.simulation_only_reasons:
                    raise ValueError("local Plan lacks physical memory admission")
                plans[key] = plan.to_dict()
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                rejected[key] = str(error)
        selected = tuple(
            key for key in selected if set(self.candidates[key]) <= plans.keys()
        )
        needed = {key for candidate in selected for key in self.candidates[candidate]}
        templates = {self.requests[key].use.template for key in needed}
        return LocalPlans(
            config,
            {key: costs[key] for key in templates},
            {key: plans[key] for key in needed},
            selected,
            rejected,
        )
