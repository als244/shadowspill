"""Content-addressed persistence for complete PressureFit selections."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from shadowspill.ir import (
    MemorySchedule,
    Program,
    RecomputationSelection,
    ResidencySpec,
)
from shadowspill.planner.artifact_store import ArtifactStore
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationConfig, simulate
from shadowspill.simulator.indexed import (
    index_simulation_template,
    simulate_template,
)

from .admission import AdmissionFacts
from .admission.indexed import (
    encode_schedule,
    evaluate_schedule_admission,
    index_admission_facts,
)
from .diagnostics import PressureFitDiagnostics
from .diagnostics.json import without_measurements
from .plan import pressurefit
from .request import PressureFitOptions
from .result import PressureFitResult
from .serialization import _resident_slice_from_value

_SCHEMA = artifact_schema("pressurefit_selection")


class _ArtifactRecorder(Protocol):
    def __call__(
        self,
        *,
        category: str,
        kind: str,
        digest: str | None,
        path: str | Path,
        access: str,
        schema: str | None,
        dependencies: tuple[str, ...] = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PlanLookup:
    """One selected plan, and whether it was read back or planned now."""

    result: PressureFitResult
    #: True when the plan was read back rather than planned now.
    from_store: bool


class PlanStore:
    """Selected plans on disk, keyed by the request that produced them.

    The key excludes worker concurrency, which changes how the search is
    scheduled but not which plan it may answer with.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        read_enabled: bool = True,
        write_enabled: bool = True,
        overwrite: bool = False,
        artifact_recorder: _ArtifactRecorder | None = None,
    ) -> None:
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / ".cache" / "shadowspill" / "recomputation"
        )
        self.read_enabled = read_enabled
        self.write_enabled = write_enabled
        self.overwrite = overwrite
        self.artifact_recorder = artifact_recorder

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def resolve(
        self,
        program: Program,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions | None = None,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> PlanLookup:
        """Return the stored planned program when it still matches, or plan."""

        selected_options = options or PressureFitOptions()
        key = _key(
            program,
            initial_residency,
            final_residency,
            config,
            selected_options,
            admission,
            placement,
        )
        cached = (
            self._read(
                key,
                program,
                initial_residency,
                final_residency,
                config,
                selected_options,
                admission,
            )
            if self.read_enabled
            else None
        )
        if cached is not None:
            return PlanLookup(cached, True)
        result = pressurefit(
            program,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            options=selected_options,
            admission=admission,
            placement=placement,
            progress=progress,
        )
        self._write(key, result, admission)
        return PlanLookup(result, False)

    def _read(
        self,
        key: str,
        program: Program,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        options: PressureFitOptions,
        admission: AdmissionFacts | None,
    ) -> PressureFitResult | None:
        path = self.path(key)
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"planned program {path} cannot be read") from exc
        if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
            raise ValueError(f"planned program {path} has an invalid schema")
        if value.get("key_digest") != key:
            raise ValueError(f"planned program {path} has the wrong identity")
        if value.get("program_digest") != program.digest:
            raise ValueError(f"planned program {path} has the wrong Program")
        expected_boundary = {
            "initial_residency": [item.to_dict() for item in initial_residency],
            "final_residency": [item.to_dict() for item in final_residency],
            "simulation": {
                "devices": [asdict(item) for item in config.devices],
                "spill_capacity_bytes": config.spill_capacity_bytes,
            },
            "options": options.to_dict(),
            "admission_digest": admission.digest if admission is not None else None,
        }
        normalized_boundary = json.loads(
            json.dumps(expected_boundary, sort_keys=True, separators=(",", ":"))
        )
        for field, expected in normalized_boundary.items():
            if value.get(field) != expected:
                raise ValueError(f"planned program {path} has stale {field} evidence")
        schedule = MemorySchedule.from_dict(value.get("schedule"))
        raw_selections = value.get("selections")
        if not isinstance(raw_selections, list):
            raise ValueError(f"planned program {path} has invalid selections")
        selections = tuple(
            RecomputationSelection.from_value(item, f"cache.selections[{index}]")
            for index, item in enumerate(raw_selections)
        )
        schedule.validate(program, selections)
        if admission is None:
            simulation = simulate(
                program,
                schedule,
                selections=selections,
                config=config,
            )
        else:
            template = index_simulation_template(program, selections, config)
            indexed_admission = index_admission_facts(admission, template)
            physical = evaluate_schedule_admission(
                template,
                indexed_admission,
                encode_schedule(schedule, template),
            )
            simulation = simulate_template(
                template,
                schedule,
                admission=physical.simulation_admission,
            )
        diagnostics = _diagnostics_from_value(value.get("diagnostics"), path)
        if diagnostics.selected_makespan_ns != simulation.makespan_ns:
            raise ValueError(f"planned program {path} has stale simulator evidence")
        result = PressureFitResult(
            program=program,
            options=options,
            initial_residency=initial_residency,
            final_residency=final_residency,
            simulation_config=config,
            schedule=schedule,
            selections=selections,
            simulation=simulation,
            diagnostics=diagnostics,
            resident_slice=_resident_slice_from_value(
                value.get("resident_slice"), f"{path}.resident_slice"
            ),
            admission_facts=admission,
        )
        self._record(key, program.digest, path, "read")
        return result

    def _write(
        self,
        key: str,
        result: PressureFitResult,
        admission: AdmissionFacts | None,
    ) -> None:
        if not self.write_enabled:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "schema": _SCHEMA,
            "key_digest": key,
            "program_digest": result.program.digest,
            "initial_residency": [item.to_dict() for item in result.initial_residency],
            "final_residency": [item.to_dict() for item in result.final_residency],
            "simulation": {
                "devices": [asdict(item) for item in result.simulation_config.devices],
                "spill_capacity_bytes": result.simulation_config.spill_capacity_bytes,
            },
            "options": result.options.to_dict(),
            "admission_digest": admission.digest if admission is not None else None,
            "schedule": result.schedule.to_dict(),
            "selections": [item.to_dict() for item in result.selections],
            "diagnostics": result.diagnostics.to_dict(),
            "resident_slice": result.resident_slice.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if path.exists() and not self.overwrite:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(f"planned program {path} cannot be read") from exc
            try:
                existing_payload = json.loads(existing)
            except json.JSONDecodeError as exc:
                raise ValueError(f"planned program {path} cannot be read") from exc
            if without_measurements(existing_payload) != without_measurements(payload):
                raise ValueError(
                    "fresh PressureFit output differs from the stored planned program; "
                    "use overwrite_plan=True or a new implementation_revision: "
                    f"{path}"
                )
            self._record(key, result.program.digest, path, "matched")
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        self._record(key, result.program.digest, path, "write")

    def _record(
        self,
        key: str,
        program_digest: str,
        path: Path,
        access: str,
    ) -> None:
        if self.artifact_recorder is None:
            return
        self.artifact_recorder(
            category="pressurefit",
            kind="selection",
            digest=key,
            path=path,
            access=access,
            schema=_SCHEMA,
            dependencies=(program_digest,),
        )


def _key(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions,
    admission: AdmissionFacts | None,
    placement: AdmissionFacts | None = None,
) -> str:
    payload = {
        "schema": _SCHEMA,
        "program_digest": program.digest,
        "initial_residency": [item.to_dict() for item in initial_residency],
        "final_residency": [item.to_dict() for item in final_residency],
        "simulation": {
            "devices": [asdict(device) for device in config.devices],
            "spill_capacity_bytes": config.spill_capacity_bytes,
        },
        "options": options.to_dict(),
        "admission_digest": admission.digest if admission is not None else None,
        # Part of the identity: the search measures layouts against this
        # topology, so the same program under a different pool is a
        # different question and must not read a cached answer.
        "placement_digest": placement.digest if placement is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _diagnostics_from_value(value: object, path: Path) -> PressureFitDiagnostics:
    try:
        return PressureFitDiagnostics.from_value(value, "cache.diagnostics")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"planned program {path} has invalid diagnostics") from exc


__all__ = ["PlanLookup", "PlanStore"]


def open_plan_store(artifact_store: ArtifactStore) -> PlanStore:
    """Open the plan store one artifact-store policy implies."""

    return PlanStore(
        artifact_store.pressurefit_selections,
        read_enabled=artifact_store.read_enabled,
        write_enabled=artifact_store.write_enabled,
        overwrite=artifact_store.overwrite_plan,
        artifact_recorder=artifact_store.record,
    )


def resolve_plan(
    artifact_store: ArtifactStore,
    plans: PlanStore,
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PlanLookup:
    """Resolve one plan, planning only when the store does not have it.

    The request and its Program are archived first, so a plan on disk can
    always be traced back to what was asked for.
    """

    artifact_store.archive_program(program)
    selected_options = options or PressureFitOptions()
    artifact_store.archive_pressurefit_request(
        {
            "schema": artifact_schema("pressurefit_request"),
            "program_digest": program.digest,
            "initial_residency": [item.to_dict() for item in initial_residency],
            "final_residency": [item.to_dict() for item in final_residency],
            "simulation": {
                "devices": [asdict(item) for item in config.devices],
                "spill_capacity_bytes": config.spill_capacity_bytes,
            },
            "options": {
                "initial_placement": selected_options.initial_placement.value,
                "residency_strategies": list(selected_options.residency_strategies),
                "fetch_rules": list(selected_options.fetch_rules),
                "evaluate_coalesced": selected_options.evaluate_coalesced,
                "max_repair_attempts": selected_options.max_repair_attempts,
                "workers": selected_options.workers,
                "deterministic": selected_options.deterministic,
            },
            "admission": None if admission is None else admission.to_dict(),
        }
    )
    return plans.resolve(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        options=selected_options,
        admission=admission,
        placement=placement,
        progress=progress,
    )
