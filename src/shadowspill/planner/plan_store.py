"""Content-addressed persistence for complete PressureFit selections."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from shadowspill.ir import (
    MemorySchedule,
    Program,
    ResidencySpec,
    TaskAlternativeChoice,
)
from shadowspill.planner.artifact_store import ArtifactStore, digest_directory
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
from .recomputation import ShareValue, resolution_options_or_default
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
    scheduled but not which plan it may answer with, and names the resolution
    options the plan was searched over, so a plan found under one set is never
    read back for another, whatever the library's default is that day.
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
        return digest_directory(self.root, key) / "selection.json"

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
        resolution_options: Sequence[ShareValue] | None = None,
        incumbent: PressureFitResult | None = None,
    ) -> PlanLookup:
        """Return the stored planned program when it still matches, or plan.

        The plan to beat is provenance, not identity. A search handed one
        answers with it unless it does better, but what it answers is still
        a plan for this request, and the store's promise is that a request
        reads back the plan its search chose: a run that replans the budget
        it is about to execute, without the sweep's plan in hand, has to get
        the sweep's answer. So the key leaves it out, and the record says
        which plan the search was handed.

        The promise runs the other way too: a request handed a plan never
        answers worse than it. A store holding a plan that the plan in hand
        claims to beat searches again with it, and keeps whichever answer is
        better for this request, so a store filled before the plan existed
        improves rather than shadowing it.
        """

        selected_options = options or PressureFitOptions()
        chosen = resolution_options_or_default(resolution_options)
        key = _key(
            program,
            initial_residency,
            final_residency,
            config,
            selected_options,
            admission,
            placement,
            chosen,
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
                chosen,
            )
            if self.read_enabled
            else None
        )
        if cached is not None and not _claims_to_beat(incumbent, cached):
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
            resolution_options=chosen,
            incumbent=incumbent,
        )
        if cached is not None:
            if result.simulation.makespan_ns >= cached.simulation.makespan_ns:
                return PlanLookup(cached, True)
            self._write(key, result, admission, chosen, incumbent, improve=True)
            return PlanLookup(result, False)
        self._write(key, result, admission, chosen, incumbent)
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
        resolution_options: tuple[Fraction, ...],
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
            **_resolution_options_field(resolution_options),
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
            TaskAlternativeChoice.from_value(item, f"cache.selections[{index}]")
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
        resolution_options: tuple[Fraction, ...],
        incumbent: PressureFitResult | None = None,
        improve: bool = False,
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
            **_resolution_options_field(resolution_options),
            **_incumbent_field(incumbent),
            "schedule": result.schedule.to_dict(),
            "selections": [item.to_dict() for item in result.selections],
            "diagnostics": result.diagnostics.to_dict(),
            "resident_slice": result.resident_slice.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if path.exists() and not self.overwrite and not improve:
            try:
                existing = path.read_text()
            except OSError as exc:
                raise ValueError(f"planned program {path} cannot be read") from exc
            try:
                existing_payload = json.loads(existing)
            except json.JSONDecodeError as exc:
                raise ValueError(f"planned program {path} cannot be read") from exc
            # Provenance is not the answer: the same plan found with or
            # without a plan in hand is the same plan.
            if _without_provenance(existing_payload) != _without_provenance(payload):
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
        self._record(
            key, result.program.digest, path, "improved" if improve else "write"
        )

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
    placement: AdmissionFacts | None,
    resolution_options: tuple[Fraction, ...],
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
        # The resolution options are part of the question: a plan searched
        # over one set is not the answer for another. The plan to beat is
        # not: see PlanStore.resolve.
        **_resolution_options_field(resolution_options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _claims_to_beat(
    incumbent: PressureFitResult | None, stored: PressureFitResult
) -> bool:
    """Whether the plan in hand says it is faster than the plan on record.

    Its claim is the makespan it was found with, which may come from another
    calibration of the same machine; the search settles it under this one.
    """

    return (
        incumbent is not None
        and incumbent.simulation.makespan_ns < stored.simulation.makespan_ns
    )


def _without_provenance(payload: object) -> object:
    """A stored record as an answer: without measurements or the plan to beat."""

    value = without_measurements(payload)
    if isinstance(value, dict):
        return {name: item for name, item in value.items() if name != "incumbent"}
    return value


def _resolution_options_field(
    resolution_options: tuple[Fraction, ...],
) -> dict[str, list[str]]:
    """The options as a record field: exact fractions, sorted, as strings."""

    return {"resolution_options": [str(share) for share in resolution_options]}


def _incumbent_field(incumbent: PressureFitResult | None) -> dict[str, object]:
    """The plan to beat as a record field: which resolution, which schedule.

    Provenance for the request and the stored plan; never part of a key.
    """

    if incumbent is None:
        return {"incumbent": None}
    return {
        "incumbent": {
            "selections": [item.to_dict() for item in incumbent.selections],
            "schedule_digest": incumbent.schedule.digest,
        }
    }


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
    resolution_options: Sequence[ShareValue] | None = None,
    incumbent: PressureFitResult | None = None,
) -> PlanLookup:
    """Resolve one plan, planning only when the store does not have it.

    The request and its Program are archived first, so a plan on disk can
    always be traced back to what was asked for, the plan to beat included.
    """

    artifact_store.archive_program(program)
    selected_options = options or PressureFitOptions()
    chosen = resolution_options_or_default(resolution_options)
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
            **_resolution_options_field(chosen),
            **_incumbent_field(incumbent),
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
        resolution_options=chosen,
        incumbent=incumbent,
    )
