"""Content-addressed persistence for the plans a search answers with."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.ir import (
    MemorySchedule,
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationConfig, SimulationInfeasibleError
from shadowspill.store import (
    CONTRIBUTE,
    ArtifactRecorder,
    ArtifactStore,
    StorePolicy,
    atomic_text,
    digest_directory,
)

from .admission import AdmissionFacts
from .admission.layout.model import FixedLayoutAdmission
from .diagnostics import PlanningDiagnostics
from .diagnostics.json import without_measurements
from .result import ProgramPlanResult
from .search import (
    SearchAlgorithm,
    SearchOptions,
    answer_no_worse_than,
)
from .serialization import (
    _fixed_layout_from_value,
    _resident_slice_from_value,
    _simulation_admission_from_value,
    _simulation_result_from_value,
)

_SCHEMA = artifact_schema("plan_selection")


@dataclass(frozen=True, slots=True)
class PlanLookup:
    """One selected plan, whether it was read back or planned now, and what
    the store holds beside it."""

    result: ProgramPlanResult
    #: True when the plan was read back rather than planned now.
    from_store: bool
    #: The key the plan is filed under, which `PlanStore.certify` writes beside.
    key: str = ""
    #: The fixed-layout certificate read back with the plan, when one is stored.
    certificate: FixedLayoutAdmission | None = None


@dataclass(frozen=True, slots=True)
class _Verdict:
    """A search's recorded refusal: no plan, and why."""

    outcome: str
    error: str
    kind: str | None
    message: str


#: The refusals a store records. A `RuntimeError` such as a preparation failure
#: is not one of them: it may be the environment's, and is not recorded.
_VERDICTS = (PlanInfeasibleError, SimulationInfeasibleError, PlanSearchExhaustedError)


class PlanStore:
    """Selected plans on disk, keyed by the request that produced them.

    The key covers the whole question: the program, where it starts and ends,
    the machine, the planner's own options, which search ran, and what that
    search was told. Change any of them and it is a different question with a
    different answer -- a plan searched over one candidate space is never
    read back for another, whatever the library's defaults are that day.

    Worker count is not part of it. Two runs at different worker counts ask
    the same question and must read back the same answer, even though the
    shared placement gate means which worker places first can decide which
    candidates are ever measured; the stored record says how many workers
    produced the plan.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StorePolicy = CONTRIBUTE,
        artifact_recorder: ArtifactRecorder | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.policy = policy
        self.artifact_recorder = artifact_recorder

    def path(self, key: str) -> Path:
        return digest_directory(self.root, key) / "selection.json"

    def resolve(
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        search_options: SearchOptions | None = None,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
        incumbent: ProgramPlanResult | None = None,
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

        chosen = search_options if search_options is not None else SearchOptions()
        algorithm = chosen.resolved_algorithm
        key = _key(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            placement,
            chosen,
        )
        stored = (
            self._read(
                key,
                program,
                initial_residency,
                final_residency,
                config,
                admission,
                algorithm,
                chosen,
            )
            if self.policy.read_enabled
            else None
        )
        if isinstance(stored, _Verdict):
            # A verdict answers the question it was recorded for. With a plan
            # in hand the search may answer with that plan instead, so the
            # question is asked again.
            if incumbent is None:
                raise _verdict_error(stored)
            stored = None
        if stored is not None and not _claims_to_beat(incumbent, stored.result):
            return stored
        if stored is None:
            self.policy.refuse_miss("plan", key)
        try:
            result = answer_no_worse_than(
                algorithm(
                    program,
                    initial_residency=initial_residency,
                    final_residency=final_residency,
                    config=config,
                    generic=chosen.generic,
                    workers=chosen.workers,
                    admission=admission,
                    placement=placement,
                    progress=progress,
                    incumbent=incumbent,
                ),
                incumbent=incumbent,
                config=config,
                placement=placement,
            )
        except _VERDICTS as error:
            if incumbent is None:
                self._write_verdict(
                    key,
                    program,
                    initial_residency,
                    final_residency,
                    config,
                    admission,
                    algorithm,
                    chosen,
                    error,
                )
            raise
        if stored is not None:
            if result.simulation.makespan_ns >= stored.result.simulation.makespan_ns:
                return stored
            self._write(
                key, result, admission, algorithm, chosen, incumbent, improve=True
            )
            return PlanLookup(result, False, key)
        self._write(key, result, admission, algorithm, chosen, incumbent)
        return PlanLookup(result, False, key)

    def certify(self, lookup: PlanLookup, admission: FixedLayoutAdmission) -> None:
        """Write a plan's fixed-layout certificate beside it.

        The certificate is a function of the plan and the facts it was
        certified against, which its layout names by digest, so a later read
        serves it without placing or simulating again.
        """

        if not self.policy.write_enabled or not lookup.key:
            return
        path = self.path(lookup.key)
        payload = self._payload(path)
        if payload is None or "verdict" in payload:
            return
        payload["admission_certificate"] = {
            "facts_digest": admission.layout.facts_digest,
            "layout": admission.layout.to_dict(),
            "simulator_input": asdict(admission.simulator_input),
            "simulation": asdict(admission.simulation),
        }
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._record(lookup.key, lookup.result.program.digest, path, "certified")

    def _boundary(
        self,
        key: str,
        program: ShadowSpillProgram,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
    ) -> dict[str, object]:
        """The request a record answers, as the record states it."""

        return {
            "schema": _SCHEMA,
            "key_digest": key,
            "program_digest": program.digest,
            "initial_residency": [item.to_dict() for item in initial_residency],
            "final_residency": [item.to_dict() for item in final_residency],
            "simulation": {
                "devices": [asdict(item) for item in config.devices],
                "spill_capacity_bytes": config.spill_capacity_bytes,
            },
            "admission_digest": admission.digest if admission is not None else None,
            "search": algorithm.name,
            "search_options": search_options.to_dict(),
        }

    def _payload(self, path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"planned program {path} cannot be read") from exc
        if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
            raise ValueError(f"planned program {path} has an invalid schema")
        return value

    def _read(
        self,
        key: str,
        program: ShadowSpillProgram,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
    ) -> PlanLookup | _Verdict | None:
        """Read the record for `key` back, trusting what it states.

        The record's own digests and the request boundary it names are
        checked; its schedule is validated against the program; nothing is
        simulated. A record written before results were stored beside plans
        is a miss, and the fresh answer overwrites it.
        """

        path = self.path(key)
        value = self._payload(path)
        if value is None:
            return None
        expected = self._boundary(
            key,
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            algorithm,
            search_options,
        )
        normalized = json.loads(
            json.dumps(expected, sort_keys=True, separators=(",", ":"))
        )
        if value.get("key_digest") != key:
            raise ValueError(f"planned program {path} has the wrong identity")
        if value.get("program_digest") != program.digest:
            raise ValueError(f"planned program {path} has the wrong ShadowSpillProgram")
        for field, expected_value in normalized.items():
            if value.get(field) != expected_value:
                raise ValueError(f"planned program {path} has stale {field} evidence")
        verdict = value.get("verdict")
        if verdict is not None:
            if not isinstance(verdict, dict):
                raise ValueError(f"planned program {path} has an invalid verdict")
            self._record(key, program.digest, path, "read")
            return _Verdict(
                outcome=str(verdict.get("outcome")),
                error=str(verdict.get("error")),
                kind=None if verdict.get("kind") is None else str(verdict.get("kind")),
                message=str(verdict.get("message")),
            )
        if "simulation_result" not in value:
            return None
        schedule = MemorySchedule.from_dict(value.get("schedule"))
        raw_selections = value.get("selections")
        if not isinstance(raw_selections, list):
            raise ValueError(f"planned program {path} has invalid selections")
        selections = tuple(
            TaskAlternativeChoice.from_value(item, f"cache.selections[{index}]")
            for index, item in enumerate(raw_selections)
        )
        schedule.validate(program, selections)
        simulation = _simulation_result_from_value(
            value.get("simulation_result"), f"{path}.simulation_result"
        )
        diagnostics = _diagnostics_from_value(value.get("diagnostics"), path)
        if diagnostics.selected_makespan_ns != simulation.makespan_ns:
            raise ValueError(
                f"planned program {path} has inconsistent simulator evidence"
            )
        result = ProgramPlanResult(
            program=program,
            search_options=search_options,
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
        certificate = _certificate_from_value(value.get("admission_certificate"), path)
        self._record(key, program.digest, path, "read")
        return PlanLookup(result, True, key, certificate)

    def _write(
        self,
        key: str,
        result: ProgramPlanResult,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
        incumbent: ProgramPlanResult | None = None,
        improve: bool = False,
    ) -> None:
        if not self.policy.write_enabled:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            **self._boundary(
                key,
                result.program,
                result.initial_residency,
                result.final_residency,
                result.simulation_config,
                admission,
                algorithm,
                search_options,
            ),
            **_incumbent_field(incumbent),
            "schedule": result.schedule.to_dict(),
            "selections": [item.to_dict() for item in result.selections],
            "simulation_result": asdict(result.simulation),
            "diagnostics": result.diagnostics.to_dict(),
            "resident_slice": result.resident_slice.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        existing = None if self.policy.overwrite or improve else self._payload(path)
        # A verdict, or a record from before results were stored beside plans,
        # is superseded by an answer; anything else must be the same answer.
        if (
            existing is not None
            and "verdict" not in existing
            and "simulation_result" in existing
        ):
            # Provenance is not the answer: the same plan found with or
            # without a plan in hand is the same plan.
            if _answer(existing) != _answer(payload):
                raise ValueError(
                    "a fresh search differs from the stored planned program; "
                    "use a 'refresh' store mode or a new export_bypass_key: "
                    f"{path}"
                )
            self._record(key, result.program.digest, path, "matched")
            return
        atomic_text(path, encoded)
        self._record(
            key, result.program.digest, path, "improved" if improve else "write"
        )

    def _write_verdict(
        self,
        key: str,
        program: ShadowSpillProgram,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
        error: BaseException,
    ) -> None:
        if not self.policy.write_enabled:
            return
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            **self._boundary(
                key,
                program,
                initial_residency,
                final_residency,
                config,
                admission,
                algorithm,
                search_options,
            ),
            "verdict": {
                "outcome": (
                    "exhausted"
                    if isinstance(error, PlanSearchExhaustedError)
                    else "infeasible"
                ),
                "error": type(error).__name__,
                "kind": getattr(error, "kind", None),
                "message": str(error),
            },
        }
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._record(key, program.digest, path, "verdict")

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
            category="search",
            kind="selection",
            digest=key,
            path=path,
            access=access,
            schema=_SCHEMA,
            dependencies=(program_digest,),
        )


def _key(
    program: ShadowSpillProgram,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
    placement: AdmissionFacts | None,
    search_options: SearchOptions,
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
        "admission_digest": admission.digest if admission is not None else None,
        # Part of the identity: the search measures layouts against this
        # topology, so the same program under a different pool is a
        # different question and must not read a cached answer.
        "placement_digest": placement.digest if placement is not None else None,
        # Which search, and what it was told. Both are part of the
        # question: a plan one search chose is not the answer another would
        # give, and a plan searched over one candidate space is not the
        # answer for a different one. The plan to beat is not part of it:
        # see PlanStore.resolve. The search's name is inside
        # `search_options`, so it is not repeated here.
        "search_options": search_options.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _answer(payload: dict[str, object]) -> object:
    """The part of a record that is the answer: not its provenance, and not
    the certificate a later step wrote beside it."""

    return _without_provenance(
        {
            name: value
            for name, value in payload.items()
            if name != "admission_certificate"
        }
    )


def _certificate_from_value(value: object, path: Path) -> FixedLayoutAdmission | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"planned program {path} has an invalid certificate")
    where = f"{path}.admission_certificate"
    return FixedLayoutAdmission(
        layout=_fixed_layout_from_value(value.get("layout"), f"{where}.layout"),
        simulator_input=_simulation_admission_from_value(
            value.get("simulator_input"), f"{where}.simulator_input"
        ),
        simulation=_simulation_result_from_value(
            value.get("simulation"), f"{where}.simulation"
        ),
    )


def _verdict_error(verdict: _Verdict) -> Exception:
    """The refusal a stored verdict stands for, raised as the search raised it."""

    if verdict.outcome == "exhausted":
        return PlanSearchExhaustedError(verdict.message)
    return PlanInfeasibleError(
        verdict.message,
        kind=verdict.kind if verdict.kind is not None else verdict.error,
    )


def _claims_to_beat(
    incumbent: ProgramPlanResult | None, stored: ProgramPlanResult
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


def _incumbent_field(incumbent: ProgramPlanResult | None) -> dict[str, object]:
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


def _diagnostics_from_value(value: object, path: Path) -> PlanningDiagnostics:
    try:
        return PlanningDiagnostics.from_value(value, "cache.diagnostics")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"planned program {path} has invalid diagnostics") from exc


__all__ = ["PlanLookup", "PlanStore"]


def open_plan_store(artifact_store: ArtifactStore) -> PlanStore:
    """Open the plan store one artifact-store policy implies."""

    return PlanStore(
        artifact_store.plan_selections,
        policy=artifact_store.plan_policy,
        artifact_recorder=artifact_store.record,
    )


def resolve_plan(
    artifact_store: ArtifactStore,
    plans: PlanStore,
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    search_options: SearchOptions | None = None,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
    incumbent: ProgramPlanResult | None = None,
) -> PlanLookup:
    """Resolve one plan, planning only when the store does not have it.

    The request and its program are archived first, so a plan on disk can
    always be traced back to what was asked for, the plan to beat included.
    """

    artifact_store.archive_program(program)
    chosen = search_options if search_options is not None else SearchOptions()
    algorithm = chosen.resolved_algorithm
    artifact_store.archive_plan_request(
        {
            "schema": artifact_schema("plan_request"),
            "program_digest": program.digest,
            "initial_residency": [item.to_dict() for item in initial_residency],
            "final_residency": [item.to_dict() for item in final_residency],
            "simulation": {
                "devices": [asdict(item) for item in config.devices],
                "spill_capacity_bytes": config.spill_capacity_bytes,
            },
            # Both records in full, derived from the option types
            # themselves: an option added later is archived without a
            # second edit, and a request never records less than the key
            # it produced.
            "search": algorithm.name,
            "search_options": chosen.to_dict(),
            "admission": None if admission is None else admission.to_dict(),
            **_incumbent_field(incumbent),
        }
    )
    return plans.resolve(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        search_options=chosen,
        admission=admission,
        placement=placement,
        progress=progress,
        incumbent=incumbent,
    )
