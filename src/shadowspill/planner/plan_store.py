"""Content-addressed persistence for the plans a search answers with."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
from .diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    GraphPairOutcome,
    PlanningDiagnostics,
    graph_pair_outcomes,
)
from .diagnostics.json import without_measurements
from .diagnostics.plan import PlanSummary, summarize_selected_plan
from .result import ProgramPlanResult, ResolutionPlan
from .search import (
    SearchAlgorithm,
    SearchOptions,
    answer_no_worse_than,
)
from .search.toolkit.resolution import recompute_share
from .serialization import (
    _boolean,
    _fixed_layout_from_value,
    _integer,
    _list,
    _resident_slice_from_value,
    _simulation_admission_from_value,
    _simulation_result_from_value,
)

_SCHEMA = artifact_schema("plan_selection")
_SUMMARY_SCHEMA = artifact_schema("plan_summary")
_RESOLUTION_SCHEMA = artifact_schema("plan_resolution")


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
class PlanSummaryLookup:
    """What a stored plan promises, read without the plan.

    Everything a caller comparing many plans asks of each one -- the makespan,
    the :class:`PlanSummary`, the outcome of every graph-pair selection the
    search evaluated, and whether the search answered with the plan it was
    handed -- from the small record the store keeps beside the plan.
    """

    #: The key the plan is filed under, which `plan_program` reads it by.
    key: str
    #: The makespan of the plan as its certificate re-simulated it.
    makespan_ns: int
    summary: PlanSummary
    graph_pair_outcomes: tuple[GraphPairOutcome, ...]
    #: True when the search answered with the plan it was handed to beat.
    answered_with_incumbent: bool


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

    def summary_path(self, key: str) -> Path:
        return digest_directory(self.root, key) / "summary.json"

    def resolution_path(self, key: str, label: str) -> Path:
        """Where a kept resolution's record sits, beside the answer's."""

        return (
            digest_directory(self.root, key) / "resolutions" / label / "selection.json"
        )

    def summary(
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        search_options: SearchOptions | None = None,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
    ) -> PlanSummaryLookup | None:
        """What the store knows about a question, without reading its plan.

        The same key `resolve` computes, answered from the summary beside the
        plan. A record that has a certified plan and no summary -- a store
        written before summaries were kept -- answers from the plan once and
        keeps the summary it built, when the mode allows writing, so a store
        learns on first use. A recorded refusal is raised as `resolve` raises
        it. `None` is a miss, or a plan nobody has certified yet: the caller
        plans, and `resolve` applies the store's mode to the miss.

        The plan to beat is not taken here. A plan in hand that claims to be
        faster than the summary says is a question only a search settles, and
        `resolve` is where it is asked.
        """

        if not self.policy.read_enabled:
            return None
        chosen = search_options if search_options is not None else SearchOptions()
        key = _key(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            placement,
            chosen,
        )
        path = self.summary_path(key)
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            value = None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"plan summary {path} cannot be read") from exc
        if value is not None:
            if not isinstance(value, dict) or value.get("schema") != _SUMMARY_SCHEMA:
                raise ValueError(f"plan summary {path} has an invalid schema")
            if value.get("key_digest") != key:
                raise ValueError(f"plan summary {path} has the wrong identity")
            if value.get("program_digest") != program.digest:
                raise ValueError(
                    f"plan summary {path} has the wrong ShadowSpillProgram"
                )
            self._record(key, program.digest, path, "read", summary=True)
            return _summary_from_value(value, key, path)
        stored = self._read(
            key,
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            chosen.resolved_algorithm,
            chosen,
        )
        if isinstance(stored, _Verdict):
            raise _verdict_error(stored)
        if stored is None or stored.certificate is None:
            return None
        return self._write_summary(
            key, certified_result(stored.result, stored.certificate)
        )

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
        keep_resolutions: bool = False,
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

        `keep_resolutions` asks the search for every resolved program's
        best plan and files each beside the answer (`resolution_path`) when
        the search runs; a hit reads those records too. Without it, a hit
        reads only the selected answer.
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
                keep_resolutions=keep_resolutions,
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
            self.policy.refuse_miss(
                "plan", key, request=_request_summary(program, config)
            )
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
                    # Passed only when asked, so a search that never heard
                    # of it is called as before.
                    **({"keep_resolutions": True} if keep_resolutions else {}),
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
                key,
                result,
                admission,
                algorithm,
                chosen,
                incumbent,
                improve=True,
                keep_resolutions=keep_resolutions,
            )
            return PlanLookup(result, False, key)
        self._write(
            key,
            result,
            admission,
            algorithm,
            chosen,
            incumbent,
            keep_resolutions=keep_resolutions,
        )
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
        # The summary is what the certified plan promises, so it is written
        # here, with the certificate, and by nothing that precedes one.
        payload["admission_certificate"] = {
            "facts_digest": admission.layout.facts_digest,
            "layout": admission.layout.to_dict(),
            "simulator_input": asdict(admission.simulator_input),
            "simulation": asdict(admission.simulation),
        }
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._record(lookup.key, lookup.result.program.digest, path, "certified")
        self._write_summary(lookup.key, certified_result(lookup.result, admission))

    def certify_resolution(
        self, lookup: PlanLookup, plan: ResolutionPlan, admission: FixedLayoutAdmission
    ) -> None:
        """Write a kept resolution's certificate beside its record, as
        `certify` does for the answer's. A resolution the store did not
        file -- a hit, or a mode that writes nothing -- is left alone."""

        if not self.policy.write_enabled or not lookup.key:
            return
        label = next(
            (
                name
                for name, item in resolution_labels(lookup.result).items()
                if item is plan
            ),
            None,
        )
        if label is None:
            return
        path = self.resolution_path(lookup.key, label)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            return
        payload["admission_certificate"] = {
            "facts_digest": admission.layout.facts_digest,
            "layout": admission.layout.to_dict(),
            "simulator_input": asdict(admission.simulator_input),
            "simulation": asdict(admission.simulation),
        }
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._record(
            lookup.key, lookup.result.program.digest, path, "certified", resolution=True
        )

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
            **_request(
                program,
                initial_residency,
                final_residency,
                config,
                admission,
                search_options,
            ),
            "key_digest": key,
            "search": algorithm.name,
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
        *,
        keep_resolutions: bool = False,
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
        retained = (
            self._read_resolutions(key, program, normalized) if keep_resolutions else ()
        )
        if retained is None:
            # The process may have stopped while writing alternative records.
            # Treat incomplete retention as a cache miss, never a partial answer.
            return None
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
            resolutions=retained,
        )
        certificate = _certificate_from_value(value.get("admission_certificate"), path)
        self._record(key, program.digest, path, "read")
        return PlanLookup(result, True, key, certificate)

    def _read_resolutions(
        self, key: str, program: ShadowSpillProgram, boundary: dict
    ) -> tuple[ResolutionPlan, ...] | None:
        """Restore retained alternatives for consumers that need complete plans."""
        expected = {**boundary, "schema": _RESOLUTION_SCHEMA}
        plans = {}
        directory = digest_directory(self.root, key) / "resolutions"
        try:
            labels = json.loads((directory / "manifest.json").read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"resolution manifest {directory} cannot be read") from exc
        if (
            not isinstance(labels, list)
            or any(
                not isinstance(label, str)
                or not label
                or label in {".", ".."}
                or Path(label).name != label
                for label in labels
            )
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"resolution manifest {directory} has invalid labels")
        for label in labels:
            path = self.resolution_path(key, label)
            try:
                value = json.loads(path.read_text())
            except FileNotFoundError:
                return None
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"resolution plan {path} cannot be read") from exc
            if not isinstance(value, dict):
                raise ValueError(f"resolution plan {path} has an invalid schema")
            for field, expected_value in expected.items():
                if value.get(field) != expected_value:
                    raise ValueError(
                        f"resolution plan {path} has stale {field} evidence"
                    )
            identity = value.get("resolution")
            if (
                not isinstance(identity, dict)
                or identity.get("label") != path.parent.name
                or not isinstance(identity.get("selection_id"), str)
                or not isinstance(identity.get("candidate_id"), str)
            ):
                raise ValueError(f"resolution plan {path} has invalid identity")
            selection_id = identity["selection_id"]
            if selection_id in plans:
                raise ValueError(f"resolution plan {path} repeats {selection_id}")
            selections = tuple(
                TaskAlternativeChoice.from_value(item, f"{path}.selections[{index}]")
                for index, item in enumerate(_list(value.get("selections"), str(path)))
            )
            schedule = MemorySchedule.from_dict(value.get("schedule"))
            schedule.validate(program, selections)
            plans[selection_id] = ResolutionPlan(
                selection_id,
                selections,
                identity["candidate_id"],
                schedule,
                _simulation_result_from_value(
                    value.get("simulation_result"), f"{path}.simulation_result"
                ),
                _resident_slice_from_value(
                    value.get("resident_slice"), f"{path}.resident_slice"
                ),
            )
            self._record(key, program.digest, path, "read", resolution=True)
        return tuple(plans[key] for key in sorted(plans))

    def _write(
        self,
        key: str,
        result: ProgramPlanResult,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
        incumbent: ProgramPlanResult | None = None,
        improve: bool = False,
        keep_resolutions: bool = False,
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
        else:
            # A summary describes the plan it was written beside; a new plan
            # has none until it is certified.
            self.summary_path(key).unlink(missing_ok=True)
            atomic_text(path, encoded)
            self._record(
                key, result.program.digest, path, "improved" if improve else "write"
            )
        if keep_resolutions:
            self._write_resolutions(key, result, admission, algorithm, search_options)

    def _write_resolutions(
        self,
        key: str,
        result: ProgramPlanResult,
        admission: AdmissionFacts | None,
        algorithm: SearchAlgorithm,
        search_options: SearchOptions,
    ) -> None:
        """Every resolved program's best plan beside the answer, one record
        each in the answer's own form less the search diagnostics, replacing
        whatever an earlier search of this key left there."""

        directory = digest_directory(self.root, key) / "resolutions"
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        boundary = self._boundary(
            key,
            result.program,
            result.initial_residency,
            result.final_residency,
            result.simulation_config,
            admission,
            algorithm,
            search_options,
        )
        for label, plan in resolution_labels(result).items():
            payload: dict[str, object] = {
                **boundary,
                "schema": _RESOLUTION_SCHEMA,
                "resolution": {
                    "label": label,
                    "selection_id": plan.selection_id,
                    "candidate_id": plan.candidate_id,
                    "recompute_share": str(
                        recompute_share(result.program, plan.selections)
                    ),
                    "selected": plan.selection_id
                    == result.diagnostics.selected_selection_id,
                },
                "schedule": plan.schedule.to_dict(),
                "selections": [item.to_dict() for item in plan.selections],
                "simulation_result": asdict(plan.simulation),
                "resident_slice": plan.resident_slice.to_dict(),
            }
            path = self.resolution_path(key, label)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(
                path, json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            self._record(key, result.program.digest, path, "write", resolution=True)
        # Published last: readers can distinguish complete retention from an
        # interrupted write without making partial artifacts look successful.
        atomic_text(
            directory / "manifest.json", json.dumps(sorted(resolution_labels(result)))
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
        self.summary_path(key).unlink(missing_ok=True)
        atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._record(key, program.digest, path, "verdict")

    def _write_summary(self, key: str, result: ProgramPlanResult) -> PlanSummaryLookup:
        """Write what a certified plan promises beside it, and return it.

        The record is read back through the same reader a later call uses,
        so what this call returns is exactly what the store will answer.
        """

        path = self.summary_path(key)
        payload = {
            "schema": _SUMMARY_SCHEMA,
            "key_digest": key,
            "program_digest": result.program.digest,
            "schedule_digest": result.schedule.digest,
            "makespan_ns": result.simulation.makespan_ns,
            "answered_with_incumbent": (
                result.diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID
            ),
            "summary": summarize_selected_plan(result).as_dict(),
            "graph_pair_outcomes": [
                item.as_dict() for item in graph_pair_outcomes(result)
            ],
        }
        if self.policy.write_enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(
                path, json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            self._record(key, result.program.digest, path, "write", summary=True)
        return _summary_from_value(payload, key, path)

    def _record(
        self,
        key: str,
        program_digest: str,
        path: Path,
        access: str,
        *,
        summary: bool = False,
        resolution: bool = False,
    ) -> None:
        if self.artifact_recorder is None:
            return
        kind, schema = "selection", _SCHEMA
        if summary:
            kind, schema = "summary", _SUMMARY_SCHEMA
        elif resolution:
            kind, schema = "resolution", _RESOLUTION_SCHEMA
        self.artifact_recorder(
            category="search",
            kind=kind,
            digest=key,
            path=path,
            access=access,
            schema=schema,
            dependencies=(program_digest,),
        )


def resolution_labels(result: ProgramPlanResult) -> dict[str, ResolutionPlan]:
    """Each kept resolution under the name its record is filed by: the share
    of the flexible groups it recomputes, as ``recompute_<share>``, with a
    digest of its selection appended when two resolutions share one."""

    labels: dict[str, ResolutionPlan] = {}
    for plan in result.resolutions:
        share = recompute_share(result.program, plan.selections)
        label = f"recompute_{share.numerator}" + (
            f"of{share.denominator}" if share.denominator != 1 else ""
        )
        if label in labels:
            label += "_" + hashlib.sha256(plan.selection_id.encode()).hexdigest()[:8]
        labels[label] = plan
    return labels


def certified_result(
    result: ProgramPlanResult, certificate: FixedLayoutAdmission
) -> ProgramPlanResult:
    """The plan as its certificate re-simulated it: what a caller is handed.

    The search's own simulation is logical. The certificate ran the same
    schedule over the fixed layout, so the two agree on everything the
    schedule decides, and the certified one is what the answer reports.
    """

    return replace(
        result,
        simulation=certificate.simulation,
        diagnostics=result.diagnostics.replace_selected_makespan(
            certificate.simulation.makespan_ns
        ),
    )


def _summary_from_value(
    value: dict[str, object], key: str, path: Path
) -> PlanSummaryLookup:
    where = f"plan summary {path}"
    outcomes = _list(value.get("graph_pair_outcomes"), f"{where}.graph_pair_outcomes")
    return PlanSummaryLookup(
        key=key,
        makespan_ns=_integer(value.get("makespan_ns"), f"{where}.makespan_ns"),
        summary=PlanSummary.from_dict(value.get("summary"), f"{where}.summary"),
        graph_pair_outcomes=tuple(
            GraphPairOutcome.from_dict(item, f"{where}.graph_pair_outcomes[{index}]")
            for index, item in enumerate(outcomes)
        ),
        answered_with_incumbent=_boolean(
            value.get("answered_with_incumbent"), f"{where}.answered_with_incumbent"
        ),
    )


def _request_summary(program: ShadowSpillProgram, config: SimulationConfig) -> str:
    """The inputs of a plan key a reader compares first, on one line."""

    device = config.devices[0]
    return (
        f"program {program.digest[:12]}, capacity {device.capacity_bytes} B,"
        f" fetch {device.fetch_solo_bandwidth_bytes_per_second}/"
        f"{device.fetch_concurrent_bandwidth_bytes_per_second} B/s solo/concurrent"
        f" latency {device.fetch_latency_ns} ns (metadata only),"
        f" evict {device.evict_solo_bandwidth_bytes_per_second}/"
        f"{device.evict_concurrent_bandwidth_bytes_per_second} B/s solo/concurrent"
        f" latency {device.evict_latency_ns} ns (metadata only),"
        f" spill {config.spill_capacity_bytes} B"
    )


def _request(
    program: ShadowSpillProgram,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
    search_options: SearchOptions,
) -> dict[str, object]:
    """What a question to the store is made of, as every record states it.

    Which search, and what it was told, are part of it: a plan one search
    chose is not the answer another would give, and a plan searched over one
    candidate space is not the answer for a different one. The plan to beat is
    not part of it: see `PlanStore.resolve`. The search's name is inside
    `search_options`, so a record repeats it only where a reader wants it.
    """

    return {
        "schema": _SCHEMA,
        "program_digest": program.digest,
        "initial_residency": [item.to_dict() for item in initial_residency],
        "final_residency": [item.to_dict() for item in final_residency],
        "simulation": {
            "devices": [asdict(device) for device in config.devices],
            "spill_capacity_bytes": config.spill_capacity_bytes,
        },
        "admission_digest": admission.digest if admission is not None else None,
        "search_options": search_options.to_dict(),
    }


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
        **_request(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            search_options,
        ),
        # Part of the identity: the search measures layouts against this
        # topology, so the same program under a different pool is a
        # different question and must not read a cached answer.
        "placement_digest": placement.digest if placement is not None else None,
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


__all__ = [
    "PlanLookup",
    "PlanStore",
    "PlanSummaryLookup",
    "certified_result",
    "resolution_labels",
]


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
    keep_resolutions: bool = False,
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
        keep_resolutions=keep_resolutions,
    )
