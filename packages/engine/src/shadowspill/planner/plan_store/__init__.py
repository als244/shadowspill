"""Optional content-addressed planning cache and search orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.ir import ResidencySpec, ShadowSpillProgram
from shadowspill.simulator import SimulationConfig, SimulationInfeasibleError
from shadowspill.store import (
    CONTRIBUTE,
    ArtifactRecorder,
    StorePolicy,
    digest_directory,
)

from ..admission import AdmissionFacts
from ..admission.layout.model import FixedLayoutAdmission
from ..result import ProgramPlanResult, ResolutionPlan, certified_result
from ..search import SearchOptions, answer_no_worse_than
from ..serialization.certificate_file import CERTIFICATE_FILE, write_certificate
from .reader import read_selection
from .records import (
    RESOLUTION_SCHEMA,
    SCHEMA,
    SUMMARY_SCHEMA,
    PlanLookup,
    PlanSummaryLookup,
    Verdict,
)
from .request import PlanRequest
from .resolutions import resolution_labels
from .summary import read_summary, write_summary
from .writer import write_selection, write_verdict


class PlanStore:
    """Cache selected plans by program, residency, machine and search options.

    Worker count and incumbent provenance do not affect identity. A better
    incumbent triggers a fresh search; cached answers may never conceal it.
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

    def certificate_path(self, key: str) -> Path:
        return digest_directory(self.root, key) / CERTIFICATE_FILE

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
        """Read certified summary evidence without loading the selected schedule."""
        return read_summary(
            self,
            PlanRequest(
                program,
                initial_residency,
                final_residency,
                config,
                admission,
                placement,
                search_options if search_options is not None else SearchOptions(),
            ),
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
        """Read a compatible cached answer or search and save a better result."""
        chosen = search_options if search_options is not None else SearchOptions()
        algorithm = chosen.resolved_algorithm
        request = PlanRequest(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            placement,
            chosen,
        )
        stored = (
            read_selection(self, request, keep_resolutions=keep_resolutions)
            if self.policy.read_enabled
            else None
        )
        if isinstance(stored, Verdict):
            # A verdict answers the question it was recorded for. With a plan
            # in hand the search may answer with that plan instead, so the
            # question is asked again.
            if incumbent is None:
                raise stored.exception()
            stored = None
        if stored is not None and (
            incumbent is None
            or incumbent.simulation.makespan_ns >= stored.result.simulation.makespan_ns
        ):
            return stored
        if stored is None:
            self.policy.refuse_miss("plan", request.key, request=request.describe())
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
                    **({"keep_resolutions": True} if keep_resolutions else {}),
                ),
                incumbent=incumbent,
                config=config,
                placement=placement,
            )
        except (
            PlanInfeasibleError,
            SimulationInfeasibleError,
            PlanSearchExhaustedError,
        ) as error:
            if incumbent is None:
                write_verdict(self, request, error)
            raise
        if (
            stored is not None
            and result.simulation.makespan_ns >= stored.result.simulation.makespan_ns
        ):
            return stored
        write_selection(
            self,
            request,
            result,
            incumbent,
            improve=stored is not None,
            keep_resolutions=keep_resolutions,
        )
        return PlanLookup(result, False, request.key)

    def certify(self, lookup: PlanLookup, admission: FixedLayoutAdmission) -> None:
        """Write a plan's fixed-layout certificate beside it.

        The certificate is a function of the plan and the facts it was
        certified against, which its layout names by digest, so a later read
        serves it without placing or simulating again.
        """

        if not self.policy.write_enabled or not lookup.key:
            return
        path = self.path(lookup.key)
        if not path.exists():
            return
        # Its own file: the plan record is not read or written again.
        write_certificate(self.certificate_path(lookup.key), admission)
        self._record(lookup.key, lookup.result.program.digest, path, "certified")
        # The summary is what the certified plan promises, so it is written
        # here, with the certificate, and by nothing that precedes one.
        write_summary(self, lookup.key, certified_result(lookup.result, admission))

    def certify_resolution(
        self, lookup: PlanLookup, plan: ResolutionPlan, admission: FixedLayoutAdmission
    ) -> None:
        """Certify a retained alternative only when its record exists."""

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
        if not path.exists():
            return
        write_certificate(path.parent / CERTIFICATE_FILE, admission)
        self._record(
            lookup.key, lookup.result.program.digest, path, "certified", resolution=True
        )

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
        kind, schema = "selection", SCHEMA
        if summary:
            kind, schema = "summary", SUMMARY_SCHEMA
        elif resolution:
            kind, schema = "resolution", RESOLUTION_SCHEMA
        self.artifact_recorder(
            category="search",
            kind=kind,
            digest=key,
            path=path,
            access=access,
            schema=schema,
            dependencies=(program_digest,),
        )


__all__ = ["PlanStore"]
