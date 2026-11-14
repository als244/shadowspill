"""Connect the optional artifact store to planning requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from shadowspill.ir import (
    ResidencySpec,
    ShadowSpillProgram,
)
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationConfig
from shadowspill.store import ArtifactStore

from ..admission import AdmissionFacts
from ..result import ProgramPlanResult
from ..search import SearchOptions
from . import PlanStore
from .records import PlanLookup
from .request import incumbent_field


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
            **incumbent_field(incumbent),
        }
    )
    lookup = plans.resolve(
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
    if chosen.trace_directory is not None:
        algorithm.record_trace(
            lookup.result,
            options=chosen,
            admission=admission,
            placement=placement,
            progress=progress,
        )
    return lookup
