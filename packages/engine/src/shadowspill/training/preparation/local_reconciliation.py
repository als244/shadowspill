"""Admit submitted schedules using retained measurements and existing rechecks."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.profiling.apply import apply_profiles

from .bundle import TrainingBundle
from .cost_evidence import communication_edits, profile_edits
from .family_checks import _collectives, _orderings
from .family_identity import candidate_identity, plan_identity, validate_profile
from .family_records import PlannedCandidate, WorkerPlans
from .family_selection import select_family
from .local_plans import LocalPlans
from .planning_requests import PlanningRequest, planning_requests
from .shared_planning import plan_sources
from .templates import TemplateInventory


def reconcile_local_plans(
    submitted: LocalPlans,
    root: Path,
    inventory: TemplateInventory,
) -> tuple[TrainingBundle, Mapping[str, Any]]:
    """The caller authenticates/stages a completed preparation before this call.

    No client executable bytes are loaded. Original captures supply all executable
    payloads. The returned bundle still requires service admission, including CPU
    re-derivation of every distinct proposed Program and schedule, before use.
    This never invokes build/profile, per-Plan search, or a workload factory.
    """
    if submitted.config != inventory.layouts.setup.config:
        raise ValueError("local Plans differ from the resolved configuration")
    exported = planning_requests(inventory)
    candidates = exported["candidates"]
    if not set(submitted.candidates) <= candidates.keys():
        raise ValueError("local Plans contain unprepared candidates")
    needed = {
        key for candidate in submitted.candidates for key in candidates[candidate]
    }
    if needed != submitted.plans.keys():
        raise ValueError("local Plans must contain exactly their selected dependencies")
    requests = {
        key: PlanningRequest.from_dict(exported["requests"][key]) for key in needed
    }
    templates = {request.use.template for request in requests.values()}
    if not templates <= submitted.profiles.keys() <= inventory.templates.keys():
        raise ValueError("local Plans must contain their exported source profiles")
    sources, communication, edits = {}, {}, {}
    payloads: dict[str, bytes] = {}
    for key in submitted.profiles:
        original = ProfiledProgram.load(root / "calibrated" / f"{key}.ssp")
        validate_profile(inventory, key, original)
        edits[key] = {
            "tasks": profile_edits(original.profiles, submitted.profiles[key]),
            "communication": {},
        }
        sources[key] = apply_profiles(original.program, submitted.profiles[key])
        communication[key] = CommunicationInfo.load(
            root / "communication" / f"{key}.ssp"
        )
        payloads.update(original._archive_payloads())
    plans: dict[int, dict[str, Plan]] = {}
    decoded: dict[int, Plan] = {}
    seen_communication: dict[str, CommunicationInfo] = {}
    for name, request in requests.items():
        key = request.use.template
        value = submitted.plans[name]
        if id(value) not in decoded:
            decoded[id(value)] = Plan.from_dict(plain(value), payloads=payloads)
        plan = decoded[id(value)]
        if plan.pool_memory is not None:
            raise ValueError("local Plans require the exported pool memory policy")
        if (
            key in seen_communication
            and seen_communication[key] != plan.communication_info
        ):
            raise ValueError("local Plans disagree on edited communication costs")
        seen_communication[key] = plan.communication_info
        edits[key]["communication"] = communication_edits(
            communication[key], plan.communication_info
        )
        plans.setdefault(request.rank, {})[name] = plan
    if submitted.profiles.keys() != templates | plan_sources(plans):
        raise ValueError("local Plans must contain exactly their source profiles")
    rows = []
    for family in inventory.families.values():
        for candidate in family:
            for ordering in _orderings(candidate):
                key = candidate_identity(candidate, ordering)
                if key not in submitted.candidates:
                    continue
                workers = {
                    rank: WorkerPlans(
                        plan_identity(worker.main, ordering),
                        None
                        if worker.optimizer is None
                        else plan_identity(worker.optimizer, None),
                    )
                    for rank, worker in candidate.workers.items()
                }
                _collectives(workers, plans)
                rows.append(
                    PlannedCandidate(
                        key,
                        candidate,
                        ordering,
                        workers,
                        max(
                            plans[rank][worker.main].timeline.makespan_ns / 1e9
                            for rank, worker in workers.items()
                        ),
                    )
                )
    prepared = select_family(inventory, rows, plans, (), max_family_attempts=4096)
    # The bundle contains only selected captures; the measured export continues
    # to retain every original candidate and its original numeric evidence.
    needed_sources = {
        use.template
        for family in prepared.families.values()
        for row in family
        for worker in row.source.workers.values()
        for use in (worker.main, worker.optimizer)
        if use is not None
    } | plan_sources(prepared.plans)
    from .bundle_manifest import manifest

    bundle = TrainingBundle(
        submitted.config,
        {key: sources[key] for key in needed_sources},
        prepared.plans,
        manifest(prepared),
    )
    return bundle, {key: value for key, value in edits.items() if any(value.values())}
