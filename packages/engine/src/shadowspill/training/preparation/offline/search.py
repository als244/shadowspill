"""Search exported measurements without requiring an executable proposal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiles import Profiles
from shadowspill.communication.info import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.memory_search import SearchOptions
from shadowspill.planning.plan import plan_program
from shadowspill.profiling.apply import apply_profiles
from shadowspill.serialization.sharing import writing_metadata
from shadowspill.training.preparation.artifacts.cost_evidence import (
    communication_edits,
    profile_edits,
)
from shadowspill.training.preparation.capture.templates import instantiate_program
from shadowspill.training.preparation.identity import without_initialization
from shadowspill.training.preparation.offline.plans import LocalPlans
from shadowspill.training.preparation.planning.reservations import (
    spill_reservations,
)
from shadowspill.training.preparation.planning.sharing import (
    plan_sources,
    representative_sources,
    representative_use,
)
from shadowspill_client.protocol.training import ResolvedTraining

if TYPE_CHECKING:
    from shadowspill.training.preparation.offline.inputs import PlanningInputs


@dataclass(frozen=True)
class LocalSearch:
    config: ResolvedTraining
    profiles: Mapping[str, Profiles]
    plans: Mapping[str, Plan]
    candidates: tuple[str, ...]
    rejected: Mapping[str, str]

    def proposal(self) -> LocalPlans:
        # Equivalent workers share a complete Plan, including its implementation.
        # Keep that identity through the numeric proposal instead of expanding
        # the same schedule once per worker before saving it.
        with writing_metadata():
            return LocalPlans(
                self.config,
                self.profiles,
                {key: value.to_dict() for key, value in self.plans.items()},
                self.candidates,
                self.rejected,
            )


def search(
    inputs: PlanningInputs,
    *,
    config: ResolvedTraining,
    profiles: Mapping[str, Profiles] | None = None,
    communication: Mapping[str, CommunicationInfo] | None = None,
    candidates: Sequence[str] | None = None,
) -> LocalSearch:
    if config.memory.device_gib == 0 or config.memory.spill_gib == 0:
        raise ValueError(
            "offline planning needs budgets in GiB: zero is automatic and only "
            "the service sizes it (the preparation's budgets are in "
            "inputs.config.memory)"
        )
    normalized = replace(
        config,
        memory=inputs.config.memory,
        execution=inputs.config.execution,
        origins=inputs.config.origins,
        preparation=replace(
            config.preparation,
            reuse=inputs.config.preparation.reuse,
            origins=inputs.config.preparation.origins,
            planning=replace(
                config.preparation.planning,
                search=inputs.config.preparation.planning.search,
                workers=inputs.config.preparation.planning.workers,
            ),
        ),
    )
    if without_initialization(normalized) != without_initialization(inputs.config):
        raise ValueError("local planning settings require different exported cases")
    supplied = {} if profiles is None else profiles
    comm = {} if communication is None else communication
    if (
        not supplied.keys() <= inputs.sources.keys()
        or not comm.keys() <= inputs.sources.keys()
    ):
        raise ValueError("edited costs refer to an unexported capture")
    costs = {
        key: supplied.get(key, source.profiles)
        for key, source in inputs.sources.items()
    }
    communication = {
        key: comm.get(key, value) for key, value in inputs.communication.items()
    }
    for key, source in inputs.sources.items():
        profile_edits(source.profiles, costs[key])
        communication_edits(inputs.communication[key], communication[key])
    sources = {
        key: apply_profiles(source.program, costs[key])
        for key, source in inputs.sources.items()
    }
    selected = tuple(inputs.candidates if candidates is None else candidates)
    if not set(selected) <= inputs.candidates.keys() or len(set(selected)) != len(
        selected
    ):
        raise ValueError("local planning requires distinct exported candidates")
    required = {key for candidate in selected for key in inputs.candidates[candidate]}
    selected_templates = {inputs.requests[key].use.template for key in required}
    representatives = representative_sources(
        {key: sources[key] for key in selected_templates}, communication=communication
    )
    settings = config.preparation.planning
    assert settings.search is not None and settings.workers is not None
    search = replace(
        SearchOptions.from_dict(settings.search, "search"), workers=settings.workers
    )
    plans: dict[str, Plan] = {}
    rejected = {}
    cached: dict[tuple[str, int], Plan | str] = {}
    # An export without candidate descriptions names no operation and batch to
    # reserve for; the joint check after planning still decides.
    reservations = (
        spill_reservations(
            {key: inputs.requests[key] for key in required},
            {candidate: inputs.candidates[candidate] for candidate in selected},
            inputs.descriptions,
            {
                key: instantiate_program(
                    inputs.requests[key].use,
                    inputs.sources[inputs.requests[key].use.template].program,
                )
                for key in required
            },
        )
        if all(candidate in inputs.descriptions for candidate in selected)
        else dict.fromkeys(required, 0)
    )
    budgets = MemoryBudgets(
        int(config.memory.device_gib * (1 << 30)),
        int(config.memory.spill_gib * (1 << 30)),
    )
    for key in sorted(required):
        request = replace(
            inputs.requests[key],
            use=representative_use(inputs.requests[key].use, representatives),
        )
        reserved = reservations[key]
        if reserved > budgets.spill_bytes:
            rejected[key] = (
                f"the run's other operations keep {reserved} bytes of persistent "
                f"spill; the budget is {budgets.spill_bytes}"
            )
            continue
        if (request.id, reserved) in cached:
            found = cached[request.id, reserved]
            if isinstance(found, str):
                rejected[key] = found
            else:
                plans[key] = found
            continue
        template = request.use.template
        program = instantiate_program(request.use, inputs.sources[template].program)
        try:
            plan = plan_program(
                apply_profiles(program, costs[template]),
                memory_budgets=budgets,
                communication_info=communication[template],
                search_options=search,
                ordering=request.ordering,
                reserved_spill_bytes=reserved,
            )
            if plan.simulation_only_reasons:
                raise ValueError("local Plan lacks physical memory admission")
            plans[key] = plan
            cached[request.id, reserved] = plan
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            rejected[key] = str(error)
            cached[request.id, reserved] = str(error)
    selected = tuple(
        key for key in selected if set(inputs.candidates[key]) <= plans.keys()
    )
    needed = {key for candidate in selected for key in inputs.candidates[candidate]}
    retained = {key: plans[key] for key in needed}
    templates = {inputs.requests[key].use.template for key in needed} | plan_sources(
        {0: retained}
    )
    return LocalSearch(
        config,
        {key: costs[key] for key in templates},
        retained,
        selected,
        rejected,
    )
