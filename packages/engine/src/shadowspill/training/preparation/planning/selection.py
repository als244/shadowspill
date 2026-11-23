"""Retain each requested capability and jointly admitted packing alternatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from itertools import product
from types import MappingProxyType
from typing import Literal, cast

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.shared import SharedMemory, shared_memory
from shadowspill.planning.spill import spill_configuration
from shadowspill.training.preparation.capture.templates import TemplateInventory
from shadowspill.training.preparation.planning.identity import execution_names
from shadowspill.training.preparation.planning.records import (
    PlannedCandidate,
    PlanRejection,
    PreparedPlans,
)
from shadowspill_client.inputs.coverage import CoverageCertificate, certify_coverage
from shadowspill_client.inputs.coverage_bounds import CoverageError
from shadowspill_client.protocol.operations import OperationKey


def certificate(
    inventory: TemplateInventory, candidate: PlannedCandidate
) -> CoverageCertificate:
    setup = inventory.layouts.setup
    placement = setup.context.parallelism.inputs
    assert placement is not None
    return certify_coverage(
        setup.context.schema,
        setup.config.batches[candidate.batch],
        placement,
        [candidate.source.layout.candidate.schedule],
        grouping=cast(
            Literal["independent", "grouped"], setup.model.component.batching
        ),
    )


class _HostDemand:
    """The host memory a Plan set decides, as its workers reserve it.

    Each rank reserves its spill pool, sized from its joint spill, and the
    largest trace budget among its Plans; a Plan's trace budget is computed
    once.
    """

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self._traces: dict[int, int] = {}

    def _trace(self, plan: Plan) -> int:
        from shadowspill.training.worker.trace_files import plan_trace_budget

        key = id(plan)
        if key not in self._traces:
            self._traces[key] = plan_trace_budget(plan).host_bytes
        return self._traces[key]

    def require(
        self,
        plans: Mapping[int, Mapping[str, Plan]],
        memory: Mapping[int, SharedMemory],
    ) -> None:
        if self.limit is None:
            return
        pools = traces = 0
        for rank, members in plans.items():
            first = next(iter(members.values()))
            pools += spill_configuration(
                memory[rank].spill_bytes, first.memory_budgets.spill_bytes, None
            ).capacity
            traces += max(
                self._trace(path)
                for plan in members.values()
                for path in (plan, plan._accumulation_plan)
                if path is not None
            )
        if pools + traces > self.limit:
            gib = 1 << 30
            raise PlanInfeasibleError(
                f"the Plans need {(pools + traces) / gib:.2f} GiB of host memory "
                f"across {len(plans)} workers ({pools / gib:.2f} GiB of spill "
                f"pools, {traces / gib:.2f} GiB of trace budgets); the node "
                f"leaves {self.limit / gib:.2f} GiB for them",
                kind="prepared_host_memory",
                required_bytes=pools + traces,
                capacity_bytes=self.limit,
            )


def require_host_memory(
    plans: Mapping[int, Mapping[str, Plan]],
    memory: Mapping[int, SharedMemory],
    limit: int | None,
) -> None:
    """Refuse a Plan set whose spill pools and trace budgets exceed ``limit``."""
    _HostDemand(limit).require(plans, memory)


def joint_memory(
    candidates: Sequence[PlannedCandidate], plans: Mapping[int, Mapping[str, Plan]]
) -> tuple[dict[int, Mapping[str, Plan]], dict[int, SharedMemory]]:
    selected: dict[int, dict[str, Plan]] = {}
    for candidate in candidates:
        for rank, worker in candidate.workers.items():
            for name in worker.members():
                selected.setdefault(rank, {})[name] = plans[rank][name]
    memory = {}
    shared: dict[tuple[tuple[str, int], ...], SharedMemory] = {}
    frozen: dict[int, Mapping[str, Plan]] = {}
    for rank, members in sorted(selected.items()):
        # Deterministic primary/alias ownership, independent of search completion.
        ordered = dict(sorted(members.items()))
        first = next(iter(ordered))
        identity = tuple((name, id(plan)) for name, plan in ordered.items())
        if identity not in shared:
            shared[identity] = shared_memory(
                ordered[first],
                {name: plan for name, plan in ordered.items() if name != first},
            )
        memory[rank] = shared[identity]
        frozen[rank] = MappingProxyType(ordered)
    return frozen, memory


def rejection(
    candidate: PlannedCandidate, error: Exception, *, stage: str
) -> PlanRejection:
    source = candidate.source.layout.candidate
    return PlanRejection(
        candidate.id,
        candidate.operation,
        candidate.batch,
        source.case,
        source.count,
        None if candidate.ordering is None else candidate.ordering.label,
        stage,
        str(error),
        isinstance(error, PlanSearchExhaustedError),
    )


def _execution_family(
    candidates: Sequence[PlannedCandidate], plans: Mapping[int, Mapping[str, Plan]]
) -> tuple[list[PlannedCandidate], dict[int, dict[str, Plan]]]:
    """Assign shared invocation names before choosing primary/alias ownership."""
    renamed = []
    named: dict[int, dict[str, Plan]] = {}
    for row in candidates:
        names = execution_names(row.source, row.ordering)
        for rank, worker in row.workers.items():
            target = named.setdefault(rank, {})
            for previous, name in zip(worker.members(), names.members(), strict=True):
                plan = plans[rank][previous]
                if name in target and target[name] is not plan:
                    raise ValueError("one prepared Plan use has different schedules")
                target[name] = plan
        renamed.append(
            replace(row, workers=MappingProxyType(dict.fromkeys(row.workers, names)))
        )
    return renamed, named


def select_family(
    inventory: TemplateInventory,
    candidates: Sequence[PlannedCandidate],
    plans: Mapping[int, Mapping[str, Plan]],
    rejected: Sequence[PlanRejection],
    *,
    max_family_attempts: int,
    host_limit_bytes: int | None = None,
) -> PreparedPlans:
    """No partial result: every operation/explicit batch retains a candidate.

    Search complete capability combinations before adding alternative layouts.
    A bounded stop remains search exhaustion, never a proof of infeasibility.
    Unexpected shared-state/schema failures propagate rather than pruning work.
    """
    candidates, plans = _execution_family(candidates, plans)
    host = _HostDemand(host_limit_bytes)
    setup = inventory.layouts.setup
    ordered = sorted(candidates, key=lambda c: (c.duration_seconds, c.id))
    requirements = [
        (op, batch)
        for op in inventory.families
        for batch in range(len(setup.config.batches))
    ]
    witnesses: dict[tuple[OperationKey, int], list[PlannedCandidate]] = {
        key: [] for key in requirements
    }
    proofs = {}
    for row in ordered:
        with suppress(CoverageError):
            proofs[row.id] = certificate(inventory, row)
        witnesses[row.operation, row.batch].append(row)
    missing = [key for key, values in witnesses.items() if not values]
    if missing:
        messages = [
            f"{op.operation}/{op.outputs}/{op.mode}, batch {batch}"
            for op, batch in missing
        ]
        unknown = any(
            r.search_exhausted and (r.operation, r.batch) in missing for r in rejected
        )
        message = "no admitted candidate for " + "; ".join(messages)
        if unknown:
            raise PlanSearchExhaustedError(message, diagnostics=tuple(rejected))
        raise PlanInfeasibleError(
            message, kind="prepared_coverage", diagnostics=tuple(rejected)
        )
    attempts = 0
    last_error = None
    selected: list[PlannedCandidate] | None = None
    selected_plans: dict[int, Mapping[str, Plan]] = {}
    memory: dict[int, SharedMemory] = {}
    for combination in product(*(witnesses[key] for key in requirements)):
        if attempts == max_family_attempts:
            raise PlanSearchExhaustedError(
                f"joint coverage search reached {max_family_attempts} "
                "complete families",
                diagnostics=tuple(rejected)
                + (() if last_error is None else (last_error,)),
            )
        attempts += 1
        try:
            selected_plans, memory = joint_memory(combination, plans)
            host.require(selected_plans, memory)
        except PlanInfeasibleError as error:
            last_error = error
            continue
        selected = list(combination)
        break
    if selected is None:
        assert last_error is not None
        if any(row.search_exhausted for row in rejected):
            raise PlanSearchExhaustedError(
                "no complete coverage family admitted; some searches exhausted",
                diagnostics=(*rejected, last_error),
            )
        raise PlanInfeasibleError(
            "no complete coverage witness family fits joint runtime admission "
            f"({attempts} tried; the last: {last_error})",
            kind="prepared_shared_memory",
            diagnostics=(*rejected, last_error),
        )
    coverage = {
        (c.operation, c.batch): proofs[c.id] for c in selected if c.id in proofs
    }
    decisions = list(rejected)
    included = {row.id for row in selected}

    # Retain one ordering per operation/batch/shape/count. Alternative orderings
    # participated in witness search; execution only needs the admitted choice.
    def logical_case(c: PlannedCandidate) -> tuple[OperationKey, int, int, int]:
        source = c.source.layout.candidate
        return c.operation, c.batch, source.case, source.count

    cases = {logical_case(row) for row in selected}
    for candidate in ordered:
        if candidate.id in included or logical_case(candidate) in cases:
            continue
        try:
            additional, admitted = joint_memory([*selected, candidate], plans)
            host.require(additional, admitted)
        except PlanInfeasibleError as error:
            decisions.append(rejection(candidate, error, stage="joint_memory"))
            continue
        selected.append(candidate)
        included.add(candidate.id)
        cases.add(logical_case(candidate))
        selected_plans, memory = additional, admitted
        if candidate.id in proofs:
            coverage.setdefault(
                (candidate.operation, candidate.batch), proofs[candidate.id]
            )
    families = {
        key: tuple(
            sorted(
                (c for c in selected if c.operation == key),
                key=lambda c: (c.duration_seconds, c.id),
            )
        )
        for key in inventory.families
    }
    return PreparedPlans(
        inventory,
        MappingProxyType(families),
        MappingProxyType(selected_plans),
        MappingProxyType(coverage),
        MappingProxyType(memory),
        tuple(decisions),
        attempts,
    )
