"""Choose coverage witnesses first, then retain jointly admitted alternatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from types import MappingProxyType
from typing import Literal, cast

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.shared import SharedMemory, shared_memory
from shadowspill.training.inputs.coverage import CoverageCertificate, certify_coverage
from shadowspill.training.inputs.coverage_bounds import CoverageError
from shadowspill_client.protocol.operations import OperationKey

from .family_records import PlannedCandidate, PlanRejection, PreparedPlans
from .templates import TemplateInventory


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
            Literal["independent", "grouped"], setup.model.definition.batching
        ),
    )


def joint_memory(
    candidates: Sequence[PlannedCandidate], plans: Mapping[int, Mapping[str, Plan]]
) -> tuple[dict[int, Mapping[str, Plan]], dict[int, SharedMemory]]:
    selected: dict[int, dict[str, Plan]] = {}
    for candidate in candidates:
        for rank, worker in candidate.workers.items():
            for name in worker.members():
                selected.setdefault(rank, {})[name] = plans[rank][name]
    memory = {}
    frozen: dict[int, Mapping[str, Plan]] = {}
    for rank, members in sorted(selected.items()):
        # Deterministic primary/alias ownership, independent of search completion.
        ordered = dict(sorted(members.items()))
        first = next(iter(ordered))
        memory[rank] = shared_memory(
            ordered[first],
            {name: plan for name, plan in ordered.items() if name != first},
        )
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


def select_family(
    inventory: TemplateInventory,
    candidates: Sequence[PlannedCandidate],
    plans: Mapping[int, Mapping[str, Plan]],
    rejected: Sequence[PlanRejection],
    *,
    max_family_attempts: int,
) -> PreparedPlans:
    """No partial result: every operation/explicit batch retains a construction.

    Search complete witness combinations before adding optional faster layouts.
    A bounded stop remains search exhaustion, never a proof of infeasibility.
    Unexpected shared-state/schema failures propagate rather than pruning work.
    """
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
        try:
            proofs[row.id] = certificate(inventory, row)
        except CoverageError:
            continue
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
        message = "no admitted coverage witness for " + "; ".join(messages)
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
            "no complete coverage witness family fits joint native admission",
            kind="prepared_shared_memory",
            diagnostics=(*rejected, last_error),
        )
    coverage = {(c.operation, c.batch): proofs[c.id] for c in selected}
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
        except PlanInfeasibleError as error:
            decisions.append(rejection(candidate, error, stage="joint_memory"))
            continue
        selected.append(candidate)
        included.add(candidate.id)
        cases.add(logical_case(candidate))
        selected_plans, memory = additional, admitted
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
