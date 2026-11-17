"""Saved preparation facts for small read-only summaries, without recomputation."""

from collections.abc import Iterator
from typing import Any

from shadowspill.training.preparation.artifacts.manifest import rejection_value
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill_client.protocol.plans import PlanSummaryRow, SummaryKind


def summary_rows(prepared: PreparedPlans) -> Iterator[PlanSummaryRow]:
    config = prepared.inventory.layouts.setup.config

    def row(kind: SummaryKind, **values: Any) -> PlanSummaryRow:
        return PlanSummaryRow(kind, values)

    yield row(
        "configuration",
        model=config.model.definition,
        objective=config.objective.definition,
        memory=config.memory.to_dict(),
        preset=config.preparation.preset,
        engine_source=config.engine_source.to_dict(),
        client_source=config.client_source.to_dict(),
        timing="simulated; representative inputs, not measured request latency",
        grouping=prepared.inventory.layouts.setup.model.definition.batching,
        validation_workspace=[
            {"operation": key.to_dict(), "bytes": bound.workspace_bytes}
            for key, bound in prepared.inventory.layouts.staging.items()
        ],
    )
    for index, batch in enumerate(config.batches):
        yield row("batch", index=index, **batch.to_dict())
    for index, case in enumerate(config.preparation.cases):
        yield row("build_case", index=index, **case.to_dict())
    used_templates = {
        use.template
        for candidates in prepared.families.values()
        for candidate in candidates
        for worker in candidate.source.workers.values()
        for use in (worker.main, worker.optimizer)
        if use is not None
    }
    for identity in sorted(used_templates):
        template = prepared.inventory.templates[identity]
        yield row(
            "representative",
            template=template.id,
            pattern=template.representative.to_dict(),
        )
    for candidates in prepared.families.values():
        for candidate in candidates:
            source = candidate.source.layout.candidate
            yield row(
                "candidate",
                id=candidate.id,
                operation=candidate.operation.to_dict(),
                batch=candidate.batch,
                case=source.case,
                count=source.count,
                packing_schedule=source.schedule.to_dict(),
                ordering=None
                if candidate.ordering is None
                else candidate.ordering.to_dict(),
                simulated_seconds=candidate.duration_seconds,
                scheduled_capacities=dict(
                    source.schedule.capacities(
                        prepared.inventory.layouts.setup.context.schema
                    )
                ),
                workers=[
                    {
                        "rank": rank,
                        "main": worker.main,
                        "optimizer": worker.optimizer,
                    }
                    for rank, worker in sorted(candidate.workers.items())
                ],
                templates=[
                    {
                        "rank": rank,
                        "main": worker.main.template,
                        "optimizer": None
                        if worker.optimizer is None
                        else worker.optimizer.template,
                    }
                    for rank, worker in sorted(candidate.source.workers.items())
                ],
            )
    for rank, plans in sorted(prepared.plans.items()):
        for identity, plan in sorted(plans.items()):
            yield row(
                "worker_plan",
                rank=rank,
                id=identity,
                simulated=True,
                declared_spill_bytes=plan.memory_budgets.spill_bytes,
                **plan.summary,
            )
    for rejected in prepared.rejected:
        yield row("rejected_candidate", **rejection_value(rejected))
    for (operation, batch_index), proof in prepared.coverage.items():
        yield row(
            "coverage",
            operation=operation.to_dict(),
            batch=batch_index,
            schedule=proof.schedule.name,
            construction=proof.construction,
            grouping=proof.grouping,
            max_datums=proof.max_datums,
            buffer_count=len(proof.schedule.buffers),
            proof=proof.to_dict(),
        )
    for rank, memory in sorted(prepared.memory.items()):
        yield row(
            "shared_memory",
            rank=rank,
            fixed_slice_bytes=memory.fixed_slice_bytes,
            execution_bytes=memory.execution_bytes,
            spill_bytes=memory.spill_bytes,
        )
