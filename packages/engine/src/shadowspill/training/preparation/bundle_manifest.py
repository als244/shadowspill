"""Canonical descriptions of saved capabilities, coverage and shared admission."""

from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.planning.shared import SharedMemory

from .family_records import PlannedCandidate, PlanRejection, PreparedPlans


def candidate_value(candidate: PlannedCandidate) -> dict[str, Any]:
    source = candidate.source.layout.candidate
    return {
        "id": candidate.id,
        "operation": candidate.operation.to_dict(),
        "batch": candidate.batch,
        "case": source.case,
        "count": source.count,
        "ordering": None
        if candidate.ordering is None
        else candidate.ordering.to_dict(),
        "layout": candidate.source.layout.signature(),
        "workers": [
            {"rank": rank, "main": w.main, "optimizer": w.optimizer}
            for rank, w in sorted(candidate.workers.items())
        ],
        "duration_seconds": candidate.duration_seconds,
    }


def memory_value(rank: int, primary: str, memory: SharedMemory) -> dict[str, Any]:
    return {
        "rank": rank,
        "primary": primary,
        "aliases": plain(memory.aliases),
        "host": plain(memory.host),
        "fixed_slice_bytes": memory.fixed_slice_bytes,
        "execution_bytes": memory.execution_bytes,
        "spill_bytes": memory.spill_bytes,
    }


def rejection_value(row: PlanRejection) -> dict[str, Any]:
    return {
        "candidate": row.candidate,
        "operation": row.operation.to_dict(),
        "batch": row.batch,
        "case": row.case,
        "count": row.count,
        "ordering": row.ordering,
        "stage": row.stage,
        "reason": row.reason,
        "search_exhausted": row.search_exhausted,
    }


def manifest(prepared: PreparedPlans) -> dict[str, Any]:
    return {
        "candidates": [
            candidate_value(row)
            for row in sorted(
                (c for rows in prepared.families.values() for c in rows),
                key=lambda row: row.id,
            )
        ],
        "coverage": [
            {"operation": op.to_dict(), "batch": batch, "proof": proof.to_dict()}
            for op in prepared.families
            for (key, batch), proof in prepared.coverage.items()
            if key == op
        ],
        "memory": [
            memory_value(rank, min(prepared.plans[rank]), value)
            for rank, value in sorted(prepared.memory.items())
        ],
        "rejected": [rejection_value(row) for row in prepared.rejected],
        "family_attempts": prepared.family_attempts,
    }
