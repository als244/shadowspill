"""Strict reconstruction of a saved capability family from installed CPU hooks."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import plain
from shadowspill.step.ordering import StepDataOrdering
from shadowspill_client._records import integer, text
from shadowspill_client.inputs.coverage import CoverageCertificate
from shadowspill_client.protocol.operations import OperationKey

from .bundle_manifest import candidate_value, manifest
from .bundle_plans import validate_plan
from .family_checks import _collectives, _orderings
from .family_identity import candidate_identity, execution_names, validate_profile
from .family_records import PlannedCandidate, PlanRejection, PreparedPlans
from .family_selection import certificate, joint_memory
from .templates import TemplateCandidate, TemplateInventory

if TYPE_CHECKING:
    from .bundle import TrainingBundle


def _case(row: Mapping[str, Any]) -> tuple[OperationKey, int, int, int]:
    return (
        OperationKey.from_dict(row["operation"]),
        integer(row["batch"], "batch"),
        integer(row["case"], "case"),
        integer(row["count"], "count", minimum=1),
    )


def _candidate(
    row: Mapping[str, Any], source: TemplateCandidate, bundle: TrainingBundle
) -> PlannedCandidate:
    ordering = (
        None
        if row["ordering"] is None
        else StepDataOrdering.from_dict(row["ordering"], "ordering")
    )
    if ordering not in _orderings(source):
        raise ValueError("saved candidate traversal was not prepared")
    workers = dict.fromkeys(source.workers, execution_names(source, ordering))
    if any(
        name not in bundle.plans.get(rank, {})
        for rank, worker in workers.items()
        for name in worker.members()
    ):
        raise ValueError("bundle is missing a required worker Plan dependency")
    result = PlannedCandidate(
        candidate_identity(source, ordering),
        source,
        ordering,
        MappingProxyType(workers),
        max(
            bundle.plans[rank][worker.main].timeline.makespan_ns / 1e9
            for rank, worker in workers.items()
        ),
    )
    if candidate_value(result) != row:
        raise ValueError(
            "saved candidate differs from its prepared layout or Plan identities"
        )
    return result


def recheck_bundle(
    bundle: TrainingBundle, inventory: TemplateInventory
) -> PreparedPlans:
    if bundle.config != inventory.layouts.setup.config:
        raise ValueError("bundle and prepared setup differ")
    metadata = plain(bundle.metadata)
    if set(metadata) != {
        "candidates",
        "coverage",
        "memory",
        "rejected",
        "family_attempts",
    }:
        raise ValueError("unsupported saved family metadata schema")
    available = {
        (
            key,
            row.layout.candidate.batch,
            row.layout.candidate.case,
            row.layout.candidate.count,
        ): row
        for key, rows in inventory.families.items()
        for row in rows
    }
    rows, seen = [], set()
    for saved in metadata["candidates"]:
        case_key = _case(saved)
        if case_key in seen or case_key not in available:
            raise ValueError("bundle has a duplicate or unprepared candidate")
        seen.add(case_key)
        rows.append(_candidate(saved, available[case_key], bundle))
    requirements = {
        (op, batch)
        for op in inventory.families
        for batch in range(len(bundle.config.batches))
    }
    if {(row.operation, row.batch) for row in rows} != requirements:
        raise ValueError(
            "bundle does not preserve every requested operation/batch capability"
        )
    needed_sources = {
        use.template
        for row in rows
        for worker in row.source.workers.values()
        for use in (worker.main, worker.optimizer)
        if use is not None
    }
    if bundle.sources.keys() != needed_sources:
        raise ValueError("bundle must contain exactly the referenced source profiles")
    for source_key, source in bundle.sources.items():
        validate_profile(inventory, source_key, source)
    needed_plans: dict[int, set[str]] = {}
    checked = set()
    for row in rows:
        for rank, worker in row.source.workers.items():
            for role in ("main", "optimizer"):
                use = getattr(worker, role)
                if use is None:
                    continue
                name = getattr(row.workers[rank], role)
                needed_plans.setdefault(rank, set()).add(name)
                if (rank, name) not in checked:
                    validate_plan(
                        inventory,
                        use,
                        row.ordering,
                        bundle.sources[use.template],
                        bundle.plans[rank][name],
                    )
                    checked.add((rank, name))
        _collectives(row.workers, bundle.plans)
    if {rank: set(plans) for rank, plans in bundle.plans.items()} != needed_plans:
        raise ValueError("bundle must contain exactly the referenced worker Plans")
    for plans in bundle.plans.values():
        first = next(iter(plans.values()))
        if any(plan.pool_memory != first.pool_memory for plan in plans.values()):
            raise ValueError("saved worker Plans disagree on pool calibration")
    coverage = {}
    for saved in metadata["coverage"]:
        key = (
            OperationKey.from_dict(saved["operation"]),
            integer(saved["batch"], "batch"),
        )
        if key in coverage or key not in requirements:
            raise ValueError("bundle has duplicate or unexpected coverage")
        proof = CoverageCertificate.from_dict(saved["proof"])
        witness = next(
            (
                row
                for row in rows
                if (row.operation, row.batch) == key
                and row.source.layout.candidate.schedule == proof.schedule
            ),
            None,
        )
        if (
            witness is None
            or certificate(inventory, witness).to_dict() != proof.to_dict()
        ):
            raise ValueError("saved coverage has no matching admitted witness")
        coverage[key] = proof
    selected, memory = joint_memory(rows, bundle.plans)
    families = {
        key: tuple(
            sorted(
                (row for row in rows if row.operation == key),
                key=lambda c: (c.duration_seconds, c.id),
            )
        )
        for key in inventory.families
    }
    rejected = []
    for item in metadata["rejected"]:
        operation, batch, case, count = _case(item)
        if type(item["search_exhausted"]) is not bool:
            raise ValueError("search_exhausted must be boolean")
        if item["ordering"] is not None:
            StepDataOrdering.from_label(item["ordering"])
        rejected.append(
            PlanRejection(
                text(item["candidate"], "candidate"),
                operation,
                batch,
                case,
                count,
                item["ordering"],
                text(item["stage"], "stage"),
                text(item["reason"], "reason"),
                item["search_exhausted"],
            )
        )
    result = PreparedPlans(
        inventory,
        MappingProxyType(families),
        MappingProxyType(selected),
        MappingProxyType(coverage),
        MappingProxyType(memory),
        tuple(rejected),
        integer(metadata["family_attempts"], "family_attempts", minimum=1),
    )
    if manifest(result) != metadata:
        raise ValueError(
            "saved family metadata differs from rechecked coverage/shared admission"
        )
    return result
