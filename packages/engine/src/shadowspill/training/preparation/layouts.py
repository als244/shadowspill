"""Admit each resolved input/output candidate before GPU phase workers start."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from shadowspill.training.inputs.adapters import InputBinding, InputLayout
from shadowspill.training.inputs.coverage import CoverageCertificate, certify_coverage
from shadowspill.training.inputs.coverage_bounds import CoverageError
from shadowspill.training.inputs.staging import (
    InputStaging,
    StagingLimitError,
    admit_input_staging,
)
from shadowspill.training.outputs import OutputLayout
from shadowspill_client.protocol.operations import OperationKey

from .binding import OperationBinding
from .expansion import ScheduleCandidate
from .operations import OperationContract
from .setup import ResolvedSetup


@dataclass(frozen=True)
class LayoutRejection:
    operation: OperationKey
    batch: int
    case: int
    count: int
    stage: str
    reason: str


class LayoutCoverageError(CoverageError):
    """No partial preparation is returned when a requested capability is absent."""

    def __init__(self, message: str, rejected: tuple[LayoutRejection, ...]) -> None:
        super().__init__(message)
        self.rejected = rejected


@dataclass(frozen=True)
class LayoutCandidate:
    operation: OperationKey
    candidate: ScheduleCandidate
    bindings: Mapping[tuple[int, int], OperationBinding]
    outputs: OutputLayout
    staging: InputStaging

    def signature(self) -> dict[str, Any]:
        """One output layout per schedule, not a copy for every worker/buffer."""
        return {
            "operation": self.operation.to_dict(),
            "batch": self.candidate.batch,
            "case": self.candidate.case,
            "count": self.candidate.count,
            "outputs": self.outputs.to_dict(),
            "inputs": [
                {"buffer": buffer, "rank": rank, "binding": binding.inputs.signature()}
                for (buffer, rank), binding in sorted(self.bindings.items())
            ],
            "staging_bytes": self.staging.staging_bytes,
            "input_spool_bytes": self.staging.spool_bytes,
        }


@dataclass(frozen=True)
class PreparedLayouts:
    setup: ResolvedSetup
    families: Mapping[OperationKey, tuple[LayoutCandidate, ...]]
    coverage: Mapping[tuple[OperationKey, int], CoverageCertificate]
    rejected: tuple[LayoutRejection, ...]


def _candidate(
    setup: ResolvedSetup,
    contract: OperationContract,
    candidate: ScheduleCandidate,
    adapters: dict[tuple[Any, ...], InputBinding],
) -> LayoutCandidate:
    schema = setup.context.schema
    placement = setup.context.parallelism.inputs
    assert placement is not None
    schedule = candidate.schedule
    outputs = OutputLayout(schema, contract.outputs, schedule, placement)
    inputs: dict[tuple[int, int], InputBinding] = {}
    bindings = {}
    for buffer, ((partition, _), physical) in enumerate(
        zip(schedule.locations, schedule.buffers, strict=True)
    ):
        for rank in placement.partitions.groups[partition]:
            # Adapters depend on exactly this immutable CPU layout, not on the
            # operation name, count or candidate traversal order.
            key = (physical.max_datums, tuple(sorted(physical.fields.items())), rank)
            if key not in adapters:
                adapters[key] = setup.model.bind_inputs(
                    InputLayout(schema, physical, placement, rank)
                )
            inputs[buffer, rank] = adapters[key]
            bindings[buffer, rank] = OperationBinding(
                setup.model, contract, adapters[key], outputs, buffer
            )
    staging = admit_input_staging(
        schema,
        setup.config.batches[candidate.batch],
        placement,
        schedule,
        inputs,
        setup.config.execution,
    )
    return LayoutCandidate(
        contract.key, candidate, MappingProxyType(bindings), outputs, staging
    )


def prepare_layouts(
    setup: ResolvedSetup, *, max_bindings: int = 65536
) -> PreparedLayouts:
    """Resolve all adapters, output geometry and host bounds, then prove coverage.

    Known staging infeasibility removes only that alternative, with a reason.
    Definition/schema failures propagate. Every operation and explicit batch
    option must still have a constructive coverage witness. This CPU step
    constructs no tensors/models and does not certify native pools or collectives.
    """
    if type(max_bindings) is not int or max_bindings < 1:
        raise ValueError("max_bindings must be a positive integer")
    placement = setup.context.parallelism.inputs
    assert placement is not None
    total = sum(
        len(buffers) * len(group)
        for expansion in setup.expansions.values()
        for candidate in expansion.candidates
        for buffers, group in zip(
            candidate.schedule.partitions, placement.partitions.groups, strict=True
        )
    )
    if total > max_bindings:
        raise ValueError(
            f"operation/worker input bindings exceed configured limit {max_bindings}"
        )
    grouping = setup.model.definition.batching
    if grouping not in ("independent", "grouped"):
        raise CoverageError(
            "whole-batch dependencies require the lower-level custom graph API"
        )
    adapters: dict[tuple[Any, ...], InputBinding] = {}
    families, coverage = {}, {}
    rejected = [
        LayoutRejection(key, row.batch, row.case, row.count, "capacity", row.reason)
        for key, expanded in setup.expansions.items()
        for row in expanded.rejected
    ]
    for key, expanded in setup.expansions.items():
        contract = OperationContract(key, setup.objective)
        admitted = []
        for candidate in expanded.candidates:
            try:
                admitted.append(_candidate(setup, contract, candidate, adapters))
            except StagingLimitError as error:
                rejected.append(
                    LayoutRejection(
                        key,
                        candidate.batch,
                        candidate.case,
                        candidate.count,
                        "staging",
                        str(error),
                    )
                )
        for batch_index, batch in enumerate(setup.config.batches):
            schedules = [
                row.candidate.schedule
                for row in admitted
                if row.candidate.batch == batch_index
            ]
            try:
                if not schedules:
                    raise CoverageError("no input/output candidate fits host staging")
                certificate = certify_coverage(
                    setup.context.schema,
                    batch,
                    placement,
                    schedules,
                    grouping=grouping,
                )
            except CoverageError as error:
                raise LayoutCoverageError(
                    f"{key.operation}/{key.outputs}/{key.mode}, "
                    f"batch {batch_index}: {error}",
                    tuple(rejected),
                ) from error
            coverage[key, batch_index] = certificate
        families[key] = tuple(admitted)
    return PreparedLayouts(
        setup, MappingProxyType(families), MappingProxyType(coverage), tuple(rejected)
    )
