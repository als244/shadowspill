"""Select saved capabilities and finish whole-call CPU inputs before acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from shadowspill.training.inputs.packing import (
    PackedAssignment,
    PackingError,
    pack_batch,
)
from shadowspill.training.inputs.spool import InputSpool
from shadowspill.training.inputs.validation import ValidatedBatch, validate_batch
from shadowspill.training.inputs.values import VALIDATION_SCRATCH_BYTES
from shadowspill.training.preparation.family_records import (
    PlannedCandidate,
    PreparedPlans,
)
from shadowspill.training.preparation.operations import OperationContract
from shadowspill_client import BatchConfig
from shadowspill_client._records import number
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.protocol.payload import BatchSnapshot, snapshot_batch
from shadowspill_client.protocol.requests import STATE_OPERATIONS, OperationRequest
from shadowspill_client.protocol.results import RequestStatus

from .controls import CallControls, call_controls, operation_key
from .inputs import RequestInputs
from .state import Transition


@dataclass
class PreparedCall:
    request: OperationRequest
    key: OperationKey | None
    candidate: PlannedCandidate | None
    controls: CallControls
    participates: bool | None
    budget: BufferBudget
    assignment: PackedAssignment | None = None
    spool: InputSpool | None = None
    _workspace: BufferLease | None = None
    _empty_snapshot: BatchSnapshot | None = None
    closed: bool = False
    ranks: tuple[int, ...] = ()

    def worker_plans(self, transition: Transition) -> Mapping[int, str | None]:
        """Select main/masked/optimizer forms using the existing state decision."""
        if self.closed:
            raise RuntimeError("prepared request inputs are closed")
        if (
            transition.request_id != self.request.identity.request_id
            or transition.command != self.request.operation
            or (
                self.participates is not None
                and transition.participates != self.participates
            )
        ):
            raise ValueError("operation transition differs from its prepared request")
        if self.candidate is None:
            if (
                self.key is not None
                or self.request.operation not in STATE_OPERATIONS
                or not self.ranks
            ):
                raise ValueError("prepared state command requires its worker ranks")
            return MappingProxyType(dict.fromkeys(self.ranks))
        assert self.key is not None
        result = {}
        for rank, worker in self.candidate.workers.items():
            if self.key.operation == "optim_step":
                plan = None if transition.skipped else worker.optimizer
            else:
                plan = worker.masked if transition.skipped else worker.main
            if plan is None and not (
                self.key.operation == "optim_step" and transition.skipped
            ):
                raise ValueError("prepared family is missing its required worker form")
            result[rank] = plan
        return MappingProxyType(result)

    def close(self) -> None:
        if self.spool is not None:
            self.spool.close()
            self.spool = None
        self.assignment = None
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None
        if self._empty_snapshot is not None:
            self._empty_snapshot.close()
            self._empty_snapshot = None
        self.closed = True


class PreparedRequests:
    """Runtime uses only this rechecked, jointly admitted family for its lifetime.

    This is the preprocessing callback for RequestInputs, not another execution
    loop. Construction takes PreparedPlans from preparation/bundle rechecking.
    It never captures, compiles, plans, imports model code by request, or invokes
    a model. Complete CPU preprocessing and spooling precede durable acceptance.
    """

    def __init__(self, prepared: PreparedPlans, *, cache_root: Path) -> None:
        self.prepared = prepared
        self.setup = prepared.inventory.layouts.setup
        self.cache_root = cache_root
        self._families = {
            key: tuple(sorted(rows, key=lambda row: (row.duration_seconds, row.id)))
            for key, rows in prepared.families.items()
        }
        if self._families.keys() != {
            key for key in self.setup.config.operations if key.operation != "optim_step"
        }:
            raise ValueError("prepared request family differs from fixed capabilities")
        for key, rows in self._families.items():
            for batch in range(len(self.setup.config.batches)):
                proof = prepared.coverage[key, batch]
                if proof.schedule not in [
                    row.source.layout.candidate.schedule
                    for row in rows
                    if row.operation == key and row.batch == batch
                ]:
                    raise ValueError(
                        "prepared request family is missing its coverage witness"
                    )
        optimizer = [
            row
            for key, rows in self._families.items()
            if key.operation == "forward_backward"
            for row in rows
            if all(w.optimizer is not None for w in row.workers.values())
        ]
        self._optimizer = (
            min(optimizer, key=self._optimizer_cost) if optimizer else None
        )
        if (
            OperationKey("optim_step", None) in self.setup.config.operations
            and self._optimizer is None
        ):
            raise ValueError("prepared family has no optimizer-only dependency")
        self._limits = tuple(
            {name: number(value, name) for name, value in batch.limits.items()}
            for batch in self.setup.config.batches
        )
        # This union permits schema/value validation once. Actual admission below
        # still requires one whole original BatchConfig; dimensions from separate
        # options must never be combined into a newly advertised capacity.
        self._validation_limits = BatchConfig(
            limits={
                name: max(row[name] for row in self._limits) for name in self._limits[0]
            }
        )

    def _optimizer_cost(self, row: PlannedCandidate) -> tuple[float, str]:
        costs = []
        for rank, worker in row.workers.items():
            assert worker.optimizer is not None
            costs.append(
                self.prepared.plans[rank][worker.optimizer].timeline.makespan_ns / 1e9
            )
        return max(costs), row.id

    def register(
        self, inputs: RequestInputs[PreparedCall], request: OperationRequest
    ) -> RequestStatus:
        """Derive upload limits from preparation, never from submitted byte claims."""
        if request.operation in STATE_OPERATIONS:
            return inputs.register(request, staging_bytes=1, envelope=None)
        try:
            key = operation_key(self.setup, request)
        except ValueError as error:
            # Persist a normal ordered rejection without reading a rejected body.
            # A known valid bound supplies a temporary metadata slot; it does not
            # grant an unprepared capability or accept any of the payload bytes.
            rejected_bound = next(
                iter(self.prepared.inventory.layouts.staging.values())
            )
            status = inputs.register(
                request,
                staging_bytes=rejected_bound.staging_bytes,
                envelope=rejected_bound.envelope
                if request.payload is not None
                else None,
            )
            return (
                inputs.reject(
                    request.identity.request_id,
                    code="unprepared_operation",
                    message=str(error),
                )
                if status.state == "uploading"
                else status
            )
        bound = self.prepared.inventory.layouts.staging.get(key)
        return inputs.register(
            request,
            staging_bytes=1 if bound is None else bound.staging_bytes,
            envelope=None if bound is None else bound.envelope,
        )

    def __call__(
        self,
        request: OperationRequest,
        snapshot: BatchSnapshot | None,
        budget: BufferBudget,
    ) -> PreparedCall:
        if request.operation in STATE_OPERATIONS:
            if snapshot is not None:
                raise ValueError("state commands have no tensor input")
            return PreparedCall(
                request,
                None,
                None,
                CallControls(),
                None,
                budget,
                ranks=tuple(self.prepared.plans),
            )
        key = operation_key(self.setup, request)
        controls = call_controls(self.setup, request, key)
        if key.operation == "optim_step":
            if snapshot is not None:
                raise ValueError("optimizer-only requests have no tensor input")
            assert self._optimizer is not None
            return PreparedCall(request, key, self._optimizer, controls, None, budget)
        bound = self.prepared.inventory.layouts.staging[key]
        empty = None
        workspace = None
        try:
            if snapshot is None:
                if request.datum_count or request.payload is not None:
                    raise ValueError("request is missing its complete input payload")
                snapshot = empty = snapshot_batch([], budget, timeout=0)
            if snapshot.closed or len(snapshot.datums) != request.datum_count:
                raise ValueError("owned payload differs from the request Datum count")
            workspace = budget.reserve(
                bound.workspace_bytes + VALIDATION_SCRATCH_BYTES, timeout=0
            )
            contract = OperationContract(key, self.setup.objective)
            validated = validate_batch(
                snapshot,
                self.setup.context.schema,
                self._validation_limits,
                required_paths=contract.required_paths,
            )
            participates = (
                None
                if key.operation == "forward"
                else bool(
                    validated.participating_positions(self.setup.config.participation)
                )
            )
            workspace.shrink(bound.workspace_bytes)
            candidate, assignment = self._select(key, validated)
            spool = InputSpool(
                assignment,
                {
                    key: value.inputs
                    for key, value in candidate.source.layout.bindings.items()
                },
                budget,
                root=self.cache_root,
            )
            return PreparedCall(
                request,
                key,
                candidate,
                controls,
                participates,
                budget,
                assignment,
                spool,
                workspace,
                empty,
            )
        except BaseException:
            if workspace is not None:
                workspace.close()
            if empty is not None:
                empty.close()
            raise

    def _select(
        self, key: OperationKey, validated: ValidatedBatch
    ) -> tuple[PlannedCandidate, PackedAssignment]:
        allowed = {
            i
            for i, limits in enumerate(self._limits)
            if all(
                total <= limits[name] for name, total in validated.capacities.items()
            )
        }
        if not allowed:
            raise ValueError("input capacities fit no requested batch option")
        groups = self.setup.model.group_inputs(
            replace(validated, config=self.setup.config.batches[min(allowed)])
        )
        for candidate in self._families[key]:
            if candidate.batch not in allowed:
                continue
            limits = self.setup.config.batches[candidate.batch]
            batch = replace(validated, config=limits)
            schedule = candidate.source.layout.candidate.schedule
            proof = self.prepared.coverage[key, candidate.batch]
            try:
                assignment = pack_batch(
                    batch,
                    proof.schema,
                    limits,
                    proof.placement,
                    schedule,
                    groups=groups,
                )
            except PackingError:
                if schedule != proof.schedule:
                    continue
                assignment = proof.pack(batch, groups=groups)
            return candidate, assignment
        raise ValueError("prepared coverage witness did not cover the validated input")
