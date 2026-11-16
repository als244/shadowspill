"""CPU validation against immutable prepared layouts, before input upload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._records import integer, number, plain
from .buffers import BufferBudget
from .datum import Datum
from .inputs.coverage import CoverageCertificate
from .inputs.selection import PackingCandidate, select_packing, validate_snapshot
from .inputs.shapes import PackingSchedule
from .inputs.values import VALIDATION_SCRATCH_BYTES
from .placement import InputPlacement
from .protocol.codec import digest, encode
from .protocol.operations import OperationKey
from .protocol.payload import BatchSnapshot, snapshot_batch
from .protocol.plans import PlanSummaryPage, summary_id
from .protocol.requests import ArtifactRef
from .protocol.training import ResolvedTraining

if TYPE_CHECKING:
    from .service import ServiceClient

MAX_VALIDATION_METADATA = 16 << 20


@dataclass(frozen=True)
class BatchValidation:
    operation: OperationKey
    capacities: Mapping[str, int | float]
    packing_checked: bool
    candidate: str | None
    scheduled_capacities: Mapping[str, int | float]
    note: str | None = None


class BatchValidator:
    """One bounded metadata cache and serialized scratch owner per training handle."""

    def __init__(
        self,
        service: ServiceClient,
        config: ResolvedTraining,
        plans: ArtifactRef | None,
    ) -> None:
        self.service, self.config, self.plans = service, config, plans
        self.lock = Lock()
        self.candidates: dict[OperationKey, list[PackingCandidate]] = {}
        self.proofs: dict[OperationKey, dict[int, CoverageCertificate]] = {}
        self.grouping: str | None = None
        self.workspaces: dict[OperationKey, int] = {}

    def _load(self) -> None:
        if self.grouping is not None:
            return
        if self.plans is None:
            raise ValueError("batch validation requires prepared Plans")
        candidates: dict[OperationKey, list[PackingCandidate]] = {}
        proofs: dict[OperationKey, dict[int, CoverageCertificate]] = {}
        grouping = None
        workspaces: dict[OperationKey, int] = {}
        costs = {}
        placement = InputPlacement.from_dict(plain(self.config.parallelism["inputs"]))
        expected = digest(self.config)
        page, consumed, count = 0, 0, 0
        seen = set()
        while True:
            summary = self.service._request(
                "GET", "/v1/artifacts/" + summary_id(self.plans, page), PlanSummaryPage
            )
            if (summary.plans, summary.page, summary.configuration_sha256) != (
                self.plans,
                page,
                expected,
            ):
                raise ValueError("validation metadata differs from the prepared run")
            consumed += 8 * len(encode(summary))
            if consumed > MAX_VALIDATION_METADATA:
                raise ValueError("batch validation metadata exceeds its client bound")
            for row in summary.rows:
                values = plain(row.values)
                if row.kind == "configuration":
                    if grouping is not None:
                        raise ValueError("duplicate validation configuration")
                    grouping = values["grouping"]
                    for item in values["validation_workspace"]:
                        key = OperationKey.from_dict(item["operation"])
                        size = integer(item["bytes"], "validation workspace")
                        if (
                            key in workspaces
                            or size + VALIDATION_SCRATCH_BYTES
                            > self.config.execution.max_staging_bytes
                        ):
                            raise ValueError("invalid prepared validation workspace")
                        workspaces[key] = size
                elif row.kind == "candidate":
                    key = OperationKey.from_dict(values["operation"])
                    identity = values["id"]
                    if identity in seen or key not in self.config.operations:
                        raise ValueError("duplicate or unexpected packing candidate")
                    seen.add(identity)
                    batch = values["batch"]
                    if type(batch) is not int or not 0 <= batch < len(
                        self.config.batches
                    ):
                        raise ValueError(
                            "packing candidate names an unknown batch option"
                        )
                    candidate = PackingCandidate(
                        identity,
                        batch,
                        PackingSchedule.from_dict(values["packing_schedule"]),
                    )
                    candidates.setdefault(key, []).append(candidate)
                    costs[identity] = number(
                        values["simulated_seconds"], "candidate duration", minimum=0
                    )
                    count += 1
                elif row.kind == "coverage":
                    key = OperationKey.from_dict(values["operation"])
                    batch = values["batch"]
                    by_batch = proofs.setdefault(key, {})
                    if batch in by_batch:
                        raise ValueError("duplicate packing fallback")
                    by_batch[batch] = CoverageCertificate.from_dict(values["proof"])
            if summary.next_page is None:
                if count != summary.candidate_count:
                    raise ValueError("incomplete prepared candidate metadata")
                break
            page = summary.next_page
        if grouping not in ("independent", "grouped"):
            raise ValueError("prepared grouping contract is unavailable")
        operations = {
            key for key in self.config.operations if key.operation != "optim_step"
        }
        if candidates.keys() != operations or workspaces.keys() != operations:
            raise ValueError("validation metadata is missing prepared capabilities")
        for rows in candidates.values():
            if {row.batch for row in rows} != set(range(len(self.config.batches))):
                raise ValueError("validation metadata is missing batch alternatives")
            for candidate in rows:
                candidate.schedule.validate(
                    self.config.inputs, self.config.batches[candidate.batch], placement
                )
            rows.sort(key=lambda row: (costs[row.id], row.id))
        for key, by_batch in proofs.items():
            for batch, proof in by_batch.items():
                if (
                    key not in operations
                    or type(batch) is not int
                    or not 0 <= batch < len(self.config.batches)
                    or proof.schema != self.config.inputs
                    or proof.config != self.config.batches[batch]
                    or proof.placement != placement
                    or proof.grouping != grouping
                    or proof.schedule
                    not in [
                        row.schedule for row in candidates[key] if row.batch == batch
                    ]
                ):
                    raise ValueError(
                        "packing fallback differs from the prepared contract"
                    )
        self.candidates, self.proofs, self.grouping = candidates, proofs, grouping
        self.workspaces = workspaces

    def _key(
        self, operation: str, outputs: str | None, mode: str | None
    ) -> OperationKey:
        key = next(
            (
                key
                for key in self.config.operations
                if key.operation == operation
                and key.mode
                == (mode or ("evaluation" if operation == "forward" else "training"))
                and (outputs is None or key.outputs == outputs)
            ),
            None,
        )
        if key is None or key.operation == "optim_step":
            raise ValueError(
                "operation/output/mode was not prepared for batch validation"
            )
        return key

    def snapshot(
        self,
        batch: Sequence[Datum],
        budget: BufferBudget,
        operation: str,
        outputs: str | None,
        mode: str | None,
    ) -> BatchSnapshot:
        """Own input bytes and their validation workspace under one staging lease."""
        with self.lock:
            key = self._key(operation, outputs, mode)
            self._load()
            workspace = self.workspaces[key] + VALIDATION_SCRATCH_BYTES
        return snapshot_batch(batch, budget, workspace_bytes=workspace)

    def validate(
        self,
        snapshot: BatchSnapshot,
        operation: str,
        outputs: str | None,
        mode: str | None,
    ) -> BatchValidation:
        with self.lock:
            key = self._key(operation, outputs, mode)
            validated = validate_snapshot(snapshot, self.config, key)
            self._load()
            capacities = MappingProxyType(dict(validated.capacities))
            if self.grouping == "grouped":
                return BatchValidation(
                    key,
                    capacities,
                    False,
                    None,
                    MappingProxyType({}),
                    "Values and limits checked; the installed model groups inputs "
                    "on the server before packing.",
                )
            placement = InputPlacement.from_dict(
                plain(self.config.parallelism["inputs"])
            )
            candidate, assignment = select_packing(
                validated,
                self.config.batches,
                placement,
                self.candidates.get(key, ()),
                proofs=self.proofs.get(key, {}),
            )
            return BatchValidation(
                key,
                capacities,
                True,
                candidate.id,
                MappingProxyType(
                    dict(assignment.schedule.capacities(self.config.inputs))
                ),
            )
