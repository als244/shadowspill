"""Bounded per-invocation artifacts using the existing step diagnostics codec."""

from __future__ import annotations

import ctypes
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, BinaryIO, cast
from zipfile import ZIP_STORED, ZipFile

from shadowspill.artifacts.plan import Plan
from shadowspill.communication.contracts import collective_measurement
from shadowspill.orchestration.launch.publication import (
    OutputSpec,
    OutputWriter,
    require_output,
    write_output,
)
from shadowspill.orchestration.launch.staging import COPY_BYTES
from shadowspill.profiling.records import measured_task
from shadowspill.runtime.abi import AllocationEvent, TraceEvent
from shadowspill.runtime.plan.common import TaskMemoryEnvelope
from shadowspill.runtime.plan.tracing import TraceCapacities, trace_capacities
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.results import ArtifactIssue

from .bindings import BoundWorkerCall
from .protocol import ticket

if TYPE_CHECKING:
    from shadowspill.diagnostics.step import StepDiagnostics
    from shadowspill.pytorch.invocation import InvocationResult

MEDIA_TYPE = "application/zip"


@dataclass(frozen=True)
class TraceBudget:
    capacities: TraceCapacities
    host_bytes: int
    file_bytes: int


def plan_trace_budget(plan: Plan) -> TraceBudget:
    """Inspect saved contracts without loading callables or opening a device.

    The service sizes the existing bounded rings from the actual selected tasks,
    with a floor of 1,024 records. Overflow remains explicit in the existing trace
    format. Direct Executor callers retain their ordinary default capacity.
    Python/backend overhead is a conservative deployment allowance, qualified
    alongside WorkerHost; it is not a universal allocator peak proof.
    """
    tasks = tuple(plan._costed_program.selected_tasks(plan.selections))
    keys = {task.task_id: task.profile_id for task in plan.program.tasks}
    contracts = {}
    envelopes = []
    for task in tasks:
        contract = None
        if task.requires_entrypoint:
            key = keys[task.task_id]
            if key not in contracts:
                measurement = (
                    collective_measurement(plan.program, key, plan.communication_info)
                    if key in plan.program.bindings.get("collective_tasks", {})
                    else measured_task(
                        plan.profiled_program.task_profiles[key],
                        plan.program.bindings["manifests"][key],
                    )
                )
                contracts[key] = measurement.allocation_contract
            contract = contracts[key]
        envelopes.append((task, TaskMemoryEnvelope(allocation_contract=contract)))
    capacities = trace_capacities(envelopes, plan.schedule.actions, floor=1024)
    names = plan.program.bindings.get("task_names", {})
    labels = sum(
        len(str(names.get(task.task_id, task.task_id)).encode()) for task in tasks
    ) + sum(len(item.alias_group_id.encode()) for item in plan._costed_program.objects)
    world = (
        1 if plan.program.parallelism is None else plan.program.parallelism.world_size
    )
    # Covers structured task/transfer dictionaries, collective and host-copy
    # rows, string escaping, and bounded invocation association metadata.
    metadata = (
        (64 << 10)
        + (len(tasks) + len(plan.schedule.actions)) * (16 << 10)
        + 32 * labels
        + capacities.events * (4096 + 32 * world)
    )
    raw = capacities.events * ctypes.sizeof(
        TraceEvent
    ) + capacities.allocation_events * ctypes.sizeof(AllocationEvent)
    # The uncompressed column archive is smaller than the ABI rows; include
    # ZIP/NPY headers. Runtime rings, copied rows, NumPy column scratch, Python
    # projections and encoder chunks can coexist before publication completes.
    file_bytes = metadata + raw + (1 << 20)
    host_bytes = 8 * metadata + 4 * raw + (32 << 20)
    return TraceBudget(capacities, host_bytes, file_bytes)


def family_trace_budgets(
    plans: Mapping[str, Plan],
) -> Mapping[tuple[str, bool], TraceBudget]:
    return MappingProxyType(
        {
            (name, accumulated): plan_trace_budget(path)
            for name, plan in plans.items()
            for accumulated, path in ((False, plan), (True, plan._accumulation_plan))
            if path is not None
        }
    )


class _Stream:
    """Adapt ZIP writes to the admitted chunk writer without whole-file copies."""

    def __init__(self, output: OutputWriter) -> None:
        self.output, self.size = output, 0
        self.digest = hashlib.sha256()

    def write(self, data: bytes | memoryview, /) -> int:
        with memoryview(data).cast("B") as view:
            offset = 0
            while offset < len(view):
                part = view[offset : offset + COPY_BYTES]
                try:
                    count = self.output.write(part)
                    self.digest.update(part[:count])
                finally:
                    part.release()
                offset += count
                self.size += count
            return offset

    def tell(self) -> int:
        return self.size

    def flush(self) -> None:
        pass


def _json(output: IO[bytes], value: object) -> None:
    for part in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(
        value
    ):
        output.write(part.encode())


class TraceFiles:
    def __init__(
        self,
        budgets: Mapping[tuple[str, bool], TraceBudget],
        *,
        bundle: ArtifactRef,
        root: Path,
    ) -> None:
        self.budgets, self.bundle, self.root = budgets, bundle, root

    def budget(self, call: BoundWorkerCall) -> TraceBudget | None:
        if not call.command.request.runtime_trace or call.plan is None:
            return None
        return self.budgets[call.plan, call.command.accumulate_gradients]

    def spec(self, call: BoundWorkerCall) -> OutputSpec | None:
        budget = self.budget(call)
        return (
            None
            if budget is None
            else OutputSpec(
                "trace." + ticket(call.command).call_sha256,
                budget.file_bytes,
                MEDIA_TYPE,
                exact_size=False,
            )
        )

    def check(self, call: BoundWorkerCall) -> None:
        spec = self.spec(call)
        if spec is not None:
            require_output(self.root, spec)

    def collect(
        self, call: BoundWorkerCall, invocation: InvocationResult[object]
    ) -> tuple[ArtifactRef | None, ArtifactIssue | None]:
        spec = self.spec(call)
        if spec is None:
            raise ValueError("trace collection requires a prepared invocation")
        try:
            diagnostics = invocation.diagnostics
            if diagnostics is None:
                raise ValueError("traced invocation omitted its diagnostic evidence")
            reference = write_output(
                self.root,
                spec,
                lambda output: self._write(call, diagnostics, output, spec),
            )
            return reference, None
        except Exception as error:
            # Only called after required execution/terminal transfers succeeded.
            # The failed partial file remains owned by the agent through cleanup.
            message = (str(error).strip() or type(error).__name__)[:1024].rstrip()
        return None, ArtifactIssue("runtime_trace_failed", message, spec.id)

    def _write(
        self,
        call: BoundWorkerCall,
        diagnostics: StepDiagnostics,
        output: OutputWriter,
        spec: OutputSpec,
    ) -> ArtifactRef:
        from shadowspill.diagnostics.step_file import write_step_events

        command = call.command
        stream = _Stream(output)
        with ZipFile(cast(BinaryIO, stream), "w", compression=ZIP_STORED) as archive:
            with archive.open("manifest.json", "w", force_zip64=True) as member:
                _json(
                    member,
                    {
                        "version": 1,
                        "kind": "runtime_trace",
                        "request": command.request.identity.to_dict(),
                        "operation": command.request.operation,
                        "history_id": command.history_id,
                        "completed_updates_before": command.completed_updates,
                        "rank": command.scope.rank,
                        "invocation": command.invocation,
                        "plans": self.bundle.to_dict(),
                        "plan": call.plan,
                        "accumulate_gradients": command.accumulate_gradients,
                        "skipped": command.skipped,
                    },
                )
            with archive.open("step.json", "w", force_zip64=True) as member:
                _json(member, diagnostics.as_dict(events=False))
            with archive.open("step.events.npz", "w", force_zip64=True) as member:
                write_step_events(member, diagnostics)
        return ArtifactRef(
            spec.id, stream.digest.hexdigest(), stream.size, spec.media_type
        )
