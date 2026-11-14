"""Resolve private worker commands only against the fixed admitted Plan family."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from shadowspill.training.inputs.spool import SpooledFrame
from shadowspill.training.preparation.binding import OperationBinding
from shadowspill.training.preparation.family_records import PreparedPlans
from shadowspill.training.session.controls import (
    CallControls,
    call_controls,
    operation_key,
)
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import Transition
from shadowspill.training.session.worker_inputs import ReadyInputLoader
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.controls import Scalar
from shadowspill_client.protocol.payload import MEDIA_TYPE
from shadowspill_client.protocol.requests import ArtifactRef

from .protocol import CODEC, WorkerCall, WorkerScope


def worker_call(
    call: PreparedCall, transition: Transition, scope: WorkerScope, invocation: int
) -> WorkerCall:
    """Translate the existing coordinator decision, never recompute training state."""
    if scope.rank not in call.worker_plans(transition):
        raise ValueError("worker rank is absent from the prepared call")
    if call.spool is not None and call.spool.closed:
        raise ValueError("prepared input spool is closed")
    frames = (
        ()
        if call.spool is None
        else tuple(
            frame.reference
            for (_, rank), frame in sorted(call.spool.frames.items())
            if rank == scope.rank
        )
    )
    result = WorkerCall(
        scope,
        invocation,
        call.request,
        call.candidate.id,
        transition.skipped,
        transition.accumulate_gradients,
        frames,
    )
    # This must succeed before dispatch; receivers enforce the same envelope.
    CODEC.encode(result)
    return result


@dataclass(frozen=True)
class BoundWorkerCall:
    command: WorkerCall
    plan: str | None
    controls: CallControls
    hyperparams: Mapping[str, Scalar]
    frames: tuple[tuple[SpooledFrame, OperationBinding], ...]

    @property
    def input_bytes(self) -> int:
        return max((frame.layout.size.staging for frame, _ in self.frames), default=0)

    def verify_inputs(self, budget: BufferBudget) -> None:
        """Finish bounded local integrity checks before any participant executes."""
        for frame, _ in self.frames:
            with frame.read(budget):
                pass

    def input_loader(self, budget: BufferBudget) -> ReadyInputLoader | None:
        return (
            ReadyInputLoader(self.frames, self.controls.loss_params, budget)
            if self.frames
            else None
        )


class WorkerBindings:
    """One immutable prepared family and identity for this worker's lifetime.

    The agent owns staged files and supplies the opaque-reference resolver.
    This resolver never sees paths from the network. Binding is CPU-only and
    cannot compile, plan, run an adapter, or import a requested model module.
    """

    def __init__(self, prepared: PreparedPlans, scope: WorkerScope) -> None:
        if scope.rank not in prepared.plans or not prepared.plans[scope.rank]:
            raise ValueError("worker rank has no admitted Plans")
        self.prepared, self.scope = prepared, scope
        self.setup = prepared.inventory.layouts.setup
        self.primary = min(prepared.plans[scope.rank])
        candidates = {}
        for rows in prepared.families.values():
            for row in rows:
                if row.id in candidates:
                    raise ValueError("prepared candidate IDs must be unique")
                candidates[row.id] = row
        self._candidates = MappingProxyType(candidates)

    def bind(
        self, command: WorkerCall, resolve: Callable[[ArtifactRef], Path]
    ) -> BoundWorkerCall:
        if command.scope != self.scope:
            raise ValueError("command belongs to another worker scope")
        key = operation_key(self.setup, command.request)
        controls = call_controls(self.setup, command.request, key)
        candidate = self._candidates.get(command.candidate)
        if candidate is None or self.scope.rank not in candidate.workers:
            raise ValueError("command has no saved candidate for this worker")
        worker = candidate.workers[self.scope.rank]
        if key.operation == "optim_step":
            if candidate.operation.operation != "forward_backward":
                raise ValueError("optimizer requires a prepared backward dependency")
            plan = None if command.skipped else worker.optimizer
        else:
            if candidate.operation != key:
                raise ValueError("saved candidate differs from the requested operation")
            plan = worker.masked if command.skipped else worker.main
        if plan is None and not (key.operation == "optim_step" and command.skipped):
            raise ValueError("saved candidate is missing its required worker form")
        if plan is not None and plan not in self.prepared.plans[self.scope.rank]:
            raise ValueError("worker form has no admitted Plan")
        bindings = (
            []
            if key.operation == "optim_step"
            else [
                binding
                for (_, rank), binding in sorted(
                    candidate.source.layout.bindings.items()
                )
                if rank == self.scope.rank
            ]
        )
        if len(command.frames) != len(bindings):
            raise ValueError("worker frames differ from the complete saved schedule")
        frames = []
        for reference, binding in zip(command.frames, bindings, strict=True):
            if (
                reference.size_bytes != binding.inputs.frame.size.wire
                or reference.media_type != MEDIA_TYPE
            ):
                raise ValueError("input reference differs from its prepared frame")
            frames.append(
                (SpooledFrame(resolve(reference), binding.inputs, reference), binding)
            )
        hyperparams: dict[str, Scalar] = {}
        if key.operation in ("step", "optim_step") and not command.skipped:
            optimizer = self.setup.optimizer
            assert optimizer is not None and plan is not None
            for control, targets in optimizer.definition.control_bindings.items():
                for target in targets:
                    hyperparams[target] = controls.optim[control]
            program = self.prepared.plans[self.scope.rank][plan].program
            scale_binding = program.bindings["gradient_scale_control"]
            if not isinstance(scale_binding, str) or controls.grad_scale is None:
                raise ValueError("saved optimizer form is missing its scale control")
            hyperparams[scale_binding] = controls.grad_scale
        return BoundWorkerCall(
            command, plan, controls, MappingProxyType(hyperparams), tuple(frames)
        )
