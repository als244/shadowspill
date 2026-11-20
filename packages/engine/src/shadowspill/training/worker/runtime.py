"""Open one saved worker family through the existing Executor lifecycle."""

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.diagnostics.progress import event, phase
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.launch.workers import workers
from shadowspill.planning.spill import spill_configuration
from shadowspill.training.components.initialization import InitializationRuntime
from shadowspill.training.inputs.staging import StagingLimitError
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.captures import NsightWindow
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall, WorkerBindings
from .checkpoint_reader import WorkerCheckpointReader, validate_rank
from .checkpoint_records import RankRestore, RankWeights
from .checkpoint_writer import WorkerCheckpointWriter
from .execution import WorkerExecution
from .frozen import fresh_base
from .initialization import (
    PreparedInitialization,
    initialize,
    prepare_frozen_initialization,
    prepare_initialization,
    prepare_temporary_initialization,
    restore_staging,
)
from .loop import WorkerLoop
from .output_files import OutputSink
from .output_frame import WorkerOutputFrame, prepare_output_frames
from .protocol import WorkerCall, WorkerScope, WorkerStartup
from .startup import await_start, report_ready
from .trace_files import TraceBudget, TraceFiles, family_trace_budgets
from .weights_reader import initialize_weights
from .weights_reader import validate_rank as validate_weights_rank

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor


@dataclass(frozen=True)
class PreparedWorker:
    """CPU description from an integrity-checked, strictly rechecked bundle.

    The owning node daemon retains decoded artifacts and reserves node resources before
    launching. The bundle reference identifies that immutable artifact, not a
    filesystem path. Rank assignment never changes the Plan's compiled ordinal.
    """

    bindings: WorkerBindings
    initialization: PreparedInitialization | None
    startup: WorkerStartup
    outputs: Mapping[tuple[str, str], WorkerOutputFrame]
    traces: Mapping[tuple[str, bool], TraceBudget]
    restore: RankRestore | None = None
    frozen_base: FrozenBase | None = None
    initial_weights: RankWeights | None = None

    @property
    def chunk_bytes(self) -> int:
        if self.frozen_base is not None or self.initial_weights is not None:
            return restore_staging(self.initialization, self.startup.staging_bytes)[0]
        return min(
            64 << 20,
            self.startup.staging_bytes // 2
            if self.initialization is None
            else self.initialization.description.chunk_bytes,
        )

    @classmethod
    def resolve(
        cls,
        prepared: PreparedPlans,
        scope: WorkerScope,
        *,
        bundle: ArtifactRef,
        rank_map: Mapping[int, int] | None = None,
        restore: RankRestore | None = None,
        weights: Sequence[ArtifactRef] = (),
        initial_weights: RankWeights | None = None,
    ) -> "PreparedWorker":
        bindings = WorkerBindings(prepared, scope)
        NsysCapture.from_settings(bindings.setup.config.execution.nsys)
        assignment = RankAssignment.resolve(
            bindings.setup.context.parallelism.world_size, rank_map
        )
        plans = prepared.plans[scope.rank]
        for plan in plans.values():
            require_compiled_device(plan.program, 0)
        primary = plans[bindings.primary]
        if restore is not None and initial_weights is not None:
            raise ValueError("worker startup cannot combine weights and a checkpoint")
        saved = restore if restore is not None else initial_weights
        if saved is not None and (
            saved.rank.rank != scope.rank
            or saved.info.world_size != len(assignment.logical_by_worker)
        ):
            raise ValueError("saved initialization differs from worker assignment")
        if initial_weights is not None:
            selected = bindings.setup.config.model.initialize
            if (
                selected is None
                or selected.kind != "artifact"
                or selected.reference != initial_weights.artifact.id
                or initial_weights.info.plans != bundle
            ):
                raise ValueError(
                    "initial weights differ from the selected source or Plans"
                )
        base = (
            fresh_base(prepared, weights) if saved is None else saved.info.frozen_base
        )
        if (
            saved is not None
            and saved.info.frozen_state_id is not None
            and base is None
        ):
            raise ValueError("saved state has no declared frozen-base identity")
        initial: PreparedInitialization | None
        if saved is None:
            initial = prepare_initialization(bindings.setup)[scope.rank]
        elif base is not None:
            initial = prepare_frozen_initialization(bindings.setup, base, scope.rank)
        elif initial_weights is not None:
            initial = prepare_temporary_initialization(prepared, scope.rank)
        else:
            initial = None
        initial_bytes = (
            initial.description.peak_bytes
            if initial is not None
            else min(128 << 20, bindings.setup.config.execution.max_staging_bytes)
        )
        if base is not None or initial_weights is not None:
            _, initial_bytes = restore_staging(
                initial, bindings.setup.config.execution.max_staging_bytes
            )
        if saved is not None and initial_bytes < 2:
            raise StagingLimitError("saved initialization needs two staging chunks")
        outputs = prepare_output_frames(prepared, scope.rank)
        traces = family_trace_budgets(plans)
        staging = max(
            initial_bytes,
            bindings.input_bytes,
            *(frame.frame.write_bytes for frame in outputs.values()),
        )
        if staging > bindings.setup.config.execution.max_staging_bytes:
            raise StagingLimitError(
                "worker outputs exceed the configured staging capacity"
            )
        spill = spill_configuration(
            prepared.memory[scope.rank].spill_bytes,
            primary.memory_budgets.spill_bytes,
            None,
        )
        signature = {
            "checkpoint": None if restore is None else restore.to_dict(),
            "weights": None if initial_weights is None else initial_weights.to_dict(),
            "initializer": None if initial is None else initial.signature(),
            "frozen_base": None if base is None else base.to_dict(),
            "staging_bytes": staging,
        }
        description = WorkerStartup(
            scope,
            bundle,
            digest(signature),
            assignment.logical_by_worker,
            primary.memory_budgets.execution_bytes,
            spill.capacity,
            staging,
            max(trace.host_bytes for trace in traces.values()),
        )
        return cls(
            bindings,
            initial,
            description,
            outputs,
            traces,
            restore,
            base,
            initial_weights,
        )

    @contextmanager
    def open(
        self,
        runtime: InitializationRuntime,
        resolve: Callable[[ArtifactRef], Path],
        outputs: OutputSink,
        *,
        checkpoint_stream: BinaryIO | None = None,
        traces: TraceFiles | None = None,
        capture_progress: Callable[[NsightWindow], None] | None = None,
    ) -> Iterator["RunningWorker"]:
        # The node daemon must have isolated visibility before this import. Each saved
        # process-local compiler ordinal is zero, independent of physical GPU ID.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
            raise ValueError("worker requires one explicitly assigned visible device")
        capture_settings = NsysCapture.from_settings(
            self.bindings.setup.config.execution.nsys
        )
        if (capture_settings is None) != (capture_progress is None):
            raise ValueError("Nsight worker requires its admitted progress file")
        with phase("Import execution runtime"):
            from shadowspill.execution.executor import Executor

            from .randomness import WorkerRandomness

        rank_map = dict(enumerate(self.startup.logical_by_worker))
        with workers(
            len(rank_map),
            rank_map={rank: logical for rank, logical in rank_map.items()},
            init_method=os.environ.get("SHADOWSPILL_RENDEZVOUS"),
        ) as assignment:
            if assignment.logical_rank != self.startup.scope.rank:
                raise ValueError("launcher rank differs from the admitted worker")
            plans = dict(self.bindings.prepared.plans[self.startup.scope.rank])
            primary = plans.pop(self.bindings.primary)
            with ExitStack() as owner:
                with phase("Allocate and bind execution Plans"):
                    executor = owner.enter_context(
                        Executor(
                            primary,
                            plans=plans,
                            rank_map=rank_map,
                            reject_overbudget=self.bindings.setup.config.execution.reject_overbudget,
                        )
                    )
                budget = BufferBudget(self.startup.staging_bytes)
                diagnostics = BufferBudget(self.startup.diagnostics_bytes)
                try:
                    randomness = WorkerRandomness(
                        executor.device, self.startup.scope.rank
                    )
                    reader = WorkerCheckpointReader(
                        executor,
                        budget,
                        resolve,
                        chunk_bytes=min(self.chunk_bytes, budget.max_bytes // 2),
                        world_size=len(rank_map),
                        randomness=randomness,
                        setup=self.bindings.setup,
                        runtime=runtime,
                        frozen_base=self.frozen_base,
                    )
                    with phase("Load model values and optimizer state"):
                        if self.initial_weights is not None:
                            initialize_weights(
                                executor,
                                resolve(self.initial_weights.rank.archive),
                                self.initial_weights,
                                self.initialization,
                                runtime,
                                budget,
                                chunk_bytes=self.chunk_bytes,
                            )
                            randomness.initialize(
                                self.bindings.setup.config.execution.seed
                            )
                        elif self.restore is None:
                            assert self.initialization is not None
                            initialize(executor, self.initialization, runtime, budget)
                            randomness.initialize(
                                self.bindings.setup.config.execution.seed
                            )
                        else:
                            reader.load(
                                self.restore, reader.prepare_source(self.restore)
                            )
                    if executor.spill_capacity_bytes != self.startup.spill_bytes:
                        raise RuntimeError(
                            "worker spill capacity differs from admission"
                        )

                    capture = None
                    if capture_settings is not None:
                        from shadowspill.orchestration.capture._nsys_runtime import (
                            UpdateCapture,
                        )

                        capture = UpdateCapture(
                            capture_settings, publish=capture_progress
                        )

                    execution = WorkerExecution(
                        executor,
                        self.bindings,
                        budget,
                        outputs.collect,
                        restore=reader,
                        randomness=randomness,
                        traces=traces,
                        diagnostics=diagnostics,
                        capture=capture,
                        checkpoints=None
                        if checkpoint_stream is None
                        else WorkerCheckpointWriter(
                            executor,
                            budget,
                            checkpoint_stream,
                            plans=self.startup.bundle,
                            chunk_bytes=self.chunk_bytes,
                            randomness=randomness,
                            frozen_base=lambda: reader.frozen_base,
                        ),
                    )

                    def bind(command: WorkerCall) -> BoundWorkerCall:
                        call = execution.prepare(self.bindings.bind(command, resolve))
                        outputs.check(call)
                        if traces is not None:
                            traces.check(call)
                        return call

                    loop = WorkerLoop(
                        self.startup.scope,
                        bind,
                        execution,
                        budget,
                    )
                    try:
                        yield RunningWorker(executor, loop, budget)
                    finally:
                        loop.invalidate()
                        if capture is not None and capture.state in (
                            "pending",
                            "running",
                        ):
                            # EOF or execution failure is not an all-rank boundary;
                            # never wait in another distributed profiler barrier.
                            capture.failed(
                                "Worker exited before the selected capture "
                                "window completed."
                            )
                finally:
                    budget.close()
                    diagnostics.close()

    def serve(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        runtime: InitializationRuntime,
        resolve: Callable[[ArtifactRef], Path],
        outputs: OutputSink,
        *,
        traces: TraceFiles | None = None,
        capture_progress: Callable[[NsightWindow], None] | None = None,
    ) -> None:
        """Own the private startup/invocation streams, without an idle timeout."""
        with ExitStack() as cleanup:
            cleanup.callback(reader.close)
            cleanup.callback(writer.close)
            if self.restore is not None or self.initial_weights is not None:
                from shadowspill.checkpoint.state import PlanState

                plans = dict(self.bindings.prepared.plans[self.startup.scope.rank])
                primary = plans.pop(self.bindings.primary)
                # Every worker completes this CPU check before any receives the
                # existing startup release into collective initialization.
                state = PlanState(primary, plans, logical_rank=self.startup.scope.rank)
                if self.initial_weights is not None:
                    validate_weights_rank(
                        state,
                        resolve(self.initial_weights.rank.archive),
                        self.initial_weights,
                        chunk_bytes=self.chunk_bytes,
                        world_size=len(self.startup.logical_by_worker),
                        initialization=self.initialization,
                    )
                else:
                    assert self.restore is not None
                    validate_rank(
                        state,
                        resolve(self.restore.rank.archive),
                        self.restore,
                        chunk_bytes=self.chunk_bytes,
                        world_size=len(self.startup.logical_by_worker),
                        require_randomness=True,
                    )
            with phase("Wait for all-worker startup admission"):
                await_start(self.startup, reader, writer)
            with self.open(
                runtime,
                resolve,
                outputs,
                checkpoint_stream=writer,
                traces=traces,
                capture_progress=capture_progress,
            ) as running:
                event("✓ Worker initialized; ready for training")
                report_ready(self.startup, writer)
                running.loop.serve(reader, writer)


@dataclass(frozen=True)
class RunningWorker:
    executor: "Executor"
    loop: WorkerLoop
    staging: BufferBudget
