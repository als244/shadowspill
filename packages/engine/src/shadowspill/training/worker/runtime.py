"""Open one saved worker family through the existing Executor lifecycle."""

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.orchestration.launch.workers import workers
from shadowspill.planning.spill import spill_configuration
from shadowspill.training.definitions.initialization import InitializationRuntime
from shadowspill.training.inputs.staging import StagingLimitError
from shadowspill.training.preparation.family_records import PreparedPlans
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall, WorkerBindings
from .checkpoint_reader import WorkerCheckpointReader, validate_rank
from .checkpoint_records import RankRestore
from .checkpoint_writer import WorkerCheckpointWriter
from .execution import WorkerExecution
from .initialization import PreparedInitialization, initialize, prepare_initialization
from .loop import WorkerLoop
from .output_files import OutputSink
from .output_frame import WorkerOutputFrame, prepare_output_frames
from .protocol import WorkerCall, WorkerScope, WorkerStartup
from .startup import await_start, report_ready
from .trace_files import TraceBudget, TraceFiles, family_trace_budgets

if TYPE_CHECKING:
    from shadowspill.execution.executor import Executor


@dataclass(frozen=True)
class PreparedWorker:
    """CPU description from an integrity-checked, strictly rechecked bundle.

    The owning agent retains decoded artifacts and reserves node resources before
    launching. The bundle reference identifies that immutable artifact, not a
    filesystem path. Rank assignment never changes the Plan's compiled ordinal.
    """

    bindings: WorkerBindings
    initialization: PreparedInitialization | None
    startup: WorkerStartup
    outputs: Mapping[tuple[str, str], WorkerOutputFrame]
    traces: Mapping[tuple[str, bool], TraceBudget]
    restore: RankRestore | None = None

    @property
    def chunk_bytes(self) -> int:
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
    ) -> "PreparedWorker":
        bindings = WorkerBindings(prepared, scope)
        assignment = RankAssignment.resolve(
            bindings.setup.context.parallelism.world_size, rank_map
        )
        plans = prepared.plans[scope.rank]
        for plan in plans.values():
            require_compiled_device(plan.program, 0)
        primary = plans[bindings.primary]
        if restore is not None and (
            restore.rank.rank != scope.rank
            or restore.info.world_size != len(assignment.logical_by_worker)
        ):
            raise ValueError("checkpoint initialization differs from worker assignment")
        initial = (
            prepare_initialization(bindings.setup)[scope.rank]
            if restore is None
            else None
        )
        initial_bytes = (
            initial.description.peak_bytes
            if initial is not None
            else min(128 << 20, bindings.setup.config.execution.max_staging_bytes)
        )
        if restore is not None and initial_bytes < 2:
            raise StagingLimitError(
                "checkpoint initialization needs two staging chunks"
            )
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
        if initial is None:
            assert restore is not None
            signature = {"checkpoint": restore.to_dict(), "staging_bytes": staging}
        else:
            signature = initial.signature()
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
        return cls(bindings, initial, description, outputs, traces, restore)

    @contextmanager
    def open(
        self,
        runtime: InitializationRuntime,
        resolve: Callable[[ArtifactRef], Path],
        outputs: OutputSink,
        *,
        checkpoint_stream: BinaryIO | None = None,
        traces: TraceFiles | None = None,
    ) -> Iterator["RunningWorker"]:
        # The agent must have isolated visibility before this import. Each saved
        # process-local compiler ordinal is zero, independent of physical GPU ID.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
            raise ValueError("worker requires one explicitly assigned visible device")
        from shadowspill.execution.executor import Executor

        from .randomness import WorkerRandomness

        rank_map = dict(enumerate(self.startup.logical_by_worker))
        with workers(
            len(rank_map),
            rank_map={rank: logical for rank, logical in rank_map.items()},
        ) as assignment:
            if assignment.logical_rank != self.startup.scope.rank:
                raise ValueError("launcher rank differs from the admitted worker")
            plans = dict(self.bindings.prepared.plans[self.startup.scope.rank])
            primary = plans.pop(self.bindings.primary)
            with Executor(
                primary, plans=plans, rank_map=rank_map, reject_overbudget=True
            ) as executor:
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
                    )
                    if self.restore is None:
                        assert self.initialization is not None
                        initialize(executor, self.initialization, runtime, budget)
                        randomness.initialize(self.bindings.setup.config.execution.seed)
                    else:
                        reader.load(self.restore, reader.prepare_source(self.restore))
                    if executor.spill_capacity_bytes != self.startup.spill_bytes:
                        raise RuntimeError(
                            "worker spill capacity differs from admission"
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
                        checkpoints=None
                        if checkpoint_stream is None
                        else WorkerCheckpointWriter(
                            executor,
                            budget,
                            checkpoint_stream,
                            plans=self.startup.bundle,
                            chunk_bytes=self.chunk_bytes,
                            randomness=randomness,
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
    ) -> None:
        """Own the private startup/invocation streams, without an idle timeout."""
        with ExitStack() as cleanup:
            cleanup.callback(reader.close)
            cleanup.callback(writer.close)
            if self.restore is not None:
                from shadowspill.checkpoint.state import PlanState

                plans = dict(self.bindings.prepared.plans[self.startup.scope.rank])
                primary = plans.pop(self.bindings.primary)
                # Every worker completes this CPU check before any receives the
                # existing startup release into collective initialization.
                validate_rank(
                    PlanState(primary, plans, logical_rank=self.startup.scope.rank),
                    resolve(self.restore.rank.archive),
                    self.restore,
                    chunk_bytes=self.chunk_bytes,
                    world_size=len(self.startup.logical_by_worker),
                    require_randomness=True,
                )
            await_start(self.startup, reader, writer)
            with self.open(
                runtime, resolve, outputs, checkpoint_stream=writer, traces=traces
            ) as running:
                report_ready(self.startup, writer)
                running.loop.serve(reader, writer)


@dataclass(frozen=True)
class RunningWorker:
    executor: "Executor"
    loop: WorkerLoop
    staging: BufferBudget
