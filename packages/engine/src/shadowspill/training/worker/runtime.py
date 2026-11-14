"""Open one saved worker family through the existing Executor lifecycle."""

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.communication.binding.ranks import RankAssignment
from shadowspill.orchestration.launch.workers import workers
from shadowspill.planning.spill import spill_configuration
from shadowspill.training.definitions.initialization import InitializationRuntime
from shadowspill.training.preparation.family_records import PreparedPlans
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import WorkerBindings
from .execution import WorkerExecution
from .initialization import PreparedInitialization, initialize, prepare_initialization
from .loop import WorkerLoop
from .protocol import WorkerScope, WorkerStartup
from .startup import await_start, report_ready

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
    initialization: PreparedInitialization
    startup: WorkerStartup

    @classmethod
    def resolve(
        cls,
        prepared: PreparedPlans,
        scope: WorkerScope,
        *,
        bundle: ArtifactRef,
        rank_map: Mapping[int, int] | None = None,
    ) -> "PreparedWorker":
        bindings = WorkerBindings(prepared, scope)
        assignment = RankAssignment.resolve(
            bindings.setup.context.parallelism.world_size, rank_map
        )
        plans = prepared.plans[scope.rank]
        for plan in plans.values():
            require_compiled_device(plan.program, 0)
        primary = plans[bindings.primary]
        initial = prepare_initialization(bindings.setup)[scope.rank]
        spill = spill_configuration(
            prepared.memory[scope.rank].spill_bytes,
            primary.memory_budgets.spill_bytes,
            None,
        )
        description = WorkerStartup(
            scope,
            bundle,
            digest(initial.signature()),
            assignment.logical_by_worker,
            primary.memory_budgets.execution_bytes,
            spill.capacity,
            max(initial.description.peak_bytes, bindings.input_bytes),
        )
        return cls(bindings, initial, description)

    @contextmanager
    def open(
        self,
        runtime: InitializationRuntime,
        resolve: Callable[[ArtifactRef], Path],
        collect: Callable[[Any], ArtifactRef | None],
    ) -> Iterator["RunningWorker"]:
        # The agent must have isolated visibility before this import. Each saved
        # process-local compiler ordinal is zero, independent of physical GPU ID.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
            raise ValueError("worker requires one explicitly assigned visible device")
        from shadowspill.execution.executor import Executor

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
                try:
                    initialize(executor, self.initialization, runtime, budget)
                    if executor.spill_capacity_bytes != self.startup.spill_bytes:
                        raise RuntimeError(
                            "worker spill capacity differs from admission"
                        )
                    loop = WorkerLoop(
                        self.startup.scope,
                        lambda command: self.bindings.bind(command, resolve),
                        WorkerExecution(executor, self.bindings, budget, collect),
                        budget,
                    )
                    try:
                        yield RunningWorker(executor, loop, budget)
                    finally:
                        loop.invalidate()
                finally:
                    budget.close()

    def serve(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        runtime: InitializationRuntime,
        resolve: Callable[[ArtifactRef], Path],
        collect: Callable[[Any], ArtifactRef | None],
    ) -> None:
        """Own the private startup/invocation streams, without an idle timeout."""
        with ExitStack() as cleanup:
            cleanup.callback(reader.close)
            cleanup.callback(writer.close)
            await_start(self.startup, reader, writer)
            with self.open(runtime, resolve, collect) as running:
                report_ready(self.startup, writer)
                running.loop.serve(reader, writer)


@dataclass(frozen=True)
class RunningWorker:
    executor: "Executor"
    loop: WorkerLoop
    staging: BufferBudget
