"""Isolated CUDA task profiling: one compiled task warmed, measured, and traced."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from shadowspill.errors import PlanningError, ProfilingError
from shadowspill.profiling.wall_times import ProfilingWallTimes
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
)
from shadowspill.pytorch.compilation.compiler import CompiledTaskSet
from shadowspill.pytorch.optimizer import OpaqueOptimizerArtifact
from shadowspill.pytorch.state.storage import (
    NamedTensor,
    import_then_fill,
    release_persistent_tensors,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.failures import format_bytes, raise_if_allocator_failed
from shadowspill.task.profiles import TaskMeasurement

from ..executables import ProfileExecutable, ProfileExecutableStore
from ..runner import ProfilableArtifact
from .boundary import AllocatorBoundary
from .measurement import MeasuredTask, measure_task
from .opaque import measure_opaque_optimizer
from .saved_values import resolve_graph_pair_saved_values


@dataclass(frozen=True, slots=True)
class SavedValuePool:
    """Where a plan's saved values are kept while its backwards are measured.

    The spill pool the plan will use, holding them as the plan's own state:
    host memory the pool has already been given, and pinned, rather than more
    of it beside the pool.
    """

    runtime: Runtime
    pool: str
    owning_plan: int


class _SavedValues:
    """One forward's host copies of what it saved, imported as one target."""

    def __init__(self, values: tuple[torch.Tensor, ...]) -> None:
        self.values = values


class TaskProfiler:
    """Warm and measure compiled tasks through an installed ShadowSpill slab."""

    def __init__(
        self,
        library: Any,
        *,
        runtime_handle: int,
        plan_id: int,
        device_ordinal: int,
        warmup_iterations: int = 3,
        sample_iterations: int = 5,
        telemetry_capacity: int = 1_048_576,
        allocation_probe_seeds: int = 1,
        allocation_probe_repetitions: int = 2,
        saved_value_pool: SavedValuePool | None = None,
    ) -> None:
        if warmup_iterations < 1:
            raise ValueError("task profiler requires at least one warmup")
        if sample_iterations < 1:
            raise ValueError("task profiler requires at least one sample")
        if telemetry_capacity < 1:
            raise ValueError("task profiler telemetry capacity must be positive")
        if allocation_probe_seeds < 1 or allocation_probe_repetitions < 2:
            raise ValueError(
                "allocation paths require at least one seed and two repetitions"
            )
        self.boundary = AllocatorBoundary(
            library,
            runtime_handle=runtime_handle,
            plan_id=plan_id,
            device_ordinal=device_ordinal,
            telemetry_capacity=telemetry_capacity,
        )
        self.executables = ProfileExecutableStore(
            device_ordinal=device_ordinal,
            allocation_check=lambda operation: raise_if_allocator_failed(
                library, operation
            ),
        )
        self.warmups = warmup_iterations
        self.samples = sample_iterations
        self.probe_seeds = allocation_probe_seeds
        self.probe_repetitions = allocation_probe_repetitions
        self._profiling_wall_time_ns = 0
        self._entrypoint_warmup_wall_time_ns = 0
        self._saved_value_compilation_wall_time_ns = 0
        self._saved_values: dict[
            tuple[str, str | None, int], tuple[tuple[torch.Tensor, str], ...]
        ] = {}
        self._saved_value_pool = saved_value_pool
        self._pooled_saved_values: list[_SavedValues] = []
        #: Bytes of saved values kept in the spill pool.
        self.saved_value_bytes_in_pool = 0

    @property
    def wall_times(self) -> ProfilingWallTimes:
        """What building entrypoints, measuring, and re-warming cost.

        Compilation charged to the saved-value phase is already subtracted:
        that phase is timed where it runs, so counting it here would attribute
        the same nanoseconds twice.
        """

        return ProfilingWallTimes(
            compilation_ns=(
                self.executables.compilation_wall_time_ns
                - self._saved_value_compilation_wall_time_ns
            ),
            profiling_ns=self._profiling_wall_time_ns,
            cached_warmup_ns=self._entrypoint_warmup_wall_time_ns,
        )

    def measure(self, artifact: ProfilableArtifact) -> TaskMeasurement:
        """Measure one compiled graph or bounded eager optimizer task."""

        if isinstance(artifact, OpaqueOptimizerArtifact):
            return self._timed(
                artifact, lambda: measure_opaque_optimizer(self, artifact)
            )
        if not isinstance(artifact, GraphArtifact):
            raise TypeError(f"unsupported profiling artifact {type(artifact).__name__}")
        digest = artifact.compatibility_digest
        executable = self.executables.get(artifact)
        if not executable.example_arguments and artifact.example_arguments:
            executable = self.executables.with_arguments(executable)
        try:
            measurement = self._timed(
                artifact,
                lambda: measure_task(
                    self,
                    MeasuredTask(executable),
                    execution_provider=(
                        f"{executable.execution_provider}"
                        f"[fx_nodes={executable.graph_node_count}]"
                    ),
                ),
            )
        except BaseException:
            self.executables.remove(digest)
            raise
        self.executables.mark_warmed(digest)
        # The compiled function does not own its example arguments. Keeping
        # every unique contract's device examples alive until take_selected()
        # makes isolated profiling scale with the sum of model-stage inputs,
        # rather than the largest contract. Retain only the executable.
        self.executables.release_occurrence_values(executable)
        return measurement

    def resolve_graph_pair_saved_values(
        self,
        pair: AotGraphPair,
        metadata_digest: str | None = None,
    ) -> AotGraphPair:
        """Populate a backward's saved values from the forward that made them."""

        compilation_before = self.executables.compilation_wall_time_ns
        try:
            return resolve_graph_pair_saved_values(
                self, pair, metadata_digest, self._saved_values
            )
        finally:
            self._saved_value_compilation_wall_time_ns += (
                self.executables.compilation_wall_time_ns - compilation_before
            )

    def keep_saved_values(
        self, copies: tuple[torch.Tensor, ...], fill: Callable[[], None]
    ) -> None:
        """Keep one forward's saved-value copies in the spill pool, filled there.

        The copies are imported as state the plan owns before anything is
        written to them, so ``fill`` writes the values straight into the pool
        (:func:`import_then_fill`). A pool without room for them fails
        planning, naming what was needed: holding them beside the pool would
        spend the host memory the pool was sized to leave free.

        A profiler made without a pool measures no backward, and fills them
        where they are.
        """

        pool = self._saved_value_pool
        named = tuple(
            NamedTensor(f"saved.{index}", copy)
            for index, copy in enumerate(copies)
            if copy.untyped_storage().nbytes()
        )
        if pool is None or not named:
            fill()
            return
        size = sum(item.tensor.untyped_storage().nbytes() for item in named)
        free = int(pool.runtime.pool_statistics(pool.pool).free_bytes)
        if size > free:
            raise PlanningError(
                f"spill pool {pool.pool!r} has no room for the values a forward "
                f"saves, which its backward is measured on: {format_bytes(size)} "
                f"needed, {format_bytes(free)} free"
            )
        held = _SavedValues(copies)
        import_then_fill(
            held,
            named,
            fill,
            runtime=pool.runtime,
            pool=pool.pool,
            owning_plan=pool.owning_plan,
            _allow_in_progress_plan=True,
        )
        self._pooled_saved_values.append(held)
        self.saved_value_bytes_in_pool += size

    def release_host_memory(self) -> None:
        """Give back what measuring kept on the host, once nothing will measure.

        Two things outlive the measurements unless they are released here:

        - The copies of what each forward saved, which its backward was
          measured on (``saved_values``), in the spill pool or beside it. They
          are referenced from the backward artifacts' provenance, which
          planning keeps, so forgetting this memo alone would free nothing:
          each copy is emptied, which lets go of it wherever it is
          referenced, and then the pool is given back what they occupied.
        - Pinned host blocks that the compiler's autotuning copied mutated
          arguments into. ShadowSpill's allocator does not appear in
          PyTorch's memory statistics, so autotuning takes any mutated
          argument to exceed its device budget and copies it to pinned host
          memory, whose blocks PyTorch then keeps cached for the life of the
          process.
        """

        for values in self._saved_values.values():
            for value, _device_type in values:
                value.data = torch.empty(0, dtype=value.dtype)
        self._saved_values.clear()
        pool = self._saved_value_pool
        for held in self._pooled_saved_values:
            if pool is not None:
                release_persistent_tensors(held, runtime=pool.runtime)
        self._pooled_saved_values.clear()
        torch._C._host_emptyCache()

    def take_compiled_tasks(
        self,
        artifacts: Sequence[ProfilableArtifact],
        *,
        progress: Callable[[int, int, str, str], None] | None = None,
    ) -> CompiledTaskSet:
        """Transfer warmed entrypoints and their optimized storage contracts."""

        return self.executables.take_selected(
            artifacts, warmup=self._warm_selected_entrypoint, progress=progress
        )

    def _warm_selected_entrypoint(
        self,
        executable: ProfileExecutable,
        digest: str,
    ) -> None:
        stream = self.boundary.stream()
        started = time.perf_counter_ns()
        try:
            for _ in range(self.warmups):
                self.boundary.invoke(executable, stream)
            self.boundary.require_idle(problem=f"compiled entrypoint {digest}")
        except ProfilingError:
            raise
        except BaseException as error:
            raise profiling_error(executable.artifact, error) from error
        finally:
            self._entrypoint_warmup_wall_time_ns += time.perf_counter_ns() - started

    def _timed(
        self,
        artifact: GraphArtifact | OpaqueOptimizerArtifact,
        measure: Callable[[], TaskMeasurement],
    ) -> TaskMeasurement:
        """Charge one measurement to the profiling clock, naming what failed."""

        started = time.perf_counter_ns()
        try:
            measurement = measure()
        except ProfilingError:
            raise
        except BaseException as error:
            raise profiling_error(artifact, error) from error
        finally:
            elapsed = time.perf_counter_ns() - started
            self._profiling_wall_time_ns += elapsed
        return replace(measurement, profiling_wall_time_ns=elapsed)


def profiling_error(
    artifact: GraphArtifact | OpaqueOptimizerArtifact,
    cause: BaseException,
) -> ProfilingError:
    """Name the structural contract a profiling failure belongs to."""

    kind: str
    if isinstance(artifact, GraphArtifact):
        kind = artifact.kind
        operators = tuple(artifact.operator_targets)
    else:
        kind = "opaque_optimizer"
        operators = ()
    return ProfilingError(
        "ShadowSpill failed to profile structural contract "
        f"{artifact.compatibility_digest} "
        f"(kind={kind}, operators=[{', '.join(operators) or 'none'}]): {cause}",
        structural_contract=artifact.compatibility_digest,
        task_kind=kind,
        operators=operators,
    )


__all__ = [
    "ProfilingWallTimes",
    "SavedValuePool",
    "TaskProfiler",
    "profiling_error",
]
