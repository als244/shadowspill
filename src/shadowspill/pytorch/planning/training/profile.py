"""Profiling: every structurally unique task a step of this size can run, compiled
and measured once, independent of the walk the step will take."""

from __future__ import annotations

from dataclasses import dataclass, replace

from shadowspill.errors import (
    PlanningError,
)
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.optimizer import (
    OptimizerCapture,
    OptimizerTaskArtifact,
)
from shadowspill.pytorch.profiling import (
    ProfileEnvironment,
    ProfilingResult,
    ResolvedTaskManifests,
    profile_environment,
    profile_unique_artifacts,
    resolve_task_manifests,
    validate_compiled_profile,
)
from shadowspill.pytorch.profiling.environment import DEVICE_POOL_PROVIDER_ID
from shadowspill.pytorch.profiling.profiler import SavedValuePool, TaskProfiler
from shadowspill.runtime.bootstrap import (
    InstalledRuntime,
    validate_dynamic_execution_reservation,
)
from shadowspill.runtime.failures import format_bytes, wait_allocator_idle

from ...graph_pairs import (
    resolve_partitioned_saved_values,
)
from ..artifacts import (
    TrainingCaptureArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
)
from ..stores import PlanningStores


@dataclass(frozen=True, slots=True)
class _TrainingTaskInventory:
    compile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_keys: tuple[tuple[str, str | None], ...]
    profile_tasks: tuple[OptimizerTaskArtifact, ...]
    profile_metadata_digests: tuple[str | None, ...]


def profile_training_tasks(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    *,
    plan_id: int,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> TrainingProfileArtifacts:
    """Compile/profile each unique graph-pair and optimizer structural contract.

    Independent of the walk the step will take: every form a stage can run
    is profiled, so one profiling serves every ordering of these inputs.
    """

    state = materialized.state
    profiler = TaskProfiler(
        captured.installed.library,
        runtime_handle=captured.installed.runtime_handle,
        plan_id=plan_id,
        device_ordinal=captured.device_ordinal,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
        saved_value_pool=SavedValuePool(
            state.runtime,
            next(
                name
                for name, pool in state.runtime.pools.items()
                if pool.pool_id == state.bridge.spill_pool_id
            ),
            state.bridge.plan_handle,
        ),
    )
    try:
        return _profile_training_tasks(
            captured,
            materialized,
            profiler,
            stores=stores,
            timer=timer,
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
        )
    except BaseException:
        profiler.release_host_memory()
        raise


def _profile_training_tasks(
    captured: TrainingCaptureArtifacts,
    materialized: TrainingMaterializationArtifacts,
    profiler: TaskProfiler,
    *,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> TrainingProfileArtifacts:
    with timer.measure("saved_value_resolution"):
        partitioned = resolve_partitioned_saved_values(
            captured.partitioned,
            profiler.resolve_graph_pair_saved_values,
            tuple(workload.digest for workload in captured.workloads),
        )
    timer.progress(
        f"saved values: {format_bytes(profiler.saved_value_bytes_in_pool)} "
        "in the spill pool"
    )
    resolved_capture = replace(captured, partitioned=partitioned)
    inventory = _training_task_inventory(
        resolved_capture,
        materialized.optimizer_capture,
    )
    _report_training_profile_inventory(
        inventory,
        materialized.optimizer_capture,
        timer,
    )
    environment = profile_environment(
        device_ordinal=captured.device_ordinal,
        provider_id=DEVICE_POOL_PROVIDER_ID,
        export_bypass_key=stores.store.export_bypass_key,
    )
    manifests = _resolve_training_manifests(
        inventory,
        profiler,
        environment,
        stores,
        timer,
    )
    profiles = _profile_training_inventory(
        inventory,
        profiler,
        environment,
        manifests,
        stores,
        timer,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )
    return TrainingProfileArtifacts(
        partitioned,
        inventory.compile_tasks,
        inventory.profile_keys,
        inventory.profile_tasks,
        inventory.profile_metadata_digests,
        profiler,
        manifests,
        profiles,
    )


def _training_task_inventory(
    captured: TrainingCaptureArtifacts,
    optimizer_capture: OptimizerCapture,
) -> _TrainingTaskInventory:
    """Every structurally unique task a step of this size can run.

    Which microbatch creates a stage's gradient and which add into it is
    the ordering's decision, made when a program is lowered; every form the
    step can ask for is inventoried here, so profiling depends on the model,
    the inputs and the optimizer alone and no ordering profiles what another
    already did. The store keys compiled tasks and profiles by structural
    contract, so a form seen before, by any ordering or any run, is a lookup.
    """

    compile_by_digest: dict[str, OptimizerTaskArtifact] = {}
    profile_by_key: dict[tuple[str, str | None], OptimizerTaskArtifact] = {}
    for position, partitioned in enumerate(captured.partitioned):
        metadata_digest = captured.workloads[position].digest
        for stage in partitioned.stages:
            for option in stage.graph_pairs.variants:
                for artifact in (option.pair.forward, option.pair.backward):
                    compile_by_digest.setdefault(
                        artifact.compatibility_digest,
                        artifact,
                    )
                    profile_by_key.setdefault(
                        (
                            artifact.compatibility_digest,
                            metadata_digest,
                        ),
                        artifact,
                    )
    for task in optimizer_capture.recurrent_tasks:
        compile_by_digest.setdefault(
            task.artifact.compatibility_digest,
            task.artifact,
        )
        profile_by_key.setdefault(
            (
                task.artifact.compatibility_digest,
                None,
            ),
            task.artifact,
        )
    if optimizer_capture.initial is not None:
        compile_by_digest.setdefault(
            optimizer_capture.initial.compatibility_digest,
            optimizer_capture.initial,
        )
        profile_by_key.setdefault(
            (
                optimizer_capture.initial.compatibility_digest,
                None,
            ),
            optimizer_capture.initial,
        )
    keys = tuple(profile_by_key)
    return _TrainingTaskInventory(
        tuple(compile_by_digest.values()),
        keys,
        tuple(profile_by_key.values()),
        tuple(key[1] for key in keys),
    )


def _report_training_profile_inventory(
    inventory: _TrainingTaskInventory,
    optimizer: OptimizerCapture,
    timer: PlanningTimer,
) -> None:
    optimizer_count = sum(
        not isinstance(item, GraphArtifact) or item.kind == "optimizer"
        for item in inventory.compile_tasks
    )
    timer.progress(
        "structural artifact inventory: "
        f"graph={len(inventory.compile_tasks) - optimizer_count}, "
        f"optimizer={optimizer_count}, "
        f"unique={len(inventory.compile_tasks)}, "
        f"profile_variants={len(inventory.profile_tasks)}, "
        "optimizer_tasks="
        f"{len(optimizer.recurrent_tasks)}"
    )


def _resolve_training_manifests(
    inventory: _TrainingTaskInventory,
    profiler: TaskProfiler,
    environment: ProfileEnvironment,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> ResolvedTaskManifests:
    with timer.measure("compiler_manifest"):
        manifests = resolve_task_manifests(
            inventory.compile_tasks,
            environment=environment,
            profile_cache=stores.profiles,
            compiler=profiler.executables,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled manifest {index}/{total} {state}: {digest[:12]}"
            ),
        )
        timer.progress(
            "compiled manifest cache: "
            f"hits={manifests.cache_hits}, misses={manifests.cache_misses}"
        )
    return manifests


def _profile_training_inventory(
    inventory: _TrainingTaskInventory,
    profiler: TaskProfiler,
    environment: ProfileEnvironment,
    manifests: ResolvedTaskManifests,
    stores: PlanningStores,
    timer: PlanningTimer,
    *,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
) -> ProfilingResult:
    with timer.measure("structural_profiling"):
        return profile_unique_artifacts(
            inventory.profile_tasks,
            environment=environment,
            measure=profiler.measure,
            cache=stores.profiles,
            validate=lambda artifact, measurement: validate_compiled_profile(
                artifact,
                measurement,
                manifests.manifests,
            ),
            progress=lambda index, total, state, digest: timer.progress(
                f"structural profile {index}/{total} {state}: {digest[:12]}"
            ),
            profiling_metadata_digests=inventory.profile_metadata_digests,
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
        )


def release_build_executables(
    profiled: TrainingProfileArtifacts, installed: InstalledRuntime
) -> None:
    """Release the profiler's compiled callables and what it kept on the
    host, and prove the pool is back where planning left it."""

    profiled.profiler.executables.discard()
    profiled.profiler.release_host_memory()
    message = wait_allocator_idle(
        installed.library, installed.runtime_handle, problem="compiled task release"
    )
    if message is not None:
        raise PlanningError(message)
    validate_dynamic_execution_reservation(
        installed,
        reserved_bytes=(
            installed.fixed_execution_bytes + profiled.profiles.fixed_slab_bytes
        ),
    )
