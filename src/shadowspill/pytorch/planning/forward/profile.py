"""Every structurally unique forward task compiled and measured."""

from shadowspill.errors import (
    PlanningError,
)
from shadowspill.pytorch.profiling import (
    profile_environment,
    profile_unique_artifacts,
    resolve_task_manifests,
    validate_compiled_profile,
)
from shadowspill.pytorch.profiling.profiler import TaskProfiler
from shadowspill.pytorch.runtime_adapter.allocator import (
    validate_dynamic_execution_reservation,
)
from shadowspill.pytorch.runtime_adapter.failures import wait_allocator_idle

from ..artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
)
from ..common import (
    PlanningTimer,
)
from ..stores import PlanningStores
from .programs import _verify_manifest_identity


def profile_forward_tasks(
    captured: ForwardCaptureArtifacts,
    *,
    plan_id: int,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> ForwardProfileArtifacts:
    """Compile and profile every unique structural task contract exactly once."""

    profiler = TaskProfiler(
        captured.installed.library,
        runtime_handle=captured.installed.runtime_handle,
        plan_id=plan_id,
        device_ordinal=captured.device_ordinal,
        allocation_probe_seeds=allocation_probe_seeds,
        allocation_probe_repetitions=allocation_probe_repetitions,
    )
    environment = profile_environment(
        device_ordinal=captured.device_ordinal,
        provider_id="shadowspill.device_pool",
        export_bypass_key=stores.store.export_bypass_key,
    )
    with timer.measure("compiler_manifest"):
        manifests = resolve_task_manifests(
            captured.tasks,
            environment=environment,
            profile_cache=stores.profiles,
            compiler=profiler.executables,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled manifest {index}/{total} {state}: {digest[:12]}"
            ),
        )
    with timer.measure("structural_profiling"):
        profiles = profile_unique_artifacts(
            captured.tasks,
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
            profiling_metadata_digests=(captured.workload.digest,)
            * len(captured.tasks),
            allocation_probe_seeds=allocation_probe_seeds,
            allocation_probe_repetitions=allocation_probe_repetitions,
        )
    with timer.measure("compilation"):
        compiled_tasks = profiler.take_compiled_tasks(
            captured.tasks,
            progress=lambda index, total, state, digest: timer.progress(
                f"compiled entrypoint {index}/{total} {state}: {digest[:12]}"
            ),
        )
        _verify_manifest_identity(manifests, compiled_tasks)
        message = wait_allocator_idle(
            captured.installed.library,
            captured.installed.runtime_handle,
            problem="compiled entrypoint release",
        )
        if message is not None:
            raise PlanningError(message)
        validate_dynamic_execution_reservation(
            captured.installed,
            reserved_bytes=(
                captured.installed.fixed_execution_bytes + profiles.fixed_slab_bytes
            ),
        )
    timer.attribute_compilation_and_profiling(profiler.wall_times)
    return ForwardProfileArtifacts(profiler, manifests, profiles, compiled_tasks)
