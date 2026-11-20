"""Measure saved task implementations without build, planning or runtime pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.artifacts._values import nonnegative_int, plain
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.diagnostics.progress import event, phase
from shadowspill.operator_preparation.lifecycle import LoadedPreparations
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.resources.preparation import prepare_resources
from shadowspill.task.profiling import ProfilingOptions, resolve_options

from .session import MeasurementSession

if TYPE_CHECKING:
    from .cache import TaskProfileCache


def measure_program(
    program: Program,
    *,
    device: str | torch.device = "cuda",
    stages: Sequence[int] | None = None,
    rank_map: Mapping[int, int] | None = None,
    inputs: Mapping[str, Sequence[Any]] | None = None,
    options: ProfilingOptions | None = None,
    warmup_seconds: float | None = None,
    measure_seconds: float | None = None,
    external_headroom_bytes: int = 512 << 20,
    cache: TaskProfileCache | None = None,
) -> Profiles:
    """Measure the time and memory of each task form of a Program.

    It runs the Program's finalized tasks on the GPU they were compiled for,
    with generated inputs of the right geometry and aliasing, under
    PyTorch's native allocator; it needs no model weights and opens no
    pools. Stages sharing a form share its measurement. Fixed device memory,
    the requested headroom included, is measured on every call, even when a
    `TaskProfileCache` supplies the task measurements. Pass the result to
    `apply_profiles`.

    Parameters
    ----------
    program : Program
        A built Program, with its finalized code.
    device : str or torch.device, default "cuda"
        The GPU to measure on, with the process-local ordinal and toolchain
        the Program was built for.
    stages : sequence of int, optional
        Indices of the stages to measure; a subset leaves out standalone
        optimizer tasks. By default every task form.
    rank_map : mapping of int to int, optional
        Launcher rank to logical rank, when the Program's resources need
        every worker.
    inputs : mapping of str to sequence, optional
        Flat task arguments by profile key, instead of generated ones, for
        tasks whose control inputs cannot be generated. Disables cache
        reuse.
    options : ProfilingOptions, optional
        Timing bounds. By default ``ProfilingOptions()``.
    warmup_seconds : float, optional
        Shortcut for ``conditioning_seconds`` of ``options``; not with
        ``options``.
    measure_seconds : float, optional
        Shortcut for ``measurement_seconds`` of ``options``; not with
        ``options``.
    external_headroom_bytes : int, default 512 << 20
        Device memory each worker reserves for what libraries allocate
        lazily, in bytes. It is counted in the fixed device memory and saved;
        `Executor` reserves the same.
    cache : TaskProfileCache, optional
        Task measurements shared across Programs and processes.

    Returns
    -------
    Profiles
        A profile per measured task form, the fixed device memory, and the
        session's details in ``metadata``.

    Raises
    ------
    ValueError
        If the Program has no finalized code or was built for another GPU or
        toolchain, ``external_headroom_bytes`` is not a nonnegative integer,
        or ``options`` comes with a shortcut.
    RuntimeError
        If PyTorch's native allocator is not in use.
    ProfilingError
        If a task cannot be measured.
    """
    started = perf_counter_ns()
    options = resolve_options(options, warmup_seconds, measure_seconds)
    nonnegative_int(external_headroom_bytes, "external_headroom_bytes")
    if program.bindings.get("provider") != "torch-inductor-aot":
        raise ValueError("Program has no supported finalized task implementation")
    target = resolve_device(device)
    assert target.index is not None
    require_compiled_device(program, target.index)
    if plain(program.setup) != build_setup(target):
        raise ValueError(
            "measurement device/toolchain differs from the finalized Program"
        )
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError(
            "measure_program requires the ordinary PyTorch native allocator"
        )
    jobs = list(stage_jobs(program, stages))
    event(
        f"Profiling: {len(jobs)} task groups; "
        f"conditioning {options.conditioning_seconds:g}s; "
        f"measurement {options.measurement_seconds:g}s; "
        f"samples {options.minimum_samples}-{options.maximum_samples}"
    )
    with (
        torch.cuda.device(target),
        torch.no_grad(),
        torch.random.fork_rng(devices=[target]),
        ExitStack() as cleanup,
    ):
        preparations = LoadedPreparations(target)
        cleanup.callback(preparations.close)
        # Control-input reconstruction can invoke tasks outside a stage subset.
        # Load all binaries outside allocation observation, and drain streams
        # before unloading even if a task's measurement raises.
        cleanup.callback(torch.cuda.synchronize, target)
        before = torch.cuda.memory_allocated(target)
        with phase("Load operator preparations"):
            preparations.add(program, tuple(program.executables))
        prepared_bytes = max(0, torch.cuda.memory_allocated(target) - before)
        with phase("Prepare device and communication resources"):
            resources = cleanup.enter_context(
                prepare_resources(program, target, rank_map)
            )
        # Resource creation can initialize provider runtimes too. Binaries must
        # precede it, while the final drain precedes both kinds of cleanup.
        cleanup.callback(torch.cuda.synchronize, target)
        session = MeasurementSession(
            program,
            target,
            inputs or {},
            options,
            groups=resources.groups,
            preparations=preparations,
            cache=cache,
        )
        session.persistent += resources.retained_bytes + prepared_bytes
        cache_started = perf_counter_ns()
        with phase("Read task profile cache"):
            session.reuse(jobs)
            event(
                f"Cache: {len(session.cache_hits)} reused; "
                f"{len(session.cache_misses)} missing"
                if session.cache is not None
                else "Cache: Disabled"
            )
        tasks_started = perf_counter_ns()
        cache_wall_ns = tasks_started - cache_started
        for index, job in enumerate(jobs, 1):
            role = (
                "optimizer"
                if "optimizer" in job
                else "gradient"
                if "gradient" in job
                else None
            )
            keys = (
                [job[role]["profile_key"]]
                if role is not None
                else [
                    job[name]["profile_key"]
                    for name in ("forward", "backward")
                    if job.get(name)
                ]
            )
            if all(key in session.profiles.tasks for key in keys):
                continue
            title = (
                role.capitalize()
                if role is not None
                else f"Stage {job['stage']} · Forward/backward"
                if job.get("backward")
                else f"Stage {job['stage']} · Forward"
            )
            with phase(f"Task group {index}/{len(jobs)} · {title}"):
                if role is not None:
                    key = keys[0]
                    torch.cuda.empty_cache()
                    arguments = prepare_inputs(
                        program, key, target, supplied=session.inputs.get(key)
                    )
                    session.measure(key, arguments)
                    del arguments
                else:
                    session.pair(job)
        tasks_finished = perf_counter_ns()
        for definition in program.task_definitions:
            if definition.profile_key == "invocation_start":
                session.profiles.tasks[definition.profile_key] = TaskProfile(
                    0.0,
                    0,
                    definition.compatibility_digest,
                    metadata={"duration_kind": "static", "memory_kind": "static"},
                )
        if session.external is not None:
            session.profiles.fixed_device_bytes = (
                session.external + session.persistent + external_headroom_bytes
            )
        event(f"Measured {len(session.profiles.tasks)} task profiles")
        if session.profiles.fixed_device_bytes is not None:
            event(
                "Fixed device memory: "
                f"{session.profiles.fixed_device_bytes / (1 << 20):,.2f} MiB"
            )
        session.profiles.metadata = {
            "profiling_options": options.to_dict(),
            "profiling_phase_timings_ns": {
                "validation_and_preparation": cache_started - started,
                "cache_lookup_and_load": cache_wall_ns,
                "task_loop": tasks_finished - tasks_started,
            },
            "task_cache": {
                "enabled": session.cache is not None,
                "hits": session.cache_hits,
                "misses": session.cache_misses,
                "lookup_and_load_wall_ns": cache_wall_ns,
            },
            "memory_kind": "measured",
            "allocator": "native",
            "fixed_device_source": (
                "NVML minus native cache, plus retained data and worker headroom"
            ),
            "external_device_bytes": session.external,
            "retained_device_bytes": session.persistent,
            "external_headroom_bytes": external_headroom_bytes,
            "resource_streams": session.streams.description,
            "stream_workspace_bytes": session.streams.retained_bytes,
            "provider_resource_bytes": resources.retained_bytes,
            "prepared_operator_retained_bytes": prepared_bytes,
            "input_policy": "automatic values; geometry preserved; caller overrides",
            "profiling_scenario": plain(program.bindings.get("profiling_scenario")),
            "execution_qualification": "pending physical admission",
        }
        if resources.groups is not None:
            from .distributed import combine_profiles

            return combine_profiles(session.profiles, resources.groups)
        return session.profiles
