"""Measure saved task implementations without build, planning or runtime pools."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from shadowspill.artifacts._values import nonnegative_int, plain
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.pytorch.toolchain import build_setup

from .session import MeasurementSession


def measure_program(
    program: Program,
    *,
    device: str | torch.device = "cuda",
    stages: Sequence[int] | None = None,
    inputs: Mapping[str, Sequence[Any]] | None = None,
    warmup_seconds: float = 0.05,
    measure_seconds: float = 0.18,
    external_headroom_bytes: int = 512 << 20,
) -> Profiles:
    """Profile all unique forms, or the explicitly selected stage indices.

    A stage subset excludes standalone optimizer tasks. `inputs` overrides flat
    task arguments by profile key. Automatic inputs preserve geometry/aliases;
    unknown control semantics require explicit values. Only one pair's tensors
    are retained at a time. No model weights, pools or ArtifactStore are needed.
    Fixed device overhead includes the requested worker headroom in addition to
    observed library/context memory. Executor inherits this saved reservation.
    """
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
    with (
        torch.cuda.device(target),
        torch.no_grad(),
        torch.random.fork_rng(devices=[target]),
    ):
        session = MeasurementSession(
            program, target, inputs or {}, warmup_seconds, measure_seconds
        )
        for job in jobs:
            if "optimizer" in job or "gradient" in job:
                component = job["optimizer"] if "optimizer" in job else job["gradient"]
                key = component["profile_key"]
                if key not in session.profiles.tasks:
                    arguments = prepare_inputs(
                        program, key, target, supplied=session.inputs.get(key)
                    )
                    session.measure(key, arguments)
                    del arguments
            else:
                session.pair(job)
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
        session.profiles.metadata = {
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
            "input_policy": "automatic values; geometry preserved; caller overrides",
            "execution_qualification": "pending physical admission",
        }
        return session.profiles
