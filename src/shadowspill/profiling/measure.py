"""Measure saved task implementations without build, planning or runtime pools."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.program import Program
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.toolchain import build_setup

from .device_memory import external_device_bytes
from .inputs import prepare_inputs
from .jobs import stage_jobs
from .task import measure_task


class _Session:
    def __init__(
        self,
        program: Program,
        device: torch.device,
        inputs: Mapping[str, Sequence[Any]],
        warmup: float,
        measurement: float,
    ):
        self.program, self.device, self.inputs = program, device, inputs
        self.warmup, self.measurement = warmup, measurement
        self.functions: dict[str, Any] = {}
        self.profiles = Profiles(setup=plain(program.setup))
        self.persistent = 0
        self.external = external_device_bytes(device)

    def function(self, key: str) -> Any:
        if key not in self.functions:
            before = torch.cuda.memory_allocated(self.device)
            self.functions[key] = restore_executable(
                self.program.executables[key],
                dependencies=self.program.bindings.get("operator_dependencies", {}).get(
                    key, ()
                ),
            )
            self.persistent += max(0, torch.cuda.memory_allocated(self.device) - before)
        return self.functions[key]

    def measure(self, key: str, arguments: Sequence[Any]) -> None:
        if key in self.profiles.tasks:
            return
        result = measure_task(
            self.function(key),
            arguments,
            self.program.bindings["manifests"][key],
            device=self.device,
            warmup_seconds=self.warmup,
            measure_seconds=self.measurement,
        )
        self.profiles.tasks[key] = replace(result, compatibility_digest=key)
        self.persistent += sum(result.metadata["warmup_persistent_bytes"])
        external = external_device_bytes(self.device)
        if external is not None:
            self.external = max(external, self.external or 0)

    def pair(self, job: Mapping[str, Any]) -> None:
        forward, backward = job["forward"], job.get("backward")
        key = forward["profile_key"]
        if key in self.profiles.tasks and (
            backward is None or backward["profile_key"] in self.profiles.tasks
        ):
            return
        arguments = prepare_inputs(
            self.program, key, self.device, supplied=self.inputs.get(key)
        )
        self.measure(key, arguments)
        if backward is None or backward["profile_key"] in self.profiles.tasks:
            return
        # Saved tensors come from this exact forward and retain their true aliases.
        leaves, _ = tree_flatten(self.function(key)(*arguments))
        values = {
            slot["object_id"]: arguments[slot["leaf_index"]]
            for slot in forward["inputs"]
        }
        values.update(
            {
                slot["object_id"]: leaves[slot["leaf_index"]]
                for slot in forward["outputs"]
            }
        )
        backward_key = backward["profile_key"]
        backward_arguments = prepare_inputs(
            self.program,
            backward_key,
            self.device,
            supplied=self.inputs.get(backward_key),
            slots=backward["inputs"],
            values=values,
        )
        self.measure(backward_key, backward_arguments)


def measure_program(
    program: Program,
    *,
    device: str | torch.device = "cuda",
    stages: Sequence[int] | None = None,
    inputs: Mapping[str, Sequence[Any]] | None = None,
    warmup_seconds: float = 0.05,
    measure_seconds: float = 0.18,
) -> Profiles:
    """Profile all unique forms, or the explicitly selected stage indices.

    A stage subset excludes standalone optimizer tasks. `inputs` overrides flat
    task arguments by profile key. Automatic inputs preserve geometry/aliases;
    unknown control semantics require explicit values. Only one pair's tensors
    are retained at a time. No model weights, pools or ArtifactStore are needed.
    """
    if program.bindings.get("provider") != "torch-inductor-aot":
        raise ValueError("Program has no supported finalized task implementation")
    target = resolve_device(device)
    if plain(program.setup) != build_setup(target):
        raise ValueError(
            "measurement device/toolchain differs from the finalized Program"
        )
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError(
            "measure_program requires the ordinary PyTorch native allocator"
        )
    jobs = list(stage_jobs(program, stages))
    with torch.cuda.device(target), torch.no_grad():
        session = _Session(
            program, target, inputs or {}, warmup_seconds, measure_seconds
        )
        for job in jobs:
            if "optimizer" in job:
                key = job["optimizer"]["profile_key"]
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
            session.profiles.fixed_device_bytes = session.external + session.persistent
        session.profiles.metadata = {
            "memory_kind": "measured",
            "allocator": "native",
            "fixed_device_source": "NVML minus native cache, plus retained data",
            "external_device_bytes": session.external,
            "retained_device_bytes": session.persistent,
            "input_policy": "automatic values; geometry preserved; caller overrides",
            "execution_qualification": "pending physical admission",
        }
        return session.profiles
