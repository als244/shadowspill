"""One profiling session: restore finalized tasks and measure paired executions."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import Profiles
from shadowspill.artifacts.program import Program
from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.pytorch.control_inputs import ControlInputs
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.task_inputs import prepare_inputs

from .device_memory import external_device_bytes
from .streams import MeasurementStreams
from .task import measure_task


class MeasurementSession:
    def __init__(
        self,
        program: Program,
        device: torch.device,
        inputs: Mapping[str, Sequence[Any]],
        warmup: float,
        measurement: float,
        *,
        groups: GroupBindings | None = None,
    ):
        self.program, self.device, self.inputs = program, device, inputs
        self.groups = groups
        self.warmup, self.measurement = warmup, measurement
        self.functions: dict[str, Any] = {}
        self.controls = ControlInputs(program, device, inputs)
        self.profiles = Profiles(setup=plain(program.setup))
        self.streams = MeasurementStreams(program, device)
        self.persistent = self.streams.retained_bytes
        self.external = external_device_bytes(device)

    def function(self, key: str) -> Any:
        if key not in self.functions:
            before = torch.cuda.memory_allocated(self.device)
            self.functions[key] = restore_executable(
                self.program.executables[key],
                dependencies=self.program.bindings.get("operator_dependencies", {}).get(
                    key, ()
                ),
                triton_choices=self.program.bindings.get("triton_choices", {}).get(key),
            )
            self.persistent += max(0, torch.cuda.memory_allocated(self.device) - before)
        return self.functions[key]

    def measure(self, key: str, arguments: Sequence[Any]) -> None:
        if key in self.profiles.tasks:
            return
        with torch.cuda.stream(self.streams[key]):
            result = measure_task(
                self.function(key),
                arguments,
                self.program.bindings["manifests"][key],
                device=self.device,
                warmup_seconds=self.warmup,
                measure_seconds=self.measurement,
                continue_while=None
                if self.groups is None
                else self.groups.continue_sampling,
            )
        definition = next(
            item for item in self.program.task_definitions if item.profile_key == key
        )
        self.profiles.tasks[key] = replace(
            result, compatibility_digest=definition.compatibility_digest
        )
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
        arguments = self.controls.prepare(
            forward,
            lambda key, args: self.function(key)(*args),
            supplied=self.inputs.get(key),
        )
        self.measure(key, arguments)
        if backward is None or backward["profile_key"] in self.profiles.tasks:
            return
        # Saved tensors come from this exact forward and retain their true aliases.
        with torch.cuda.stream(self.streams[key]):
            leaves, _ = tree_flatten(self.function(key)(*arguments))
        self.streams[key].synchronize()
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
