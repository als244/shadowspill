"""One profiling session: restore finalized tasks and measure paired executions."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import Profiles
from shadowspill.artifacts.program import Program
from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.diagnostics.progress import event, phase
from shadowspill.errors import ProfilingError
from shadowspill.operator_preparation.lifecycle import LoadedPreparations
from shadowspill.pytorch.control_inputs import ControlInputs
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.numerics import (
    program_numerical_contracts,
    validate_numerical_settings,
)
from shadowspill.pytorch.task_errors import task_error
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.task.profiling import ProfilingOptions

from .cache_identity import task_identities
from .device_memory import external_device_bytes
from .streams import MeasurementStreams
from .task import measure_task

if TYPE_CHECKING:
    from .cache import TaskProfileCache


class MeasurementSession:
    def __init__(
        self,
        program: Program,
        device: torch.device,
        inputs: Mapping[str, Sequence[Any]],
        options: ProfilingOptions,
        *,
        groups: GroupBindings | None = None,
        preparations: LoadedPreparations | None = None,
        cache: TaskProfileCache | None = None,
    ):
        validate_numerical_settings(
            program_numerical_contracts(program),
            check=None if groups is None else groups.check,
        )
        self.program, self.device, self.inputs = program, device, inputs
        self.groups = groups
        self.preparations = preparations
        self.options = options
        if groups is not None and any(
            value != options.to_dict() for value in groups.gather(options.to_dict())
        ):
            raise ValueError("profiling options must agree across workers")
        self.functions: dict[str, Any] = {}
        self.controls = ControlInputs(program, device, inputs)
        self.profiles = Profiles(setup=plain(program.setup))
        self.streams = MeasurementStreams(program, device)
        self.persistent = self.streams.retained_bytes
        self.external = external_device_bytes(device)
        self.cache = cache
        self.cache_allowed = not inputs
        self.cache_identities: dict[str, dict[str, Any]] = {}
        self.cache_hits: list[str] = []
        self.cache_misses: list[str] = []

    def reuse(self, jobs: Sequence[Mapping[str, Any]]) -> None:
        """All participants agree on hits before any task enters a collective."""
        enabled = self.cache is not None and self.cache_allowed
        votes = [enabled] if self.groups is None else self.groups.gather(enabled)
        if not all(votes):
            error = (
                "required task cache is unavailable or explicit inputs disable reuse"
                if self.cache is not None and self.cache.policy.require_hit
                else None
            )
            if self.groups is not None:
                self.groups.check(error)
            if error is not None:
                raise ValueError(error)
            self.cache = None
            return
        if self.cache is None:
            return
        self.cache_identities = task_identities(
            self.program,
            jobs,
            options=self.options,
            environment={
                "allocator_config": os.environ.get(
                    "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                ),
                "streams": self.streams.description,
                "rank_assignment": None
                if self.groups is None
                else self.groups.assignment.logical_by_worker,
                "logical_rank": None
                if self.groups is None
                else self.groups.logical_rank,
                "topology": None if self.groups is None else self.groups.topology,
            },
        )
        hits, error = {}, None
        try:
            for key, identity in self.cache_identities.items():
                if (
                    record := self.cache.get(
                        identity,
                        self.program.bindings["manifests"][key],
                        minimum_samples=self.options.minimum_samples,
                    )
                ) is not None:
                    hits[key] = record
        except Exception as failure:
            error = str(failure)
        if self.groups is not None:
            self.groups.check(error)
        if error is not None:
            raise ValueError(error)
        accepted = set(hits)
        if self.groups is not None:
            accepted.intersection_update(*map(set, self.groups.gather(sorted(hits))))
        self.cache_misses = sorted(self.cache_identities.keys() - accepted)
        # Restore every reused function so its retained data is charged in this
        # candidate, just as when measuring. Do not reuse aggregate fixed bytes.
        try:
            for key in self.cache_misses:
                self.cache.policy.refuse_miss("task measurement", key)
            for key in sorted(accepted):
                self.function(key)
                record = hits[key]
                self.profiles.tasks[key] = record.profile
                self.external = max(self.external or 0, record.external_device_bytes)
                self.cache_hits.append(key)
        except Exception as failure:
            error = str(failure)
        if self.groups is not None:
            self.groups.check(error)
        if error is not None:
            raise ValueError(error)

    def function(self, key: str) -> Any:
        contracts = program_numerical_contracts(self.program, (key,))
        validate_numerical_settings(contracts)
        if key not in self.functions:
            before = torch.cuda.memory_allocated(self.device)
            self.functions[key] = restore_executable(
                self.program.executables[key],
                dependencies=self.program.bindings.get("operator_dependencies", {}).get(
                    key, ()
                ),
                triton_choices=self.program.bindings.get("triton_choices", {}).get(key),
                numerical_contracts=contracts,
            )
            if self.preparations is not None:
                self.functions[key] = self.preparations.wrap(
                    self.program, key, self.functions[key]
                )
            self.persistent += max(0, torch.cuda.memory_allocated(self.device) - before)
        return self.functions[key]

    def measure(self, key: str, arguments: Sequence[Any]) -> None:
        # Preparing controls can run producer tasks and leave large, unused
        # allocator blocks on other streams. Reclaim those before observation
        # or a cached forward's saved-tensor replay, never between timed samples.
        torch.cuda.empty_cache()
        if key in self.profiles.tasks:
            return
        definition = next(
            item for item in self.program.task_definitions if item.profile_key == key
        )
        try:
            with (
                phase(f"Measure {definition.name or 'task'} · {key[:12]}"),
                torch.cuda.stream(self.streams[key]),
            ):
                result = measure_task(
                    self.function(key),
                    arguments,
                    self.program.bindings["manifests"][key],
                    device=self.device,
                    options=self.options,
                    continue_while=None
                    if self.groups is None
                    else self.groups.continue_sampling,
                    check=None if self.groups is None else self.groups.check,
                )
                duration = (
                    "Unavailable"
                    if result.duration_seconds is None
                    else f"{result.duration_seconds * 1000:.3f} ms"
                )
                workspace = (
                    "Unavailable"
                    if result.workspace_bytes is None
                    else f"{result.workspace_bytes / (1 << 20):,.2f} MiB"
                )
                event(f"GPU duration: {duration}; workspace: {workspace}")
        except Exception as error:
            raise task_error(
                ProfilingError, self.program, key, "measurement", error
            ) from error
        self.profiles.tasks[key] = replace(
            result, compatibility_digest=definition.compatibility_digest
        )
        self.persistent += sum(result.metadata["warmup_persistent_bytes"])
        external = external_device_bytes(self.device)
        if external is not None:
            self.external = max(external, self.external or 0)
        if self.cache is not None:
            cache_error = None
            try:
                self.cache.put(
                    self.cache_identities[key],
                    self.program.bindings["manifests"][key],
                    self.profiles.tasks[key],
                    self.external,
                )
            except Exception as failure:
                cache_error = str(failure)
            if self.groups is not None:
                self.groups.check(cache_error)
            if cache_error is not None:
                raise ValueError(cache_error)

    def pair(self, job: Mapping[str, Any]) -> None:
        forward, backward = job["forward"], job.get("backward")
        key = forward["profile_key"]
        if key in self.profiles.tasks and (
            backward is None or backward["profile_key"] in self.profiles.tasks
        ):
            return
        torch.cuda.empty_cache()
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
        # Keep the real saved tensors alive, but release the forward's cached
        # scratch before allocating backward-only inputs on the default stream.
        torch.cuda.empty_cache()
        backward_arguments = prepare_inputs(
            self.program,
            backward_key,
            self.device,
            supplied=self.inputs.get(backward_key),
            slots=backward["inputs"],
            values=values,
        )
        del arguments, leaves, values
        self.measure(backward_key, backward_arguments)
