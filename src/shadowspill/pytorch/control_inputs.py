"""Generate task controls with their compiled producers, outside measurement."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.errors import CaptureError

from .task_inputs import prepare_inputs


class ControlInputs:
    """Replay only producers needed for missing integer/boolean task inputs.

    Ordinary parameters/activations use the shared profiling-input policy. Control
    tensors come from real operator execution, never invented indices. No model
    payload or generated control array is embedded in the saved Program. Each
    request retains only its dependency slice, with CPU snapshots between tasks.
    """

    def __init__(
        self,
        program: Program,
        device: torch.device,
        inputs: Mapping[str, Sequence[Any]] | None = None,
    ):
        self.program, self.device = program, device
        self.inputs = inputs or {}
        self.producers: dict[str, Mapping[str, Any]] = {}
        for job in stage_jobs(program, None):
            if "forward" in job:
                for slot in job["forward"]["outputs"]:
                    self.producers.setdefault(slot["object_id"], job["forward"])

    def prepare(
        self,
        component: Mapping[str, Any],
        invoke: Callable[[str, Sequence[Any]], Any],
        *,
        supplied: Sequence[Any] | None = None,
        values: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        key = component["profile_key"]
        if supplied is None:
            supplied = self.inputs.get(key)
        if supplied is not None:
            return prepare_inputs(self.program, key, self.device, supplied=supplied)
        resolved = dict(values or {})
        description = self.program.bindings["task_inputs"][key]
        provenance = description.get("provenance", ())
        for slot in component["inputs"]:
            position, object_id = slot["leaf_index"], slot["object_id"]
            source = provenance[position] if provenance else {}
            if (
                object_id in resolved
                or source.get("role") != "control"
                or "value" in source
            ):
                continue
            producer = self.producers.get(object_id)
            if producer is None:
                continue  # The ordinary input helper reports missing caller data.
            arguments = self.prepare(producer, invoke)
            leaves, _ = tree_flatten(invoke(producer["profile_key"], arguments))
            leaf = next(
                item["leaf_index"]
                for item in producer["outputs"]
                if item["object_id"] == object_id
            )
            value = leaves[leaf]
            if not isinstance(value, torch.Tensor):
                raise CaptureError(
                    f"control producer returned no tensor for {object_id}"
                )
            shape, stride, dtype, target = (
                value.shape,
                value.stride(),
                value.dtype,
                value.device,
            )
            snapshot = torch.empty_strided(shape, stride, dtype=dtype, device="cpu")
            snapshot.copy_(value.detach())
            del arguments, leaves, value
            if target.type == "cuda":
                restored = torch.empty_strided(
                    shape, stride, dtype=dtype, device=target
                )
                restored.copy_(snapshot)
                resolved[object_id] = restored
            else:
                resolved[object_id] = snapshot
        return prepare_inputs(
            self.program, key, self.device, slots=component["inputs"], values=resolved
        )
