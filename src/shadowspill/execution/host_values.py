"""CPU task values retain dependencies without acquiring GPU pool storage."""

from typing import Any

import torch

from shadowspill.artifacts.plan import Plan


class HostValues:
    def __init__(self, plan: Plan):
        self.values: dict[str, torch.Tensor] = {}
        self.constants: dict[str, torch.Tensor] = {}
        tasks = plan._costed_program.selected_tasks(plan.selections)
        self.last_read = {
            object_id: task.task_id for task in tasks for object_id in task.inputs
        }
        keys = {task.task_id: task.profile_id for task in plan.program.tasks}
        for entry in plan.program.bindings["entrypoints"]:
            if entry["task_id"] not in keys:
                continue
            description = plan.program.bindings["task_inputs"].get(
                keys[entry["task_id"]]
            )
            if description is None:
                continue
            geometries = dict(
                zip(
                    description["tensor_positions"],
                    description["tensor_geometry"],
                    strict=True,
                )
            )
            for slot in entry["input_slots"]:
                index, object_id = slot["leaf_index"], slot["object_id"]
                geometry = geometries[index]
                provenance = description.get("provenance", ())
                source = provenance[index] if provenance else {}
                if geometry["device_type"] == "cpu" and "value" in source:
                    self.constants[object_id] = self._constant(
                        geometry, source["value"]
                    )

    @staticmethod
    def _constant(geometry: Any, value: Any) -> torch.Tensor:
        dtype = getattr(torch, geometry["dtype"].removeprefix("torch."))
        result = torch.empty_strided(
            geometry["shape"], geometry["stride"], dtype=dtype, device="cpu"
        )
        result.copy_(
            torch.as_tensor(value, dtype=dtype, device="cpu").reshape(result.shape)
        )
        return result

    def get(self, object_id: str) -> torch.Tensor:
        if object_id in self.values:
            return self.values[object_id]
        if object_id in self.constants:
            return self.constants[object_id]
        raise RuntimeError(
            f"CPU task input {object_id} has not been supplied or produced"
        )

    def finish(self, task_id: str) -> None:
        self.values = {
            key: value
            for key, value in self.values.items()
            if key in self.last_read and self.last_read[key] != task_id
        }
