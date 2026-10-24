"""Observe collective allocations separately from collective event timing."""

from typing import Any

import torch

from shadowspill.profiling.observation.observe import observe_task
from shadowspill.task.allocations import TaskAllocationContract

from .binding.groups import GroupBindings
from .probe import CollectiveProbe
from .task import PackedCollective


def measure_memory(
    probe: CollectiveProbe | None, device: torch.device, bindings: GroupBindings
) -> dict[str, Any] | None:
    if device.type != "cuda":
        return {"memory_kind": "unresolved", "memory_error": "GPU evidence required"}
    observations = []
    for _ in range(4):
        failure = None
        try:
            if probe is not None:
                arguments: tuple[torch.Tensor, ...] = (probe.input,)
                if probe.request.operation == "all_gather" and probe.request.inplace:
                    arguments += (probe.output,)
                with torch.cuda.stream(probe.stream):
                    observations.append(
                        observe_task(
                            PackedCollective(probe.request, probe.group),
                            arguments,
                            device=device,
                        )
                    )
        except Exception as error:
            failure = str(error)
        bindings.check(failure)
    if probe is None:
        return None
    # The first allocating-output invocation warms any extra retained state.
    cold, *observations = observations
    contracts = [TaskAllocationContract.capture(item.trace) for item in observations]
    if len({item.compatibility_digest for item in contracts}) != 1 or any(
        item.persistent_bytes for item in observations
    ):
        return {
            "memory_kind": "unresolved",
            "memory_error": "warmed collective allocation lifetimes are not stable",
        }
    baseline = observations[0]
    return {
        "memory_kind": "measured",
        "workspace_bytes": baseline.workspace_bytes,
        "allocation_trace": [item.to_dict() for item in baseline.trace],
        "allocation_contract": contracts[0].to_dict(),
        "output_input_bindings": [item.to_dict() for item in baseline.input_bindings],
        "off_device_output_leaves": baseline.off_device_outputs,
        "warmup_persistent_bytes": cold.persistent_bytes,
        "memory_source": "native allocator observer; identical execution callable",
    }
