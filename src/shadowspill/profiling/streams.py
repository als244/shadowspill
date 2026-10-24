"""Prepare per-resource library state without installing an execution allocator."""

import torch

from shadowspill.artifacts.program import Program
from shadowspill.ir import ResourceKind, ResourceSpec
from shadowspill.pytorch.resource_streams import ResourceStreams


class MeasurementStreams:
    def __init__(self, program: Program, device: torch.device) -> None:
        fallback = ResourceSpec(program.devices[0].device_id, ResourceKind.COMPUTE)
        self.resources = {key: fallback for key in program.executables}
        for task in reversed(program.tasks):
            if task.profile_id in self.resources:
                self.resources[task.profile_id] = task.resource
        all_resources = set(self.resources.values()) | {
            task.resource
            for task in program.tasks
            if task.profile_id in program.executables
        }
        self.streams = ResourceStreams(device, all_resources, reuse_current=False)
        before = torch.cuda.memory_allocated(device)
        self.streams.initialize_blas(all_resources)
        # PyTorch's stream pool can recycle a stream with an existing workspace.
        # Include the declared capacities even when warmup allocates no new
        # bytes. The two libraries may share storage; summing is a safe bound.
        declared = len(all_resources) * (
            torch.backends.cuda.cublas_workspace_size()
            + torch.backends.cuda.cublaslt_workspace_size()
        )
        self.retained_bytes = max(
            declared, torch.cuda.memory_allocated(device) - before
        )
        self.description = [
            resource.to_dict()
            for resource in sorted(
                all_resources, key=lambda r: (r.device_id, r.kind, r.lane)
            )
        ]

    def __getitem__(self, key: str) -> torch.cuda.Stream:
        return self.streams[self.resources[key]]
