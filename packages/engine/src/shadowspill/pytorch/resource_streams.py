"""Stable CUDA resource streams shared by measurement and execution."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

from shadowspill.ir import ResourceKind, ResourceSpec

from .workspaces import initialize_blas


class ResourceStreams:
    def __init__(
        self,
        device: torch.device,
        resources: Iterable[ResourceSpec],
        *,
        reuse_current: bool = True,
    ) -> None:
        self.device = device
        # Execution can reuse the stream warmed during allocator installation.
        # Measurement uses new streams so preexisting BLAS workspaces cannot
        # disappear from its retained-memory observation.
        current = torch.cuda.current_stream(device)
        self._lanes = {
            resource: (
                current
                if reuse_current
                and resource.kind == ResourceKind.COMPUTE
                and resource.lane == 0
                else torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            )
            for resource in sorted(
                set(resources), key=lambda r: (r.device_id, r.kind, r.lane)
            )
        }

    def __getitem__(self, resource: ResourceSpec) -> torch.cuda.Stream:
        return self._lanes[resource]

    def initialize_blas(self, resources: Iterable[ResourceSpec]) -> None:
        for resource in sorted(
            set(resources), key=lambda r: (r.device_id, r.kind, r.lane)
        ):
            with torch.cuda.stream(self._lanes[resource]):
                initialize_blas(self.device)

    @contextmanager
    def invocation(self) -> Iterator[dict[ResourceSpec, torch.cuda.Stream]]:
        caller = torch.cuda.current_stream(self.device)
        for stream in self._lanes.values():
            stream.wait_stream(caller)
        try:
            yield self._lanes
        finally:
            # Result acquisition and the invocation marker run on the caller
            # stream, after every task lane. Joining also covers dispatch errors.
            for stream in self._lanes.values():
                caller.wait_stream(stream)
