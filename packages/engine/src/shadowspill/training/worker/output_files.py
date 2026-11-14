"""Bind immutable worker invocations to the agent's pre-reserved output files."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from shadowspill.orchestration.launch.publication import (
    OutputSpec,
    require_output,
    write_output,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MEDIA_TYPE
from shadowspill_client.protocol.requests import ArtifactRef

from .bindings import BoundWorkerCall
from .output_frame import WorkerOutputFrame
from .protocol import ticket


class OutputSink(Protocol):
    """Preflight before release, then terminal transfer using the worker's budget."""

    def check(self, call: BoundWorkerCall) -> None: ...

    def collect(
        self, call: BoundWorkerCall, value: Any, budget: BufferBudget
    ) -> ArtifactRef | None: ...


class FileOutputs:
    def __init__(
        self, frames: Mapping[tuple[str, str], WorkerOutputFrame], *, root: Path
    ) -> None:
        self.frames, self.root = frames, root

    def spec(self, call: BoundWorkerCall) -> OutputSpec | None:
        if call.plan is None:
            return None
        frame = self.frames[call.command.candidate, call.plan]
        return OutputSpec(
            "result." + ticket(call.command).call_sha256,
            frame.frame.size.wire,
            MEDIA_TYPE,
        )

    def check(self, call: BoundWorkerCall) -> None:
        spec = self.spec(call)
        if spec is not None:
            require_output(self.root, spec)

    def collect(
        self, call: BoundWorkerCall, value: Any, budget: BufferBudget
    ) -> ArtifactRef | None:
        spec = self.spec(call)
        if spec is None:
            return None
        assert call.plan is not None
        frame = self.frames[call.command.candidate, call.plan]
        return write_output(
            self.root, spec, lambda stream: frame.write(value, stream, budget, spec.id)
        )
