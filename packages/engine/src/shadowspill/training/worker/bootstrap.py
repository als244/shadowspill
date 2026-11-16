"""Installed persistent worker entry; no model factory or path arrives in a call."""

import argparse
import socket
from functools import partial
from pathlib import Path
from typing import BinaryIO, cast

from shadowspill.orchestration.capture.progress import write_progress
from shadowspill.orchestration.launch.file_grants import resolve_file, verify_file
from shadowspill.training.definitions import InitializationRuntime, load_registry
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill_client.protocol.codec import ProtocolError

from .channel import read_message
from .output_files import FileOutputs
from .protocol import WorkerLaunch
from .runtime import PreparedWorker
from .trace_files import TraceFiles


def serve(
    stream: BinaryIO,
    *,
    files: Path,
    outputs: Path,
    cache: Path,
    nsys_progress: Path | None = None,
) -> None:
    """CPU preflight the granted bundle before the existing all-worker barrier."""
    launch = read_message(stream)
    if not isinstance(launch, WorkerLaunch):
        raise ProtocolError("worker requires private launch configuration first")
    registry = load_registry(launch.packages)
    bundle = TrainingBundle.load(verify_file(files, launch.startup.bundle))
    prepared = PreparedWorker.resolve(
        bundle.recheck(registry),
        launch.startup.scope,
        bundle=launch.startup.bundle,
        rank_map=dict(enumerate(launch.startup.logical_by_worker)),
        restore=launch.restore,
        weights=launch.weights,
    )
    if prepared.startup != launch.startup:
        raise ProtocolError("worker preflight differs from the admitted startup")
    weights = {ref.id: verify_file(files, ref) for ref in launch.weights}
    initial = (
        None
        if prepared.initialization is None
        else prepared.initialization.context.weights
    )
    if set(weights) != (
        {initial.reference}
        if initial is not None and initial.kind == "artifact"
        else set()
    ):
        raise ValueError("worker weight grants differ from its initialization")
    if launch.restore is not None:
        verify_file(files, launch.restore.rank.archive)

    def artifact(identity: str) -> Path:
        if identity not in weights:
            raise ValueError("initializer requested an ungranted artifact")
        return weights[identity]

    prepared.serve(
        stream,
        stream,
        InitializationRuntime(
            cache,
            artifact,
            0
            if prepared.initialization is None
            else prepared.initialization.description.disk_bytes,
        ),
        lambda reference: resolve_file(files, reference),
        FileOutputs(prepared.outputs, root=outputs),
        traces=TraceFiles(
            prepared.traces, bundle=prepared.startup.bundle, root=outputs
        ),
        capture_progress=None
        if nsys_progress is None
        else partial(write_progress, nsys_progress),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", required=True, type=int)
    parser.add_argument("--files", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--nsys-progress", type=Path)
    args = parser.parse_args()
    if args.control_fd < 3:
        raise ValueError("worker control requires an inherited private descriptor")
    with socket.socket(fileno=args.control_fd) as control:
        if control.family != socket.AF_UNIX or control.type != socket.SOCK_STREAM:
            raise ValueError("worker requires an inherited private stream socket")
        with control.makefile("rwb", buffering=0) as stream:
            serve(
                cast(BinaryIO, stream),
                files=args.files,
                outputs=args.outputs,
                cache=args.cache,
                nsys_progress=args.nsys_progress,
            )


if __name__ == "__main__":
    main()
