"""Installed persistent worker entry; no model factory or path arrives in a call."""

import argparse
import socket
from functools import partial
from pathlib import Path
from typing import BinaryIO, cast

from shadowspill.checkpoint.file_digest import checksum_cache
from shadowspill.diagnostics.progress import Progress, phase
from shadowspill.orchestration.capture.progress import write_progress
from shadowspill.orchestration.launch.file_grants import resolve_file, verify_file
from shadowspill.training.definitions import InitializationRuntime, load_registry
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.preparation.bundle_validation import bind_bundle
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
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
    with phase("Load installed worker definitions"):
        registry = load_registry(launch.packages)
    with phase("Verify and decode saved Plans"):
        bundle = TrainingBundle.load(verify_file(files, launch.startup.bundle))
    # Only the owning controller can send WorkerLaunch on this inherited socket.
    # It has already rechecked this exact immutable artifact. Resolve local source
    # identities and bindings, without re-deriving IR or simulating every schedule.
    with phase("Bind installed sources and worker configuration"):
        inventory = capture_inventory(
            prepare_layouts(resolve_training(registry, config=bundle.config))
        )
        prepared = PreparedWorker.resolve(
            bind_bundle(bundle, inventory),
            launch.startup.scope,
            bundle=launch.startup.bundle,
            rank_map=dict(enumerate(launch.startup.logical_by_worker)),
            restore=launch.restore,
            weights=launch.weights,
            initial_weights=launch.initial_weights,
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
        if initial is not None
        and initial.kind == "artifact"
        and prepared.initialization is not None
        and prepared.initialization.context.temporary is None
        else set()
    ):
        raise ValueError("worker weight grants differ from its initialization")
    if launch.restore is not None:
        verify_file(files, launch.restore.rank.archive)
    if launch.initial_weights is not None:
        verify_file(files, launch.initial_weights.rank.archive)

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
            prepared.traces,
            bundle=prepared.startup.bundle,
            root=outputs,
            plans=prepared.bindings.prepared.plans[prepared.startup.scope.rank],
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
    parser.add_argument("--checksum-cache", type=Path)
    parser.add_argument("--nsys-progress", type=Path)
    args = parser.parse_args()
    if args.control_fd < 3:
        raise ValueError("worker control requires an inherited private descriptor")
    with (
        Progress(lambda message: print(message, flush=True)).activate(),
        checksum_cache(args.checksum_cache),
        socket.socket(fileno=args.control_fd) as control,
    ):
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
